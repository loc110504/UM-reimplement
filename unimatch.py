import argparse
import logging
import os
import pprint

import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

from dataset.semi import SemiDataset
from model.semseg.deeplabv3plus import DeepLabV3Plus
from supervised import evaluate
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, init_log, AverageMeter

# Định nghĩa các tham số cần thiết
parser = argparse.ArgumentParser(
    description='Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--labeled-id-path', type=str, required=True)
parser.add_argument('--unlabeled-id-path', type=str, required=True)
parser.add_argument('--save-path', type=str, required=True)

def main():
    args = parser.parse_args()
    # Load cấu hình từ file YAML (bao gồm các thông số như learning rate, batch_size, crop_size, ...)
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    # Khởi tạo logger và SummaryWriter cho TensorBoard
    logger = init_log('global', logging.INFO)
    logger.propagate = 0
    writer = SummaryWriter(args.save_path)
    os.makedirs(args.save_path, exist_ok=True)
    rank = 0
    world_size = 1

    # In ra cấu hình tổng hợp
    all_args = {**cfg, **vars(args), 'ngpus': world_size}
    logger.info('{}\n'.format(pprint.pformat(all_args)))

    # Tăng tốc độ xử lý với GPU (phù hợp cho input cố định: segmentation, classification)
    cudnn.benchmark = True
    cudnn.enabled = True

    # Khởi tạo mô hình DeepLabV3+ và optimizer với 2 nhóm tham số 
    model = DeepLabV3Plus(cfg)
    optimizer = SGD([{'params': model.backbone.parameters(), 'lr': cfg['lr']},
                    {'params': [param for name, param in model.named_parameters() if 'backbone' not in name],
                     'lr': cfg['lr'] * cfg['lr_multi']}],
                    lr=cfg['lr'], momentum=0.9, weight_decay=1e-4)
    
    logger.info('Total params: {:.1f}M\n'.format(count_params(model)))
    # Chuyển mô hình và optimizer sang GPU
    model.cuda()

    # Chọn loss function cho labeled data
    if cfg['criterion']['name'] == 'CELoss':
        criterion_l = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).cuda()
    elif cfg['criterion']['name'] == 'OHEM':
        criterion_l = ProbOhemCrossEntropy2d(**cfg['criterion']['kwargs']).cuda()
    else:
        raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])
    
    # Loss cho unlabeled data
    criterion_u = nn.CrossEntropyLoss(reduction='none').cuda()

    # Tạo dataset và DataLoader cho:
    # - Dữ liệu không nhãn (train_u)
    # - Dữ liệu có nhãn (train_l) với số mẫu bằng số file của train_u
    # - Tập validation (val)
    trainset_u = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_u',
                            cfg['crop_size'], args.unlabeled_id_path)

    trainset_l = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_l',
                             cfg['crop_size'], args.labeled_id_path, nsample=len(trainset_u.ids))
    valset = SemiDataset(cfg['dataset'], cfg['data_root'], 'val')

    # Sử dụng DataLoader thông thường với shuffle
    trainloader_l = DataLoader(trainset_l, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=1, drop_last=True, shuffle=True)
    trainloader_u = DataLoader(trainset_u, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=1, drop_last=True, shuffle=True)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1, drop_last=False)

    total_iters = len(trainloader_u) * cfg['epochs']
    previous_best = 0.0
    start_epoch = 0
    
    # Kiểm tra nếu đã có checkpoint lưu trước đó, nạp lại trạng thái của mô hình và optimizer
    checkpoint_path = os.path.join(args.save_path, 'latest.pth')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        previous_best = checkpoint['previous_best']
        
        if rank == 0:
            logger.info('************ Load from checkpoint at epoch %i\n' % start_epoch)

    # Vòng lặp huấn luyện theo epoch
    for epoch in range(start_epoch, cfg['epochs']):
        if rank == 0:
            logger.info('===========> Epoch: {:}, LR: {:.5f}, Previous best: {:.2f}'.format(
                epoch, optimizer.param_groups[0]['lr'], previous_best))

        # Khởi tạo các bộ đếm trung bình cho loss
        total_loss = AverageMeter()
        total_loss_x = AverageMeter()
        total_loss_s = AverageMeter()
        total_loss_w_fp = AverageMeter()
        total_mask_ratio = AverageMeter()

        # Lấy đồng thời batch từ dữ liệu có nhãn và hai batch từ dữ liệu không nhãn
        # Batch thứ nhất của unlabeled dùng cho loss chính, batch thứ hai dùng cho cutmix
        loader = zip(trainloader_l, trainloader_u, trainloader_u)

        for i, ((img_x, mask_x),
                (img_u_w, img_u_s1, img_u_s2, ignore_mask, cutmix_box1, cutmix_box2),
                (img_u_w_mix, img_u_s1_mix, img_u_s2_mix, ignore_mask_mix, _, _)) in enumerate(loader):

            # Chuyển dữ liệu vào GPU
            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            img_u_w = img_u_w.cuda()
            img_u_s1, img_u_s2, ignore_mask = img_u_s1.cuda(), img_u_s2.cuda(), ignore_mask.cuda()
            cutmix_box1, cutmix_box2 = cutmix_box1.cuda(), cutmix_box2.cuda()
            img_u_w_mix = img_u_w_mix.cuda()
            img_u_s1_mix, img_u_s2_mix = img_u_s1_mix.cuda(), img_u_s2_mix.cuda()
            ignore_mask_mix = ignore_mask_mix.cuda()

            # Dự đoán trên ảnh không nhãn mix để lấy pseudo-label và độ tin cậy
            with torch.no_grad():
                model.eval()
                pred_u_w_mix = model(img_u_w_mix).detach()
                conf_u_w_mix = pred_u_w_mix.softmax(dim=1).max(dim=1)[0]
                mask_u_w_mix = pred_u_w_mix.argmax(dim=1)

            # Áp dụng cutmix: thay thế vùng được đánh dấu trong ảnh không nhãn gốc
            img_u_s1[cutmix_box1.unsqueeze(1).expand_as(img_u_s1) == 1] = \
                img_u_s1_mix[cutmix_box1.unsqueeze(1).expand_as(img_u_s1) == 1]
            img_u_s2[cutmix_box2.unsqueeze(1).expand_as(img_u_s2) == 1] = \
                img_u_s2_mix[cutmix_box2.unsqueeze(1).expand_as(img_u_s2) == 1]

            model.train()

            num_lb, num_ulb = img_x.shape[0], img_u_w.shape[0]
            
            # Ghép ảnh có nhãn và không nhãn, yêu cầu mô hình trả về 2 đầu ra:
            # - preds: dự đoán chính cho cả dữ liệu có nhãn và không nhãn.
            # - preds_fp: dự đoán phụ (cho phần pseudo-label weakly) của dữ liệu không nhãn.
            preds, preds_fp = model(torch.cat((img_x, img_u_w)), True)
            pred_x, pred_u_w = preds.split([num_lb, num_ulb])
            pred_u_w_fp = preds_fp[num_lb:]

            # Dự đoán từ hai ảnh biến đổi (augmentation) của dữ liệu không nhãn
            pred_u_s1, pred_u_s2 = model(torch.cat((img_u_s1, img_u_s2))).chunk(2)

            # Tính pseudo-label và độ tin cậy cho dữ liệu không nhãn
            pred_u_w = pred_u_w.detach()
            conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]
            mask_u_w = pred_u_w.argmax(dim=1)

            # Tạo bản sao để áp dụng cutmix cho pseudo-label và độ tin cậy
            mask_u_w_cutmixed1, conf_u_w_cutmixed1, ignore_mask_cutmixed1 = \
                mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()
            mask_u_w_cutmixed2, conf_u_w_cutmixed2, ignore_mask_cutmixed2 = \
                mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()

            mask_u_w_cutmixed1[cutmix_box1 == 1] = mask_u_w_mix[cutmix_box1 == 1]
            conf_u_w_cutmixed1[cutmix_box1 == 1] = conf_u_w_mix[cutmix_box1 == 1]
            ignore_mask_cutmixed1[cutmix_box1 == 1] = ignore_mask_mix[cutmix_box1 == 1]

            mask_u_w_cutmixed2[cutmix_box2 == 1] = mask_u_w_mix[cutmix_box2 == 1]
            conf_u_w_cutmixed2[cutmix_box2 == 1] = conf_u_w_mix[cutmix_box2 == 1]
            ignore_mask_cutmixed2[cutmix_box2 == 1] = ignore_mask_mix[cutmix_box2 == 1]
            
            # ===Tính loss cho dữ liệu có nhãn===
            loss_x = criterion_l(pred_x, mask_x)

            # ==Tính loss cho dữ liệu không nhãn từ 2 phiên bản (sử dụng pseudo-label và độ tin cậy)==
            loss_u_s1 = criterion_u(pred_u_s1, mask_u_w_cutmixed1)
            loss_u_s1 = loss_u_s1 * ((conf_u_w_cutmixed1 >= cfg['conf_thresh']) & (ignore_mask_cutmixed1 != 255))
            loss_u_s1 = loss_u_s1.sum() / ((ignore_mask_cutmixed1 != 255).sum().item() + 1e-6)

            loss_u_s2 = criterion_u(pred_u_s2, mask_u_w_cutmixed2)
            loss_u_s2 = loss_u_s2 * ((conf_u_w_cutmixed2 >= cfg['conf_thresh']) & (ignore_mask_cutmixed2 != 255))
            loss_u_s2 = loss_u_s2.sum() / ((ignore_mask_cutmixed2 != 255).sum().item() + 1e-6)
            
            # ===Tính loss cho phần dự đoán phụ từ pseudo-label của dữ liệu không nhãn===
            loss_u_w_fp = criterion_u(pred_u_w_fp, mask_u_w)
            loss_u_w_fp = loss_u_w_fp * ((conf_u_w >= cfg['conf_thresh']) & (ignore_mask != 255))
            loss_u_w_fp = loss_u_w_fp.sum() / ((ignore_mask != 255).sum().item() + 1e-6)

            # ===Tổng hợp loss theo trọng số===
            loss = (loss_x + loss_u_s1 * 0.25 + loss_u_s2 * 0.25 + loss_u_w_fp * 0.5) / 2.0
            
            # Backpropagation và cập nhật trọng số
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Cập nhật các chỉ số trung bình để logging
            total_loss.update(loss.item())
            total_loss_x.update(loss_x.item())
            total_loss_s.update((loss_u_s1.item() + loss_u_s2.item()) / 2.0)
            total_loss_w_fp.update(loss_u_w_fp.item())
            
            mask_ratio = ((conf_u_w >= cfg['conf_thresh']) & (ignore_mask != 255)).sum().item() / ((ignore_mask != 255).sum().item() + 1e-6)
            total_mask_ratio.update(mask_ratio)

            # Cập nhật learning rate theo schedule
            iters = epoch * len(trainloader_u) + i
            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']

            # Ghi log loss và các chỉ số qua TensorBoard
            writer.add_scalar('train/loss_all', loss.item(), iters)
            writer.add_scalar('train/loss_x', loss_x.item(), iters)
            writer.add_scalar('train/loss_s', (loss_u_s1.item() + loss_u_s2.item()) / 2.0, iters)
            writer.add_scalar('train/loss_w_fp', loss_u_w_fp.item(), iters)
            writer.add_scalar('train/mask_ratio', mask_ratio, iters)
                
            if (i % (len(trainloader_u) // 8) == 0):
                logger.info('Iters: {:}, Total loss: {:.3f}, Loss x: {:.3f}, Loss s: {:.3f}, Loss w_fp: {:.3f}, Mask ratio: {:.3f}'.format(
                    i, total_loss.avg, total_loss_x.avg, total_loss_s.avg,
                    total_loss_w_fp.avg, total_mask_ratio.avg))

        # Sau mỗi epoch, đánh giá mô hình trên tập validation
        eval_mode = 'sliding_window' if cfg['dataset'] == 'cityscapes' else 'original'
        mIoU, iou_class = evaluate(model, valloader, eval_mode, cfg)
        logger.info('***** Evaluation {} ***** >>>> MeanIoU: {:.2f}'.format(eval_mode, mIoU))

        for (cls_idx, iou) in enumerate(iou_class):
            logger.info('***** Evaluation ***** >>>> Class [{} {}] IoU: {:.2f}'.format(
                cls_idx, CLASSES[cfg['dataset']][cls_idx], iou))
            
        writer.add_scalar('eval/mIoU', mIoU, epoch)
        for i, iou in enumerate(iou_class):
            writer.add_scalar('eval/{}_IoU'.format(CLASSES[cfg['dataset']][i]), iou, epoch)

        # Lưu checkpoint của mô hình sau mỗi epoch
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'previous_best': max(mIoU, previous_best),
        }
        torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
        if mIoU > previous_best:
            torch.save(checkpoint, os.path.join(args.save_path, 'best.pth'))
            previous_best = mIoU
            
if __name__ == '__main__':
    main()
