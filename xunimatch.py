import argparse
import logging
import os
import pprint

import torch
import torch.nn.functional as F
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

# === 1. GradCAM Hook và compute_heatmap ===
class GradCAMHook:
    def __init__(self):
        self.activations = []
        self.gradients = []

    def forward_hook(self, module, inp, outp):
        self.activations.append(outp.detach())

    def backward_hook(self, module, grad_in, grad_out):
        self.gradients.append(grad_out[0].detach())


def compute_gradcam(activations, gradients):
    if not activations or not gradients:
        return None
    activation = activations[-1]   # [B, C, H, W]
    gradient = gradients[-1]       # [B, C, H, W]
    weights = gradient.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
    cam = (weights * activation).sum(dim=1)            # [B, H, W]
    cam = F.relu(cam)
    return cam.cpu().numpy()


def compute_heatmap_maps(model, hook, imgs, mode):
    """
    Trả về tensor [B, H, W] đã normalize trong [0,1].
    mode: 'false' => need_fp=False, 'true' => need_fp=True
    """
    device = imgs.device
    # forward để lấy logits
    if mode == 'false':
        outs = model(imgs)
    else:
        outs = model(imgs, True)[1]

    probs = outs.softmax(dim=1)    # [B, C, H, W]
    preds = probs.argmax(dim=1)    # [B, H, W]

    cams = []
    for b in range(imgs.size(0)):
        model.zero_grad()
        hook.activations.clear()
        hook.gradients.clear()
        # backprop sample b
        outs[b, preds[b]].sum().backward(retain_graph=True)
        cam_np = compute_gradcam(hook.activations, hook.gradients)  # (H, W)
        if cam_np is None:
            h, w = imgs.shape[2], imgs.shape[3]
            cam_t = torch.zeros(h, w, device=device)
        else:
            cam = torch.from_numpy(cam_np).to(device)
            cam_t = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cams.append(cam_t)
    return torch.stack(cams, dim=0)  # [B, H, W]

# Định nghĩa các tham số cần thiết
parser = argparse.ArgumentParser(
    description='Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--labeled-id-path', type=str, required=True)
parser.add_argument('--unlabeled-id-path', type=str, required=True)
parser.add_argument('--save-path', type=str, required=True)


def main():
    args = parser.parse_args()
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    # Logger và TensorBoard
    logger = init_log('global', logging.INFO)
    logger.propagate = 0
    writer = SummaryWriter(args.save_path)
    os.makedirs(args.save_path, exist_ok=True)
    rank = 0
    world_size = 1

    # In cấu hình
    all_args = {**cfg, **vars(args), 'ngpus': world_size}
    logger.info(f"{pprint.pformat(all_args)}\n")

    cudnn.benchmark = True
    cudnn.enabled = True

    # Model và optimizer
    model = DeepLabV3Plus(cfg)
    optimizer = SGD([
        {'params': model.backbone.parameters(), 'lr': cfg['lr']},
        {'params': [p for n, p in model.named_parameters() if 'backbone' not in n],
         'lr': cfg['lr'] * cfg['lr_multi']}
    ], lr=cfg['lr'], momentum=0.9, weight_decay=1e-4)

    logger.info(f"Total params: {count_params(model):.1f}M\n")
    model.cuda()

    # Đăng ký GradCAM hook
    hook = GradCAMHook()
    target_layer = model.reduce[1]
    target_layer.register_forward_hook(hook.forward_hook)
    target_layer.register_backward_hook(hook.backward_hook)

    # Loss functions
    if cfg['criterion']['name'] == 'CELoss':
        criterion_l = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).cuda()
    elif cfg['criterion']['name'] == 'OHEM':
        criterion_l = ProbOhemCrossEntropy2d(**cfg['criterion']['kwargs']).cuda()
    else:
        raise NotImplementedError(f"{cfg['criterion']['name']} not implemented")
    criterion_u = nn.CrossEntropyLoss(reduction='none').cuda()

    # DataLoader
    train_u = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_u', cfg['crop_size'], args.unlabeled_id_path)
    train_l = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_l', cfg['crop_size'], args.labeled_id_path, nsample=len(train_u.ids))
    valset = SemiDataset(cfg['dataset'], cfg['data_root'], 'val')

    trainloader_l = DataLoader(train_l, batch_size=cfg['batch_size'], pin_memory=True, num_workers=1, drop_last=True, shuffle=True)
    trainloader_u = DataLoader(train_u, batch_size=cfg['batch_size'], pin_memory=True, num_workers=1, drop_last=True, shuffle=True)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1, drop_last=False)

    total_iters = len(trainloader_u) * cfg['epochs']
    previous_best = 0.0
    start_epoch = 0

    # Load checkpoint nếu có
    ck_path = os.path.join(args.save_path, 'latest.pth')
    if os.path.exists(ck_path):
        ck = torch.load(ck_path)
        model.load_state_dict(ck['model'])
        optimizer.load_state_dict(ck['optimizer'])
        start_epoch = ck['epoch'] + 1
        previous_best = ck['previous_best']
        logger.info(f"Load checkpoint at epoch {start_epoch}\n")

    # Training loop
    for epoch in range(start_epoch, cfg['epochs']):
        if rank == 0:
            lr = optimizer.param_groups[0]['lr']
            logger.info(f"=====> Epoch: {epoch}, LR: {lr:.5f}, Best: {previous_best:.2f}")

        total_loss = AverageMeter()
        total_loss_x = AverageMeter()
        total_loss_s = AverageMeter()
        total_loss_w_fp = AverageMeter()
        total_loss_hm = AverageMeter()
        total_mask_ratio = AverageMeter()

        loader = zip(trainloader_l, trainloader_u, trainloader_u)
        for i, ((img_x, mask_x),
                (img_u_w, img_u_s1, img_u_s2, ignore_mask, cutmix_box1, cutmix_box2),
                (img_u_w_mix, img_u_s1_mix, img_u_s2_mix, ignore_mask_mix, _, _)) in enumerate(loader):

            # to device
            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            img_u_w = img_u_w.cuda()
            img_u_s1, img_u_s2 = img_u_s1.cuda(), img_u_s2.cuda()
            ignore_mask = ignore_mask.cuda()
            cutmix_box1, cutmix_box2 = cutmix_box1.cuda(), cutmix_box2.cuda()
            img_u_w_mix = img_u_w_mix.cuda()
            img_u_s1_mix, img_u_s2_mix = img_u_s1_mix.cuda(), img_u_s2_mix.cuda()
            ignore_mask_mix = ignore_mask_mix.cuda()

            # pseudo-label mix
            with torch.no_grad():
                model.eval()
                pu_mix = model(img_u_w_mix).detach()
                conf_mix = pu_mix.softmax(1).max(1)[0]
                mask_mix = pu_mix.argmax(1)

            # cutmix s1, s2
            mask1 = cutmix_box1.unsqueeze(1).expand_as(img_u_s1) == 1
            mask2 = cutmix_box2.unsqueeze(1).expand_as(img_u_s2) == 1
            img_u_s1[mask1] = img_u_s1_mix[mask1]
            img_u_s2[mask2] = img_u_s2_mix[mask2]

            model.train()
            nbx, nbu = img_x.size(0), img_u_w.size(0)

            preds, preds_fp = model(torch.cat((img_x, img_u_w)), True)
            pred_x, pred_u_w = preds.split([nbx, nbu])
            pred_u_w_fp = preds_fp[nbx:]

            pred_s1, pred_s2 = model(torch.cat((img_u_s1, img_u_s2))).chunk(2)

            # compute pseudo labels
            with torch.no_grad():
                p_u = pred_u_w.detach()
                conf_u = p_u.softmax(1).max(1)[0]
                mask_u = p_u.argmax(1)

            # apply mix on mask/conf
            mix1 = cutmix_box1 == 1
            mix2 = cutmix_box2 == 1
            m1, c1, ig1 = mask_u.clone(), conf_u.clone(), ignore_mask.clone()
            m2, c2, ig2 = mask_u.clone(), conf_u.clone(), ignore_mask.clone()
            m1[mix1] = mask_mix[mix1]; c1[mix1] = conf_mix[mix1]; ig1[mix1] = ignore_mask_mix[mix1]
            m2[mix2] = mask_mix[mix2]; c2[mix2] = conf_mix[mix2]; ig2[mix2] = ignore_mask_mix[mix2]

            # losses
            loss_x = criterion_l(pred_x, mask_x)

            loss_s1 = criterion_u(pred_s1, m1)
            mask_s1 = (c1 >= cfg['conf_thresh']) & (ig1 != 255)
            loss_s1 = (loss_s1 * mask_s1).sum() / (mask_s1.sum().float() + 1e-6)

            loss_s2 = criterion_u(pred_s2, m2)
            mask_s2 = (c2 >= cfg['conf_thresh']) & (ig2 != 255)
            loss_s2 = (loss_s2 * mask_s2).sum() / (mask_s2.sum().float() + 1e-6)

            loss_w_fp = criterion_u(pred_u_w_fp, mask_u)
            mask_w = (conf_u >= cfg['conf_thresh']) & (ignore_mask != 255)
            loss_w_fp = (loss_w_fp * mask_w).sum() / (mask_w.sum().float() + 1e-6)

            # base loss
            base = (loss_x + 0.25 * loss_s1 + 0.25 * loss_s2 + 0.5 * loss_w_fp) / 2.0

            # heatmap mse loss
            hm_false = compute_heatmap_maps(model, hook, img_u_w, 'false')
            hm_true = compute_heatmap_maps(model, hook, img_u_w, 'true')
            loss_hm = F.mse_loss(hm_true, hm_false)
            alpha = 0.2

            loss = base + alpha * loss_hm

            # backward & update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update meters
            total_loss.update(loss.item())
            total_loss_x.update(loss_x.item())
            total_loss_s.update((loss_s1.item() + loss_s2.item()) / 2.0)
            total_loss_w_fp.update(loss_w_fp.item())
            total_loss_hm.update(loss_hm.item())
            mask_ratio = mask_w.float().mean().item()
            total_mask_ratio.update(mask_ratio)

            iters = epoch * len(trainloader_u) + i
            writer.add_scalar('train/loss_all', total_loss.avg, iters)
            writer.add_scalar('train/loss_x', total_loss_x.avg, iters)
            writer.add_scalar('train/loss_s', total_loss_s.avg, iters)
            writer.add_scalar('train/loss_w_fp', total_loss_w_fp.avg, iters)
            writer.add_scalar('train/loss_hm', total_loss_hm.avg, iters)
            writer.add_scalar('train/mask_ratio', total_mask_ratio.avg, iters)

            if i % (len(trainloader_u) // 8) == 0:
                logger.info(f"Iters: {i}, total={total_loss.avg:.3f}, x={total_loss_x.avg:.3f}, s={total_loss_s.avg:.3f}, w_fp={total_loss_w_fp.avg:.3f}, hm={total_loss_hm.avg:.3f}, mask_ratio={total_mask_ratio.avg:.3f}")

        # validation và checkpoint
        eval_mode = 'sliding_window' if cfg['dataset'] == 'cityscapes' else 'original'
        mIoU, iou_class = evaluate(model, valloader, eval_mode, cfg)
        logger.info(f"Evaluation {eval_mode} >>>> MeanIoU: {mIoU:.2f}")
        for idx, iou in enumerate(iou_class):
            logger.info(f" Class [{idx} {CLASSES[cfg['dataset']][idx]}] IoU: {iou:.2f}")
        writer.add_scalar('eval/mIoU', mIoU, epoch)

        checkpoint = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'previous_best': max(mIoU, previous_best)}
        torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
        if mIoU > previous_best:
            torch.save(checkpoint, os.path.join(args.save_path, 'best.pth'))
            previous_best = mIoU

if __name__ == '__main__':
    main()