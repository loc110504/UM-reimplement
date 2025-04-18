#!/usr/bin/env python
# coding: utf-8

import argparse
import logging
import os
import pprint
import yaml

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset.semi import SemiDataset
from model.semseg.deeplabv3plus import DeepLabV3Plus
from supervised import evaluate
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, init_log, AverageMeter


# === 1. GradCAM Hook và compute_gradcam ===

class GradCAMHook:
    def __init__(self):
        self.activations = []
        self.gradients = []

    def forward_hook(self, module, input, output):
        self.activations.append(output.detach())

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients.append(grad_output[0].detach())


def compute_gradcam(activations, gradients):
    if not activations or not gradients:
        return None
    activation = activations[-1]   # [B, C, H, W]
    gradient   = gradients[-1]     # [B, C, H, W]
    weights = gradient.mean(dim=(2,3), keepdim=True)    # [B, C, 1, 1]
    cam = (weights * activation).sum(dim=1)              # [B, H, W]
    cam = F.relu(cam)
    return cam.cpu().numpy()  # will normalize later


def compute_heatmap_maps(model, hook, images, mode, device):
    """
    Trả về tensor [B, H, W] gồm heatmap đã normalize [0,1].
    mode: 'false' -> need_fp=False, 'true' -> need_fp=True
    """
    bsz = images.size(0)
    # forward để lấy logits
    if mode == 'false':
        outs = model(images)               # [B, C, H, W]
    else:
        outs = model(images, True)[1]      # preds_fp: [B, C, H, W]

    probs = outs.softmax(dim=1)           # [B, C, H, W]
    preds = probs.argmax(dim=1)           # [B, H, W]

    cams = []
    for b in range(bsz):
        model.zero_grad()
        hook.activations.clear()
        hook.gradients.clear()

        # backprop cho sample b và class preds[b]
        outs[b, preds[b]].sum().backward(retain_graph=True)
        cam_np = compute_gradcam(hook.activations, hook.gradients)  # (H, W) numpy
        if cam_np is None:
            h, w = images.size(2), images.size(3)
            cam_t = torch.zeros(h, w, device=device)
        else:
            cam = torch.from_numpy(cam_np).to(device)
            cam_t = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cams.append(cam_t)
    return torch.stack(cams, dim=0)  # [B, H, W]


# === 2. Main training ===

def main():
    parser = argparse.ArgumentParser(
        description='Unimatch + Heatmap MSE Loss for Semi-Supervised Segmentation')
    parser.add_argument('--config',          type=str, required=True)
    parser.add_argument('--labeled-id-path', type=str, required=True)
    parser.add_argument('--unlabeled-id-path', type=str, required=True)
    parser.add_argument('--save-path',       type=str, required=True)
    args = parser.parse_args()

    # Load config
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    # Logger & TensorBoard
    logger = init_log('global', logging.INFO)
    logger.propagate = 0
    writer = SummaryWriter(args.save_path)
    os.makedirs(args.save_path, exist_ok=True)

    # In config
    all_args = {**cfg, **vars(args)}
    logger.info('{}\n'.format(pprint.pformat(all_args)))

    cudnn.benchmark = True
    cudnn.enabled   = True

    # Model + optimizer
    model = DeepLabV3Plus(cfg)
    optimizer = SGD([
            {'params': model.backbone.parameters(), 'lr': cfg['lr']},
            {'params': [p for n,p in model.named_parameters() if 'backbone' not in n],
             'lr': cfg['lr'] * cfg.get('lr_multi',1.0)}
        ], lr=cfg['lr'], momentum=0.9, weight_decay=1e-4)

    logger.info('Total params: {:.1f}M'.format(count_params(model)/1e6))
    model.cuda()

    # === 2.1 Đăng ký GradCAM hook ===
    hook = GradCAMHook()
    target_layer = model.reduce[1]
    target_layer.register_forward_hook(hook.forward_hook)
    target_layer.register_backward_hook(hook.backward_hook)

    # Các loss
    if cfg['criterion']['name'] == 'CELoss':
        criterion_l = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).cuda()
    elif cfg['criterion']['name'] == 'OHEM':
        criterion_l = ProbOhemCrossEntropy2d(**cfg['criterion']['kwargs']).cuda()
    else:
        raise NotImplementedError
    criterion_u = nn.CrossEntropyLoss(reduction='none').cuda()

    # Datasets & loaders
    train_u = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_u', cfg['crop_size'], args.unlabeled_id_path)
    train_l = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_l', cfg['crop_size'], args.labeled_id_path, nsample=len(train_u.ids))
    valset  = SemiDataset(cfg['dataset'], cfg['data_root'], 'val')

    loader_l = DataLoader(train_l, batch_size=cfg['batch_size'], shuffle=True,  pin_memory=True, num_workers=1, drop_last=True)
    loader_u = DataLoader(train_u, batch_size=cfg['batch_size'], shuffle=True,  pin_memory=True, num_workers=1, drop_last=True)
    valloader = DataLoader(valset, batch_size=1, shuffle=False, pin_memory=True, num_workers=1)

    total_iters   = len(loader_u) * cfg['epochs']
    previous_best = 0.0
    start_epoch   = 0
    latest_ckpt   = os.path.join(args.save_path, 'latest.pth')

    # Resume nếu có ckpt
    if os.path.exists(latest_ckpt):
        ck = torch.load(latest_ckpt)
        model.load_state_dict(ck['model'])
        optimizer.load_state_dict(ck['optimizer'])
        start_epoch   = ck['epoch'] + 1
        previous_best = ck['previous_best']
        logger.info(f"Resumed from epoch {start_epoch}")

    # === 2.2 Vòng epoch ===
    for epoch in range(start_epoch, cfg['epochs']):
        logger.info(f"Epoch {epoch+1}/{cfg['epochs']} - LR: {optimizer.param_groups[0]['lr']:.6f}")
        model.train()

        meters = {
            'loss_all':   AverageMeter(),
            'loss_x':     AverageMeter(),
            'loss_s':     AverageMeter(),
            'loss_w_fp':  AverageMeter(),
            'loss_hm':    AverageMeter(),
            'mask_ratio': AverageMeter(),
        }

        loader = zip(loader_l, loader_u, loader_u)
        for i, ((img_x, mask_x),
                (img_u_w, img_u_s1, img_u_s2, ignore_mask, cm1, cm2),
                (img_u_w2, img_u_s1_m, img_u_s2_m, ignore_mask_m, _, _)) in enumerate(loader):

            # Đưa lên GPU
            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            img_u_w = img_u_w.cuda()
            img_u_s1, img_u_s2 = img_u_s1.cuda(), img_u_s2.cuda()
            ignore_mask = ignore_mask.cuda()
            cm1, cm2 = cm1.cuda(), cm2.cuda()
            img_u_w2 = img_u_w2.cuda()
            img_u_s1_m, img_u_s2_m = img_u_s1_m.cuda(), img_u_s2_m.cuda()
            ignore_mask_m = ignore_mask_m.cuda()

            # Tạo pseudo-label từ mix
            with torch.no_grad():
                model.eval()
                pu_w_mix = model(img_u_w2).detach()
                cu_conf  = pu_w_mix.softmax(dim=1).max(dim=1)[0]
                cu_mask  = pu_w_mix.argmax(dim=1)

            # Áp cutmix lên s1, s2
            img_u_s1[cm1==1] = img_u_s1_m[cm1==1]
            img_u_s2[cm2==1] = img_u_s2_m[cm2==1]

            model.train()
            nbx, nbu = img_x.size(0), img_u_w.size(0)

            # Forward chính và phụ
            preds, preds_fp = model(torch.cat([img_x, img_u_w]), True)
            pred_x, pred_u_w = preds.split([nbx, nbu])
            pred_w_fp       = preds_fp[nbx:]

            # Forward s1, s2
            pred_s1, pred_s2 = model(torch.cat([img_u_s1, img_u_s2])).chunk(2)

            # Pseudo-label & conf mask
            with torch.no_grad():
                p_u_w = pred_u_w.detach()
                conf_u = p_u_w.softmax(dim=1).max(dim=1)[0]
                m_u    = p_u_w.argmax(dim=1)

            # Cutmix mask/conf
            m1, c1, ig1 = m_u.clone(), conf_u.clone(), ignore_mask.clone()
            m2, c2, ig2 = m_u.clone(), conf_u.clone(), ignore_mask.clone()
            m1[cm1==1] = cu_mask[cm1==1]; c1[cm1==1] = cu_conf[cm1==1]; ig1[cm1==1] = ignore_mask_m[cm1==1]
            m2[cm2==1] = cu_mask[cm2==1]; c2[cm2==1] = cu_conf[cm2==1]; ig2[cm2==1] = ignore_mask_m[cm2==1]

            # === Loss labeled ===
            loss_x = criterion_l(pred_x, mask_x)

            # === Loss s1, s2 ===
            l_s1 = criterion_u(pred_s1, m1)
            mask1 = (c1>=cfg['conf_thresh']) & (ig1!=255)
            l_s1 = (l_s1 * mask1).sum() / (mask1.sum().float()+1e-6)
            l_s2 = criterion_u(pred_s2, m2)
            mask2 = (c2>=cfg['conf_thresh']) & (ig2!=255)
            l_s2 = (l_s2 * mask2).sum() / (mask2.sum().float()+1e-6)

            # === Loss auxiliary ===
            l_fp = criterion_u(pred_w_fp, m_u)
            mask_w = (conf_u>=cfg['conf_thresh']) & (ignore_mask!=255)
            l_fp = (l_fp * mask_w).sum() / (mask_w.sum().float()+1e-6)

            # === Heatmap MSE Loss ===
            hm_false = compute_heatmap_maps(model, hook, img_u_w, 'false', img_u_w.device)
            hm_true  = compute_heatmap_maps(model, hook, img_u_w, 'true',  img_u_w.device)
            loss_hm  = F.mse_loss(hm_false, hm_true)
            alpha    = 0.2

            # === Tổng hợp ===
            base = (loss_x + 0.25*(l_s1 + l_s2) + 0.5 * l_fp) / 2.0
            loss = base + alpha * loss_hm

            # backward & step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update meters & TB
            iters = epoch * len(loader_u) + i
            meters['loss_all'].update(loss.item())
            meters['loss_x'].update(loss_x.item())
            meters['loss_s'].update(0.5*(l_s1.item()+l_s2.item()))
            meters['loss_w_fp'].update(l_fp.item())
            meters['loss_hm'].update(loss_hm.item())
            meters['mask_ratio'].update(mask_w.float().mean().item())

            writer.add_scalar('train/loss_all',    meters['loss_all'].avg, iters)
            writer.add_scalar('train/loss_x',      meters['loss_x'].avg, iters)
            writer.add_scalar('train/loss_s',      meters['loss_s'].avg, iters)
            writer.add_scalar('train/loss_w_fp',   meters['loss_w_fp'].avg, iters)
            writer.add_scalar('train/loss_hm',     meters['loss_hm'].avg, iters)
            writer.add_scalar('train/mask_ratio',  meters['mask_ratio'].avg, iters)

            if i % (len(loader_u)//8) == 0:
                logger.info(
                    f"Iters {i}/{len(loader_u)}: loss={meters['loss_all'].avg:.4f}, "
                    f"x={meters['loss_x'].avg:.4f}, s={meters['loss_s'].avg:.4f}, "
                    f"fp={meters['loss_w_fp'].avg:.4f}, hm={meters['loss_hm'].avg:.4f}, "
                    f"mask_ratio={meters['mask_ratio'].avg:.3f}"
                )

        # === Validation & checkpoint ===
        model.eval()
        mode = 'sliding_window' if cfg['dataset']=='cityscapes' else 'original'
        mIoU, iou_cls = evaluate(model, valloader, mode, cfg)
        logger.info(f"Eval mIoU: {mIoU:.2f}")
        writer.add_scalar('eval/mIoU', mIoU, epoch)
        for idx, iou in enumerate(iou_cls):
            logger.info(f"  Class {idx} [{CLASSES[cfg['dataset']][idx]}]: {iou:.2f}")

        ck = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'previous_best': max(previous_best, mIoU)
        }
        torch.save(ck, latest_ckpt)
        if mIoU > previous_best:
            torch.save(ck, os.path.join(args.save_path, 'best.pth'))
            previous_best = mIoU


if __name__ == '__main__':
    main()
