import torch
import numpy as np
from util.utils import AverageMeter, intersectionAndUnion
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

# evaluate function for validation
def evaluate(model, loader, mode, cfg):
    """
    Đánh giá mô hình segmentation trên tập validation.
    
    Args:
        model: mô hình segmentation (DeepLabV3+)
        loader: DataLoader chứa ảnh và mask validation
        mode: chế độ đánh giá, một trong ['original', 'center_crop', 'sliding_window']
        cfg: dictionary cấu hình từ file YAML

    Returns:
        mIoU: mean Intersection over Union
        iou_class: IoU cho từng lớp
    """
    model.eval() # đưa model vào chế độ đánh giá
    assert mode in ['original', 'center_crop', 'sliding_window'] # kiểm tra mode hợp lệ
    # Khởi tạo biến để lưu tổng Intersection và Union từng class
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()

    with torch.no_grad():
        for img, mask, id in loader:
            img = img.cuda()

            if mode == 'sliding_window':
                # Sliding window để dự đoán ảnh có kích thước lớn hơn crop_size
                grid = cfg['crop_size']
                b, _, h, w = img.shape
                # final: tensor chứa tổng các dự đoán cho mỗi pixel (cho voting)
                final = torch.zeros(b, cfg['nclass'], h, w).cuda()
                row = 0
                while row < h:
                    col = 0
                    while col < w:
                        # Crop từng phần của ảnh để dự đoán
                        pred = model(img[:, :, row: min(h, row + grid), col: min(w, col + grid)])
                        pred = pred.softmax(dim=1)
                        # Cộng dồn kết quả dự đoán vào final (sử dụng xác suất)
                        final[:, :, row:min(h, row + grid), col:min(w, col + grid)] += pred
                        col += int(grid * 2 / 3)
                    row += int(grid * 2 / 3)
                pred = final.argmax(dim=1) # lấy class có xác suất cao nhất tại mỗi pixel

            else:
                if mode == 'center_crop':
                    # Crop chính giữa ảnh (cho ảnh quá lớn so với crop size)
                    h, w = img.shape[-2:]
                    start_h, start_w = (h - cfg['crop_size']) // 2, (w - cfg['crop_size']) // 2
                    img = img[:, :, start_h:start_h + cfg['crop_size'], start_w:start_w + cfg['crop_size']]
                    mask = mask[:, start_h:start_h + cfg['crop_size'], start_w:start_w + cfg['crop_size']]
                # Dự đoán ảnh đầy đủ (original hoặc đã crop)
                pred = model(img).argmax(dim=1)
            # Tính IoU từng lớp
            intersection, union, target = intersectionAndUnion(pred.cpu().numpy(), mask.numpy(), cfg['nclass'], 255)

            # Cập nhật tổng số lượng pixel intersection & union cho từng class
            intersection_meter.update(intersection)
            union_meter.update(union)
    # Tính IoU theo class và trung bình (mean IoU)
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10) * 100.0
    mIOU = np.mean(iou_class)

    return mIOU, iou_class

# === Dummy config ===
cfg = {
    'crop_size': 64,
    'nclass': 3  # ví dụ: segmentation có 3 lớp
}

# === Dummy model: forward trả về random logits ===
class DummyModel(nn.Module):
    def __init__(self, nclass):
        super().__init__()
        self.conv = nn.Conv2d(3, nclass, kernel_size=1)  # chuyển 3 channel -> nclass

    def forward(self, x):
        # Trả về tensor shape (B, C, H, W) như model segmentation thật
        B, C, H, W = x.shape
        return torch.randn(B, cfg['nclass'], H, W).cuda()

# === Dummy dataset ===
class DummySegmentationDataset(Dataset):
    def __init__(self, num_samples=5, size=128):
        self.num_samples = num_samples
        self.size = size

    def __getitem__(self, index):
        # Tạo ảnh random và mask ngẫu nhiên với nhãn từ 0 đến nclass-1
        img = torch.rand(3, self.size, self.size)
        mask = torch.randint(0, cfg['nclass'], (self.size, self.size), dtype=torch.long)
        id = f"img_{index}"
        return img, mask, id

    def __len__(self):
        return self.num_samples

# test evaluate()
if __name__ == "__main__":
    # === Khởi tạo model và dataloader ===
    model = DummyModel(cfg['nclass']).cuda()
    dataset = DummySegmentationDataset()
    loader = DataLoader(dataset, batch_size=2)

    # === Chạy thử evaluate với từng mode ===
    for mode in ['original', 'center_crop', 'sliding_window']:
        print(f"\n=== Evaluate Mode: {mode} ===")
        mIoU, iou_class = evaluate(model, loader, mode, cfg)
        print(f"Mean IoU: {mIoU:.2f}")
        for i, iou in enumerate(iou_class):
            print(f"Class {i} IoU: {iou:.2f}")
