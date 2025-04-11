import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

class DeepLabV3PlusSMPPerturb(nn.Module):
    """
    Lớp wrapper cho DeepLabV3+ của segmentation_models_pytorch
    bổ sung khả năng perturbation theo tham số need_fp.
    
    Khi need_fp == True, forward sẽ:
      - Lấy các feature maps từ encoder.
      - Giả lập nhiễu bằng Dropout2d trên các feature của low-level (c1) và high-level (c4).
      - Nối features gốc và features nhiễu theo chiều batch (tăng gấp đôi batch_size).
      - Gọi decoder và segmentation head trên các features nối, sau đó tách kết quả ra thành 2 đầu:
            + out: dự đoán gốc.
            + out_fp: dự đoán từ nhánh perturbed.
      
    Khi need_fp == False, forward hoạt động như bình thường.
    """
    def __init__(self, encoder_name="resnet101", encoder_weights="imagenet", in_channels=3, classes=21):
        super(DeepLabV3PlusSMPPerturb, self).__init__()
        self.model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes
        )
        # Dropout2d với p=0.5 dùng để tạo nhiễu cho các feature maps
        self.dropout = nn.Dropout2d(p=0.5)
        
    def forward(self, x, need_fp=False):
        h, w = x.shape[-2:]
        if need_fp:
            # Lấy các feature maps từ encoder của SMP
            # SMP encoder thường trả về một list các feature maps từ các stage;
            # theo cách thường gặp, giả sử: c1 = features[0] (low-level), c4 = features[-1] (high-level)
            features = self.model.encoder(x)
            c1 = features[0]
            c4 = features[-1]
            
            # Tạo phiên bản nhiễu (perturbed) thông qua Dropout2d
            c1_pert = self.dropout(c1)
            c4_pert = self.dropout(c4)
            
            # Nối theo chiều batch: batch gốc và batch nhiễu => batch_size * 2
            c1_cat = torch.cat([c1, c1_pert], dim=0)
            c4_cat = torch.cat([c4, c4_pert], dim=0)
            
            # Tạo danh sách features mới bằng cách copy features ban đầu,
            # sau đó thay thế c1 và c4 bằng phiên bản nối được.
            features_new = list(features)
            features_new[0] = c1_cat
            features_new[-1] = c4_cat
            
            # Gọi decoder: trong SMP, decoder nhận đầu vào là các feature maps theo từng tầng
            decoder_output = self.model.decoder(*features_new)
            # Gọi segmentation head để dự đoán segmentation map
            seg_output = self.model.segmentation_head(decoder_output)
            # Upsample segmentation map về kích thước ban đầu của ảnh
            seg_output = F.interpolate(seg_output, size=(h, w), mode="bilinear", align_corners=False)
            
            # Vì batch đã gấp đôi, tách kết quả thành 2 phần (theo chiều batch=0):
            # phần đầu là output ban đầu, phần thứ hai là output từ nhánh perturbed.
            batch_size = x.shape[0]
            out, out_fp = seg_output.split(batch_size, dim=0)
            return out, out_fp
        else:
            # Forward bình thường: gọi mô hình SMP DeepLabV3+ mà không dùng perturbation
            seg_output = self.model(x)
            seg_output = F.interpolate(seg_output, size=(h, w), mode="bilinear", align_corners=False)
            return seg_output
