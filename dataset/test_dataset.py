import torch
from torch.utils.data import DataLoader
from semi import SemiDataset
from transform import normalize, resize, crop, hflip, blur, obtain_cutmix_box

if __name__ == '__main__':
    root_dir = '../pascal_data'

    # Khởi tạo dataset cho tập unlabeled và labeled
    trainset_u = SemiDataset('pascal', root_dir, 'train_u', 321, '../splits/pascal/1464/unlabeled.txt')
    trainset_l = SemiDataset('pascal', root_dir, 'train_l', 321, './splits/pascal/1464/labeled.txt', 1464)

    # Tạo DataLoader cho mỗi dataset
    dataloader_u = DataLoader(trainset_u, batch_size=4, shuffle=True, num_workers=0)
    dataloader_l = DataLoader(trainset_l, batch_size=4, shuffle=True, num_workers=0)

    # Kiểm tra một batch của tập unlabeled
    print("=== Unlabeled Dataset Batch ===")
    for batch in dataloader_u:
        # Ở mode 'train_u', __getitem__ trả về tuple:
        # (normalize(img_w), img_s1, img_s2, ignore_mask, cutmix_box1, cutmix_box2)
        img_w, img_s1, img_s2, ignore_mask, cutmix_box1, cutmix_box2 = batch
        print("img_w shape:", img_w.shape)
        print("img_s1 shape:", img_s1.shape)
        print("img_s2 shape:", img_s2.shape)
        print("ignore_mask shape:", ignore_mask.shape)
        print("cutmix_box1:", cutmix_box1)
        print("cutmix_box2:", cutmix_box2)
        break  # chỉ duyệt 1 batch để test

    # Kiểm tra một batch của tập labeled
    print("\n=== Labeled Dataset Batch ===")
    for batch in dataloader_l:
        # Ở mode 'train_l', __getitem__ trả về tuple: (img, mask)
        img, mask = batch
        print("img shape:", img.shape)
        print("mask shape:", mask.shape)
        break  # chỉ duyệt 1 batch để test
    

