from transformers import SegformerForSemanticSegmentation
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import tqdm
from range_image import range_image,spherical_projection
import os
import numpy as np
from velodyne_utils import read_velodyne_bin, read_label_file
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class KITTI_LiDAR(Dataset):
    def __init__(self, sequences=['01', '02', '03'], data_root='C:\\Users\\User\\LiDAR_point_cloud_segmentation',
                 target_size=(16, 512)):
        self.files = []
        self.ground_classes = [40, 44, 48, 49]
        self.target_H, self.target_W = target_size
        for seq in sequences:
            lidar_dir = os.path.join(data_root, f"velodyne-point-cloud/{seq}/velodyne/")
            label_dir = os.path.join(data_root, f"labels/{seq}/labels/")
            lidar_files = sorted([f for f in os.listdir(lidar_dir) if f.endswith('.bin')])
            label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.label')])

            # формируем пары полный путь
            for lf, labf in zip(lidar_files, label_files):
                self.files.append((
                    os.path.join(lidar_dir, lf),
                    os.path.join(label_dir, labf)
                ))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        lidar_file, label_file = self.files[idx]
        xyz = read_velodyne_bin(lidar_file)
        labels = read_label_file(label_file)
        binary_labels = np.isin(labels, self.ground_classes).astype(np.int64)
        u, v, r = spherical_projection(xyz)
        label_img = np.zeros((64, 2048), dtype=np.int64)
        for i in range(len(binary_labels)):
            label_img[v[i], u[i]] = binary_labels[i]
        input_img = range_image(xyz)
        factor_H = 64 // self.target_H
        factor_W = 2048 // self.target_W
        small_label = label_img[::factor_H, ::factor_W]
        return torch.tensor(input_img).float(), torch.tensor(small_label).long()

def train():
    Segformer = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    old_patch = Segformer.segformer.encoder.patch_embeddings[0].proj
    new_patch_embed = nn.Conv2d(5, 32, kernel_size=(7, 7), stride=(4, 4), padding=(3, 3))
    with torch.no_grad():
        new_patch_embed.weight[:, :3, :, :] = old_patch.weight
        new_patch_embed.weight[:, 3:, :, :].zero_()

    Segformer.segformer.encoder.patch_embeddings[0].proj = new_patch_embed
    Segformer.decode_head.classifier=nn.Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1))
    for name, param in Segformer.named_parameters():
        if "decode_head.classifier" in name or "patch_embeddings.0.proj" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    dataset = KITTI_LiDAR()
    loader = DataLoader(dataset, batch_size=16)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(Segformer.parameters(), lr=0.01)
    num_epochs = 3
    Segformer.train().to(device)
    for epoch in range(num_epochs):
        total_loss = 0
        loader_iter = tqdm(loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)
        for inputs, labels in loader_iter:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = Segformer(inputs).logits

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            loader_iter.set_postfix(loss=loss.item())

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(loader):.4f}")
        torch.save(Segformer.state_dict(), 'SegFormer_ground.pth')
