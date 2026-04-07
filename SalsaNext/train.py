import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from qai_hub_models.models.salsanext import Model
import tqdm
from range_image import range_image,spherical_projection
import os
import numpy as np
from velodyne_utils import read_velodyne_bin, read_label_file
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ground_classes=['40','44','48','49']
class KITTI_LiDAR(Dataset):
    def __init__(self, sequences=['01', '02', '03'], data_root='C:\\Users\\User\\LiDAR_point_cloud_segmentation'):
        self.files = []
        self.ground_classes = [40, 44, 48, 49]
        for seq in sequences:
            lidar_dir = os.path.join(data_root, f"velodyne/{seq}/velodyne/")
            label_dir = os.path.join(data_root, f"labels/{seq}/labels/")
            lidar_files = sorted([f for f in os.listdir(lidar_dir) if f.endswith('.bin')])
            label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.label')])

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
        binary_labels = np.isin(labels, ground_classes).astype(np.int64)
        u, v, r = spherical_projection(xyz)
        label_img = np.zeros((64, 2048), dtype=np.int64)
        for i in range(len(binary_labels)):
            label_img[v[i], u[i]] = binary_labels[i]
        input_img = range_image(xyz)

        ri = range_image(xyz)
        return torch.tensor(input_img).float(), torch.tensor(label_img).long()

def train():
    dataset = KITTI_LiDAR()
    loader = DataLoader(dataset, batch_size=16)

    SalsaNext = Model.from_pretrained()
    SalsaNext.model.module.logits = nn.Conv2d(32, 2, kernel_size=1)

    for name, param in SalsaNext.model.module.named_parameters():
        if  "logits" not in name:
            param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(SalsaNext.parameters(), lr=0.01)

    SalsaNext.train().to(device)
    num_epochs = 3
    for epoch in range(num_epochs):
        total_loss = 0
        loader_iter = tqdm(loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)
        for inputs, labels in loader_iter:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = SalsaNext(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            loader_iter.set_postfix(loss=loss.item())

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(loader):.4f}")

    torch.save(SalsaNext.state_dict(), 'salsanext_ground.pth')