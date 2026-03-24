from velodyne_utils import read_velodyne_bin, read_label_file, metrics
from visualization import visualization_2D, visualization_3D, gif_2D, gif_3D
from boundaries_extracting import extract_curb
import ground_filtering
import SalsaNext.inference
import SegFormer.inference
import os
import numpy as np
import pandas as pd
import time

test_folders = [
    'C:\\Users\\User\\LiDAR_point_cloud_segmentation\\velodyne\\07\\velodyne',
    'C:\\Users\\User\\LiDAR_point_cloud_segmentation\\velodyne\\09\\velodyne',
    'C:\\Users\\User\\LiDAR_point_cloud_segmentation\\velodyne\\10\\velodyne'
]
ground_classes=[40, 44, 48, 49]#40: "road" 44: "parking" 48: "sidewalk" 49: "other-ground"]
results=[]
gif_lidar=[]
gif_preds=[]
gif_curbs=[]
max_gif_frames=1
model = SegFormer.inference.load_model()

for folder in test_folders:
    files=os.listdir(folder)
    miou, f1, precision, recall, total_time = 0, 0, 0, 0, 0
    for i in files:
        filename = f'{folder}\\{i}'
        lidar_df = read_velodyne_bin(filename)
        class_labels = read_label_file(
            f"C:\\Users\\User\\LiDAR_point_cloud_segmentation\\labels\\{folder.split('\\')[-2]}\\labels\\{i.split('.')[0]}.label")
        gt_mask = np.isin(class_labels, list(ground_classes))
        start = time.time()
        geometric_pred = ground_filtering.ground_neighbours_grid_filter(lidar_df)
        pred, prob = SegFormer.inference.SegFormer(lidar_df, model)
        final_pred = geometric_pred
        final_pred[prob > 0.75] = 1
        ground_points = lidar_df[final_pred]
        curb_points, curb_lines = extract_curb(ground_points)
        end = time.time()

        total_time += (end - start)
        metric = metrics(final_pred, gt_mask)
        miou += metric[0]
        f1 += metric[1]
        precision += metric[2]
        recall += metric[3]

        seq = folder.split('\\')[-2]
        if seq == '07' and len(gif_lidar) < max_gif_frames:
            gif_lidar.append(lidar_df)
            gif_preds.append(final_pred)
            gif_curbs.append(curb_lines)

    n = len(files)
    results.append([folder.split('\\')[-2], miou / n, f1 / n, precision / n, recall / n, total_time / n])
    df = pd.DataFrame(results, columns=[
        "Sequence",
        "mIoU",
        "F1-score",
        "Precision",
        "Recall",
        "Time per frame"
    ])
print(df)

visualization_2D(lidar_df, curb_lines,color=final_pred)
visualization_3D(lidar_df, final_pred,curb_lines)

gif_2D(gif_lidar,gif_preds,gif_curbs)
gif_3D(gif_lidar,gif_preds,gif_curbs)


