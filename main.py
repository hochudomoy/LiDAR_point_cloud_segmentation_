from velodyne_utils import read_velodyne_bin, read_label_file, metrics
from visualization import visualization_2D, visualization_3D, gif_2D, gif_3D
from boundaries_extracting import extract_curb, extract_border
import ground_filtering
import SalsaNext.inference
import SegFormer.inference
import os
from identifying_road_markings import otsu_threshold, markings_search
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_folder", type=str, required=True, help="Путь к папке с файлами .bin")
parser.add_argument("--type_of_vizualization", type=str, required=True,choices=["3D_gif", "2D_gif", "3D_paint","2D_paint"], help="Тип визуализации")
parser.add_argument("--ground_segmentation", type=str, required=False,choices=["RANSAC", "grid_RANSAC", "DBSCAN","Neighbours_grid_filter","iterative_filter","SalsaNext","SegFormer","combined"],default="RANSAC", help="Алгоритм сегментации подстилающей поверхности")
parser.add_argument("--boundaries_segmentation", type=str, required=False,choices=["extract_curb","extract_border","false"],default="extract_curb", help="Алгоритм сегментации границ")
parser.add_argument("--markings_segmentation", type=str, required=False,default="true", help="Сегментация разметки: true/false")
parser.add_argument("--gif_frames", type=str, required=False, default=1, help="Количество кадров для gif")
args = parser.parse_args()

gif_lidar=[]
gif_preds=[]
gif_curbs=[]
gif_markings=[]
max_gif_frames=int(args.gif_frames)

if args.ground_segmentation=='SalsaNext': model = SalsaNext.inference.load_model()
if args.ground_segmentation=='SegFormer': model = SegFormer.inference.load_model('ground')
if args.boundaries_segmentation=='extract_border': boundaries_model = SegFormer.inference.load_model('road')

folder=args.input_folder
files=os.listdir(folder)
for i in files:
    filename = f'{folder}\\{i}'
    lidar_df = read_velodyne_bin(filename)
    if args.ground_segmentation=='RANSAC':
        final_pred = ground_filtering.ground_ransac(lidar_df)
        ground_points = lidar_df[final_pred]
    elif args.ground_segmentation=='grid_RANSAC':
        final_pred = ground_filtering.ground_grid_ransac(lidar_df)
        ground_points = lidar_df[final_pred]
    elif args.ground_segmentation=='DBSCAN':
        final_pred = ground_filtering.ground_dbscan(lidar_df)
        ground_points = lidar_df[final_pred]
    elif args.ground_segmentation=='iterative_filter':
        final_pred = ground_filtering.iterative_ground_filtering(lidar_df)
        ground_points = lidar_df[final_pred]
    elif args.ground_segmentation=='SalsaNext':
        final_pred,prob = SalsaNext.inference.SalsaNext(lidar_df,model)
        ground_points = lidar_df[final_pred]
    elif args.ground_segmentation=='SegFormer':
        final_pred,prob = SegFormer.inference.SegFormer(lidar_df, model)
        ground_points = lidar_df[final_pred]
    elif args.ground_segmentation=='"Neighbours_grid_filter"':
        final_pred = ground_filtering.ground_neighbours_grid_filter(lidar_df)
        ground_points = lidar_df[final_pred]
    else:
        final_pred = ground_filtering.ground_ransac(lidar_df)
        pred, prob = SegFormer.inference.SegFormer(lidar_df, model)
        final_pred[prob > 0.75] = 1
        ground_points = lidar_df[final_pred]
    if args.boundaries_segmentation=='extract_border':
        pred, prob = SegFormer.inference.SegFormer(lidar_df, boundaries_model)
        road_points = lidar_df[pred == 1]
        curb_lines = extract_border(ground_points)
    elif args.boundaries_segmentation=='extract_curb':
        curb_points, curb_lines = extract_curb(ground_points)
    if args.markings_segmentation=='true':
        t = otsu_threshold(ground_points['intensity'])
        candidate_markings = ground_points[ground_points['intensity'] >= t * 2]
        markings = markings_search(candidate_markings, Nl=6, Np=10)
    if len(gif_lidar) < max_gif_frames:
        gif_lidar.append(lidar_df)
        gif_preds.append(final_pred)
        if args.boundaries_segmentation != 'false': gif_curbs.append(curb_lines)
        if args.markings_segmentation != 'false': gif_markings.append(candidate_markings)
    else: break
if args.type_of_vizualization == '2D_gif': gif_2D(gif_lidar,gif_preds,gif_curbs,gif_markings)
if args.type_of_vizualization =='3D_gif':gif_3D(gif_lidar,gif_preds,gif_curbs,gif_markings)

if args.type_of_vizualization == '2D_paint':visualization_2D(lidar_df, curb_lines,candidate_markings,color=final_pred)
if args.type_of_vizualization == '3D_paint':visualization_3D(lidar_df, final_pred,curb_lines,candidate_markings)

