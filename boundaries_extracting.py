from sklearn.cluster import DBSCAN
from sklearn.linear_model import RANSACRegressor
from sklearn.neighbors import NearestNeighbors
from scipy.interpolate import splprep, splev
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
def extract_curb(ground_points, height_threshold=[0.05, 0.3], slope_threshold=0.02, k_neighbors=10,
                            block_size=5,cross_width=0.02, eps=0.7, min_samples=7):
    """1) Облако точек разбивается на блоки по оси X
       2) В каждом блоке выполняется разбиение на сечения
       3) В каждом сечении выбираются кандидаты на бордюр по перепаду высот и максимальному уклону
       4) Все выбранные кандидаты  сегментируются с помощью DBSCAN
       5) Для каждого кластера точек бордюра выполняется сглаживание и строится линия с помощью RANSAC и B-spline
       6) Повторяем для каждого блока
       """
    all_curb_points = []
    all_lines = []

    ground_points = ground_points.reset_index(drop=True)
    x_all = ground_points['x'].values
    y_all = ground_points['y'].values
    z_all = ground_points['z'].values
    coords = ground_points[['x', 'y']].values

    nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(coords)
    distances, indices = nbrs.kneighbors(coords)
    indices = indices[:, 1:]

    x_min, x_max = x_all.min(), x_all.max()
    blocks = np.arange(x_min, x_max, block_size)
    for start in blocks:
        block_mask = (x_all >= start) & (x_all < start + block_size)
        if not np.any(block_mask):
            continue
        idx_block = np.where(block_mask)[0]

        x = x_all[idx_block]
        y = y_all[idx_block]
        z = z_all[idx_block]
        cs_min, cs_max = x.min(), x.max()
        cross_sections = np.arange(cs_min, cs_max, cross_width)
        curb_idx_list = []
        for cs in cross_sections:
            mask = (x >= cs) & (x < cs + cross_width)
            if not np.any(mask):
                continue

            slice_idx = np.where(mask)[0]
            slice_idx_global = idx_block[slice_idx]
            dz = z_all[indices[slice_idx_global]] - z_all[slice_idx_global, None]
            dx = x_all[indices[slice_idx_global]] - x_all[slice_idx_global, None]
            dy = y_all[indices[slice_idx_global]] - y_all[slice_idx_global, None]

            delta_h = dz.max(axis=1)
            height_mask = (delta_h >= height_threshold[0]) & (delta_h <= height_threshold[1])
            slope = np.abs(dz / np.sqrt(dx ** 2 + dy ** 2))
            slope_mask = slope.max(axis=1) > slope_threshold

            curb_candidates = height_mask & slope_mask
            if np.any(curb_candidates):
                curb_idx_list.append(idx_block[slice_idx[curb_candidates]])
        curb_points_block=[]
        if curb_idx_list != []:
            curb_idx = np.concatenate(curb_idx_list)
            curb_points_block = ground_points.iloc[curb_idx]
            all_curb_points.append(curb_points_block)
        if curb_idx_list==[]: continue
        xy = curb_points_block[['x', 'y']].values
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(xy)
        labels = db.labels_
        for lbl in set(labels):
            if lbl == -1:
                continue

            cluster = curb_points_block[labels == lbl]
            if len(cluster) < 7:
                continue
            X = cluster['x'].values.reshape(-1, 1)
            y_c = cluster['y'].values
            ransac = RANSACRegressor(residual_threshold=0.3)
            ransac.fit(X, y_c)
            inliers = ransac.inlier_mask_
            x_in = X[inliers].flatten()
            y_in = y_c[inliers]
            z_in = cluster['z'].values[inliers]
            sort_idx = np.argsort(x_in)
            x_sorted = x_in[sort_idx]
            y_sorted = y_in[sort_idx]
            z_sorted = z_in[sort_idx]
            try: tck, _ = splprep([x_sorted, y_sorted, z_sorted], s=1)
            except: continue
            u = np.linspace(0, 1, 200)
            xs, ys, zs = splev(u, tck)

            all_lines.append((xs, ys, zs))

    all_curb_points = pd.concat(all_curb_points)
    return all_curb_points, all_lines

def split_by_distance(points, max_dist=2.0):
    lines = []
    current = [points[0]]
    for i in range(1, len(points)):
        dist = np.linalg.norm(points[i] - points[i-1])

        if dist < max_dist:
            current.append(points[i])
        else:
            if len(current) > 3:
                lines.append(np.array(current))
            current = [points[i]]
    if len(current) > 3:
        lines.append(np.array(current))
    return lines
def extract_border(road_points):
    '''1) С помощью SegFormer сегментируем дорогу
       2) Затем в плоскости XY с помощью метода главных компонент находим ось дороги и нормаль к ней
       3) Центрируем точки и переводим в новую систему координат(s - сдвиг вдоль дороги, t - отклонение от центра дороги)
       4) Вдоль оси s разделяем дорогу на сечения, в каждом сечении находим минимальную и максимальную точку отклонения от центра, соответсвенно кандидаты на левую и правую границы)
       5) Переводим точки обратно в систему координат XY
       6) Разделяем точки на линии, по расстоянию между точками'''
    road_points = road_points.to_numpy()
    pca = PCA(n_components=2)
    pca.fit(road_points[:, :2])

    direction = pca.components_[0]
    normal = pca.components_[1]
    origin = road_points[:, :2].mean(axis=0)

    xy = road_points[:, :2] - origin

    s = xy @ direction
    t = xy @ normal
    bins = np.linspace(s.min(), s.max(), 100)

    left_boundary = []
    right_boundary = []

    for i in range(len(bins) - 1):
        mask = (s >= bins[i]) & (s < bins[i + 1])
        if mask.sum() < 10:
            continue

        idx = np.where(mask)[0]

        t_slice = t[idx]

        left_idx = idx[np.argmin(t_slice)]
        right_idx = idx[np.argmax(t_slice)]

        left_boundary.append((road_points[left_idx][0], road_points[left_idx][1], road_points[left_idx][2]))
        right_boundary.append((road_points[right_idx][0], road_points[right_idx][1], road_points[right_idx][2]))

    left_boundary = np.array(left_boundary)
    right_boundary = np.array(right_boundary)

    left_xy = left_boundary[np.argsort(left_boundary[:, 0])]
    right_xy = right_boundary[np.argsort(right_boundary[:, 0])]
    left_lines = split_by_distance(left_xy, max_dist=2.0)
    right_lines = split_by_distance(right_xy, max_dist=2.0)
    lines = left_lines + right_lines
    coords_list=[]
    for line in lines:
        xs = line[:, 0]
        ys = line[:, 1]
        zs = line[:, 2]
        coords_list.append((xs, ys, zs))
    return coords_list

