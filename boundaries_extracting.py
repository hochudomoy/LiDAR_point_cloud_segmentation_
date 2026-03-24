from sklearn.cluster import DBSCAN
from sklearn.linear_model import RANSACRegressor
from sklearn.neighbors import NearestNeighbors
from scipy.interpolate import splprep, splev
import numpy as np
import pandas as pd
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
            tck, _ = splprep([x_sorted, y_sorted,z_sorted], s=1)
            u = np.linspace(0, 1, 200)
            xs, ys, zs = splev(u, tck)

            all_lines.append((xs, ys, zs))

    all_curb_points = pd.concat(all_curb_points)
    return all_curb_points, all_lines


