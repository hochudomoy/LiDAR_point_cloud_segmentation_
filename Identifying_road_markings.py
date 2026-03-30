import numpy as np
from sklearn.linear_model import RANSACRegressor
'''1) Находим порог для фильтрации по интенсивности с помощью метода Оцу
   2) Фильтруем точки земли с учётом наёденного порога * 1,5
   3) С помощью RANSAC находим линии разметки и удаляем шум. 
'''
def otsu_threshold(intensity, nbins=256):
    mean_intensity = np.mean(intensity)
    var_intensity = np.var(intensity)
    t0 = mean_intensity + var_intensity

    hist, bin_edges = np.histogram(intensity, bins=nbins)
    hist = hist.astype(np.float64)
    hist /= hist.sum()

    for i in range(len(bin_edges)):
        if bin_edges[i] >= t0:
            start_t = i
            break
    best_t = start_t
    best_sigma = -1
    for t in range(start_t, len(hist)):
        w0 = 0.0
        m0 = 0.0
        for i in range(t):
            w0 += hist[i]
            m0 += hist[i] * i
        w1 = 0.0
        m1 = 0.0
        for i in range(t, len(hist)):
            w1 += hist[i]
            m1 += hist[i] * i
        mu0 = m0 / w0
        mu1 = m1 / w1
        sigma_b = w0 * w1 * (mu0 - mu1) ** 2

        if sigma_b > best_sigma:
            best_sigma = sigma_b
            best_t = t

    return bin_edges[best_t]
def fit_line_ransac(points_xy):
    X = points_xy[:, 0].reshape(-1, 1)
    y = points_xy[:, 1]
    model = RANSACRegressor(
        residual_threshold=0.4,
        max_trials=500)
    model.fit(X, y)
    return model.inlier_mask_

def markings_search(lidar_df, Nl=10, Np=10):
    t = otsu_threshold(lidar_df['intensity'])
    candidate_markings = lidar_df[lidar_df['intensity'] >= t * 1.5]
    remaining_df = candidate_markings.copy()
    lines = []
    for _ in range(Nl):
        if len(remaining_df) < Np:
            break
        points = remaining_df[["x", "y"]].to_numpy()
        inliers = fit_line_ransac(points)
        inlier_df = remaining_df[inliers]
        outlier_df = remaining_df[~inliers]
        if len(inlier_df) <= Np:
            break
        lines.append(inlier_df)
        remaining_df = outlier_df
    return lines