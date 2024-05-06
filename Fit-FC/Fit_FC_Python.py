import warnings

from sklearn.linear_model import LinearRegression
import numpy as np
from skimage.transform import resize
from tqdm import tqdm

from functions import *
from skimage.transform import downscale_local_mean
from datetime import datetime
warnings.filterwarnings('ignore', category=DeprecationWarning, module='np')

class Fit_FC:
    def __init__(self, F_t1, C_t1, C_t2, RM_win_size=3, scale_factor=16, similar_win_size=17, similar_num=20):
        self.F_t1 = F_t1.astype(np.float32)
        self.C_t1 = C_t1.astype(np.float32)
        self.C_t2 = C_t2.astype(np.float32)
        self.RM_win_size = RM_win_size
        self.scale_factor = scale_factor
        self.similar_win_size = similar_win_size
        self.similar_num = similar_num

    def regression_model_fitting(self, band_idx):
        C_t1_band = self.C_t1[:, :, band_idx]
        C_t2_band = self.C_t2[:, :, band_idx]

        C_t1_band_pad = np.pad(C_t1_band,
                               pad_width=((self.RM_win_size//2, self.RM_win_size//2),
                                          (self.RM_win_size//2, self.RM_win_size//2)),
                               mode="reflect")
        C_t2_band_pad = np.pad(C_t2_band,
                               pad_width=((self.RM_win_size//2, self.RM_win_size//2),
                                          (self.RM_win_size//2, self.RM_win_size//2)),
                               mode="reflect")

        a = np.empty(shape=(self.C_t1.shape[0], self.C_t1.shape[1]), dtype=np.float32)
        b = np.empty(shape=(self.C_t1.shape[0], self.C_t1.shape[1]), dtype=np.float32)

        for row_idx in range(self.C_t1.shape[0]):
            for col_idx in range(self.C_t1.shape[1]):
                C_t1_win = C_t1_band_pad[row_idx:row_idx + self.RM_win_size,
                           col_idx:col_idx + self.RM_win_size]
                C_t2_win = C_t2_band_pad[row_idx:row_idx + self.RM_win_size,
                           col_idx:col_idx + self.RM_win_size]
                reg = LinearRegression().fit(C_t1_win.flatten().reshape(-1, 1), C_t2_win.flatten().reshape(-1, 1))

                a[row_idx, col_idx] = reg.coef_
                b[row_idx, col_idx] = reg.intercept_

        C_t2_RM_pred = C_t1_band * a + b
        r = C_t2_band - C_t2_RM_pred

        return a, b, r

    def calculate_distances(self):
        rows = np.linspace(start=0, stop=self.similar_win_size - 1, num=self.similar_win_size)
        cols = np.linspace(start=0, stop=self.similar_win_size - 1, num=self.similar_win_size)
        xx, yy = np.meshgrid(rows, cols, indexing='ij')

        central_row = self.similar_win_size // 2
        central_col = self.similar_win_size // 2
        distances = np.sqrt(np.square(xx - central_row) + np.square(yy - central_col))

        return distances

    def select_similar_pixels(self):
        F_t1_pad = np.pad(self.F_t1,
                          pad_width=((self.similar_win_size // 2, self.similar_win_size // 2),
                                     (self.similar_win_size // 2, self.similar_win_size // 2),
                                     (0, 0)),
                          mode="reflect")
        F_t1_similar_weights = np.empty(shape=(self.F_t1.shape[0], self.F_t1.shape[1], self.similar_num),
                                        dtype=np.float32)
        F_t1_similar_indices = np.empty(shape=(self.F_t1.shape[0], self.F_t1.shape[1], self.similar_num),
                                        dtype=np.uint32)

        distances = self.calculate_distances().flatten()
        for row_idx in tqdm(range(self.F_t1.shape[0]), desc='select_similar_pixels'):
            for col_idx in range(self.F_t1.shape[1]):
                central_pixel_vals = self.F_t1[row_idx, col_idx, :]
                neighbor_pixel_vals = F_t1_pad[row_idx:row_idx + self.similar_win_size,
                                      col_idx:col_idx + self.similar_win_size, :]
                D = np.mean(np.abs(neighbor_pixel_vals - central_pixel_vals), axis=2).flatten()
                similar_indices = np.argsort(D)[:self.similar_num]

                similar_distances = 1 + distances[similar_indices] / (self.similar_win_size//2)
                similar_weights = (1 / similar_distances) / np.sum(1 / similar_distances)

                F_t1_similar_indices[row_idx, col_idx, :] = similar_indices
                F_t1_similar_weights[row_idx, col_idx, :] = similar_weights

        return F_t1_similar_indices, F_t1_similar_weights

    def spatial_filtering(self, F_t2_RM_pred, F_t1_similar_indices, F_t1_similar_weights):
        F_t2_RM_pred_pad = np.pad(F_t2_RM_pred,
                                  pad_width=((self.similar_win_size//2, self.similar_win_size//2),
                                             (self.similar_win_size//2, self.similar_win_size//2)),
                                  mode="reflect")
        SF_pred = np.empty(shape=(self.F_t1.shape[0], self.F_t1.shape[1]), dtype=np.float32)

        for row_idx in range(self.F_t1.shape[0]):
            for col_idx in range(self.F_t1.shape[1]):
                neighbor_pixel_RM_pred = F_t2_RM_pred_pad[row_idx:row_idx + self.similar_win_size,
                                         col_idx:col_idx + self.similar_win_size]

                similar_indices = F_t1_similar_indices[row_idx, col_idx, :]
                similar_weights = F_t1_similar_weights[row_idx, col_idx, :]

                similar_RM_pred = neighbor_pixel_RM_pred.flatten()[similar_indices]
                SF_pred[row_idx, col_idx] = np.sum(similar_weights * similar_RM_pred)

        return SF_pred

    def residual_compensation(self, F_t2_SF_pred, residuals, F_t1_similar_indices, F_t1_similar_weights):
        residuals_pad = np.pad(residuals,
                               pad_width=((self.similar_win_size//2, self.similar_win_size//2),
                                          (self.similar_win_size//2, self.similar_win_size//2)),
                               mode="reflect")
        Fit_FC_pred = F_t2_SF_pred.copy()

        pred_residuals = np.empty(shape=(self.F_t1.shape[0], self.F_t1.shape[1]), dtype=np.float32)

        for row_idx in range(residuals.shape[0]):
            for col_idx in range(residuals.shape[1]):
                neighbor_pixel_residuals = residuals_pad[row_idx:row_idx + self.similar_win_size,
                                           col_idx:col_idx + self.similar_win_size]
                similar_indices = F_t1_similar_indices[row_idx, col_idx, :]
                similar_residuals = neighbor_pixel_residuals.flatten()[similar_indices]
                similar_weights = F_t1_similar_weights[row_idx, col_idx, :]
                residual = np.sum(similar_residuals * similar_weights)

                pred_residuals[row_idx, col_idx] = residual

                Fit_FC_pred[row_idx, col_idx] += residual

        return Fit_FC_pred, pred_residuals

    def fit_fc(self):
        RM_pred = np.empty(shape=self.F_t1.shape, dtype=np.float32)
        SF_pred = np.empty(shape=self.F_t1.shape, dtype=np.float32)
        Fit_FC_pred = np.empty(shape=self.F_t1.shape, dtype=np.float32)

        similar_indices, similar_weights = self.select_similar_pixels()
        print("Selected similar pixels!")

        for band_idx in tqdm(range(self.F_t1.shape[2]), desc='Fitting FC'):
            a, b, r = self.regression_model_fitting(band_idx)
            a = resize(a, output_shape=(self.F_t1.shape[0], self.F_t1.shape[1]), order=0)
            b = resize(b, output_shape=(self.F_t1.shape[0], self.F_t1.shape[1]), order=0)
            r = resize(r, output_shape=(self.F_t1.shape[0], self.F_t1.shape[1]), order=3)
            band_RM_pred = self.F_t1[:, :, band_idx] * a + b
            print(f"Finished RM prediction of band {band_idx}!")

            band_SF_pred = self.spatial_filtering(band_RM_pred, similar_indices, similar_weights)
            print(f"Finished spatial filtering of band {band_idx}!")

            band_Fit_FC_pred, pred_residuals = self.residual_compensation(band_SF_pred, r, similar_indices,
                                                                          similar_weights)
            print(f"Finished final prediction of band {band_idx}!")

            RM_pred[:, :, band_idx] = band_RM_pred
            SF_pred[:, :, band_idx] = band_SF_pred
            Fit_FC_pred[:, :, band_idx] = band_Fit_FC_pred

        return RM_pred, SF_pred, Fit_FC_pred


###########################################################
#                  Parameters setting                     #
###########################################################
RM_win_size = 3
scale_factor = 30
similar_win_size = 31
similar_num = 30

F_tb_path = r"/home/zbl/datasets_paper/LGC/val/2005_045_0214-2005_061_0302/20050214_TM.tif"
C_tb_path = r"/home/zbl/datasets_paper/LGC/val/2005_045_0214-2005_061_0302/MOD09GA_A2005045.tif"
C_tp_path = r"/home/zbl/datasets_paper/LGC/val/2005_045_0214-2005_061_0302/MOD09GA_A2005061.tif"
Fit_FC_path = r"/home/zbl/RunLog/Fit-FC/LGC/PRED_2005_045_0214-2005_061_0302.tif"

if __name__ == "__main__":
    F_tb, F_tb_profile = read_raster(F_tb_path)
    print(F_tb_profile)
    C_tb = read_raster(C_tb_path)[0]
    C_tb_coarse = downscale_local_mean(C_tb, factors=(scale_factor, scale_factor, 1))
    C_tp = read_raster(C_tp_path)[0]
    C_tp_coarse = downscale_local_mean(C_tp, factors=(scale_factor, scale_factor, 1))
    print("裁剪完成，正在处理中，请耐心等待")
    time0 = datetime.now()
    fit_fc = Fit_FC(F_tb, C_tb_coarse, C_tp_coarse,
                    RM_win_size=RM_win_size,
                    scale_factor=scale_factor,
                    similar_win_size=similar_win_size, similar_num=similar_num)
    print("初始化成功，计算中")
    F_tp_RM, F_tp_SF, F_tp_Fit_FC = fit_fc.fit_fc()
    time1 = datetime.now()
    time_span = time1 - time0
    print(f"Used {time_span.total_seconds():.2f} seconds!")

    write_raster(F_tp_Fit_FC, F_tb_profile, Fit_FC_path)

