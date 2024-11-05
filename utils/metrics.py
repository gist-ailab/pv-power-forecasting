import numpy as np
import torch
import torchmetrics
from collections import defaultdict
import os

class MetricEvaluator:
    def __init__(self, file_path):
        self.file_path = file_path
        self.site_max_capacities = {}
        self.site_preds = defaultdict(list)
        self.site_targets = defaultdict(list)
        
        # 전체 지표 초기화
        self.mae_sum, self.mse_sum, self.rmse_sum = 0, 0, 0
        self.nrmse_sum, self.mape_sum, self.mspe_sum, self.rse_sum, self.r2_sum = 0, 0, 0, 0, 0

    def update(self, preds, targets, site_index, site_max_capacities):
        """
        매 배치마다 사이트별 예측값과 실제값을 누적하며, 새 사이트가 있으면 최대 용량 정보를 업데이트
        """
        for i in range(len(site_index)):
            site_id = site_index[i].item()  # 사이트 ID
            if site_id not in self.site_max_capacities:
                self.site_max_capacities[site_id] = site_max_capacities[i]  # 최대 용량 정보 업데이트
            self.site_preds[site_id].append(preds[i].cpu().numpy())
            self.site_targets[site_id].append(targets[i].cpu().numpy())

    def calculate_metrics(self):
        num_sites = len(self.site_max_capacities)  # 누적된 사이트 개수 계산

        with open(self.file_path, "w") as file:
            file.write("Site-wise Evaluation Metrics\n")
            file.write("="*50 + "\n")

            for site_id, site_max in self.site_max_capacities.items():
                preds = np.stack(self.site_preds[site_id])
                targets = np.stack(self.site_targets[site_id])

                mae = np.mean(np.abs(preds - targets))
                mse = np.mean((preds - targets) ** 2)
                rmse = np.sqrt(mse)
                nrmse = ((rmse / site_max) * 100).item()
                mape = np.mean(np.abs((preds - targets) / targets)) * 100
                mspe = np.mean(np.square((preds - targets) / targets)) * 100
                rse = np.sqrt(np.sum((targets - preds) ** 2)) / np.sqrt(np.sum((targets - targets.mean()) ** 2))
                r2 = torchmetrics.R2Score()(torch.tensor(preds.squeeze()), torch.tensor(targets.squeeze()))

                # 사이트별 지표 기록
                file.write(f"Site {site_id} Metrics:\n")
                file.write(f"Max Capacity: {site_max}\n")
                file.write(f"MAE: {mae:.4f}\n")
                file.write(f"MSE: {mse:.4f}\n")
                file.write(f"RMSE: {rmse:.4f}\n")
                file.write(f"nRMSE: {nrmse:.4f}%\n")
                file.write(f"MAPE: {mape:.4f}%\n")
                file.write(f"MSPE: {mspe:.4f}%\n")
                file.write(f"RSE: {rse:.4f}\n")
                file.write(f"R2 Score: {r2:.4f}\n")
                file.write("="*50 + "\n")

                # 지표별 합계 업데이트
                self.mae_sum += mae
                self.mse_sum += mse
                self.rmse_sum += rmse
                self.nrmse_sum += nrmse
                self.mape_sum += mape
                self.mspe_sum += mspe
                self.rse_sum += rse
                self.r2_sum += r2

            # 평균 지표 계산 및 기록
            avg_mae = self.mae_sum / num_sites
            avg_mse = self.mse_sum / num_sites
            avg_rmse = self.rmse_sum / num_sites
            avg_nrmse = self.nrmse_sum / num_sites
            avg_mape = self.mape_sum / num_sites
            avg_mspe = self.mspe_sum / num_sites
            avg_rse = self.rse_sum / num_sites
            avg_r2 = self.r2_sum / num_sites

            file.write("\nOverall Average Metrics:\n")
            file.write("="*50 + "\n")
            file.write(f"Average MAE: {avg_mae:.4f}\n")
            file.write(f"Average MSE: {avg_mse:.4f}\n")
            file.write(f"Average RMSE: {avg_rmse:.4f}\n")
            file.write(f"Average nRMSE: {avg_nrmse:.4f}%\n")
            file.write(f"Average MAPE: {avg_mape:.4f}%\n")
            file.write(f"Average MSPE: {avg_mspe:.4f}%\n")
            file.write(f"Average RSE: {avg_rse:.4f}\n")
            file.write(f"Average R2 Score: {avg_r2:.4f}\n")
            file.write("="*50 + "\n")

        return avg_mae, avg_mse, avg_rmse, avg_nrmse, avg_mape, avg_mspe, avg_rse, avg_r2