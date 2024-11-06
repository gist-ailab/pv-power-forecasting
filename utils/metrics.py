import numpy as np
import torch
from torchmetrics import R2Score

class MetricEvaluator:
    def __init__(self, file_path):
        self.file_path = file_path
        self.preds_list = []
        self.targets_list = []

    def update(self, preds, targets):
        """
        매 배치마다 전체 예측값과 실제값을 누적하여 사용
        """
        # numpy 배열로 변환 후 누적
        self.preds_list.append(preds.cpu().numpy())
        self.targets_list.append(targets.cpu().numpy())

    def calculate_metrics(self):
        # 전체 예측 및 실제값을 통합
        preds = np.concatenate(self.preds_list)
        targets = np.concatenate(self.targets_list)

        # 텐서를 1D로 변환하여 R2Score와 호환되도록 수정
        preds_flat = torch.tensor(preds).flatten()
        targets_flat = torch.tensor(targets).flatten()

        # 전체 데이터에 대한 지표 계산
        mae = np.mean(np.abs(preds - targets))
        mse = np.mean((preds - targets) ** 2)
        rmse = np.sqrt(mse)

        # nRMSE (최대-최소 및 평균값 기준)
        targets_min = np.min(targets)
        targets_max = np.max(targets)
        targets_mean = np.mean(targets)

        nrmse_range = (rmse / (targets_max - targets_min)) * 100  # (최대-최소) 기준 nRMSE
        nrmse_mean = (rmse / targets_mean) * 100  # 평균 기준 nRMSE

        # nMAE 계산 (평균값 기준)
        nmae = (mae / targets_mean) * 100  # 평균 기준 nMAE

        # MAPE 계산
        mape = np.mean(np.abs((preds - targets) / targets)) * 100

        # R2 Score 계산 (1D 텐서로 변환된 값을 사용)
        r2 = R2Score()(preds_flat, targets_flat).item()

        # 결과를 파일에 기록
        with open(self.file_path, "w") as file:
            file.write("Overall Evaluation Metrics\n")
            file.write("="*50 + "\n")
            file.write(f"RMSE: {rmse:.4f} kW\n")
            file.write(f"nRMSE (Range): {nrmse_range:.4f}%\n")
            file.write(f"nRMSE (Mean): {nrmse_mean:.4f}%\n")
            file.write(f"MAE: {mae:.4f} kW\n")
            file.write(f"nMAE: {nmae:.4f}%\n")
            file.write(f"MAPE: {mape:.4f}%\n")
            file.write(f"R2 Score: {r2:.4f}\n")
            file.write("="*50 + "\n")

        return rmse, nrmse_range, nrmse_mean, mae, nmae, mape, r2