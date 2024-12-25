import numpy as np
import torch
import numpy as np
import torch
from sklearn.metrics import r2_score

class MetricEvaluator:
    def __init__(self, file_path):
        self.file_path = file_path
        self.preds_list = []
        self.targets_list = []
        self.inst_ids = []

    def update(self, inst_id, preds, targets):
        """매 배치마다 installation ID와 함께 예측값과 실제값을 누적"""
        self.inst_ids.append(inst_id)
        self.preds_list.append(preds)
        self.targets_list.append(targets)

    def _calculate_metrics(self, preds, targets):
        """ 주어진 그룹의 metric 계산 """
        rmse = np.sqrt(np.mean((preds - targets) ** 2))
        mae = np.mean(np.abs(preds - targets))
        mbe = np.mean(preds - targets)
        r2 = r2_score(targets, preds)
        return rmse, mae, mbe, r2

    def calculate_mape(self):
        """전체 데이터에 대한 MAPE 계산 - installation별 최대값 기준"""
        total_error = 0
        total_samples = 0
        
        for inst_id, inst_preds, inst_targets in zip(self.inst_ids, self.preds_list, self.targets_list):
            inst_max = np.max(np.abs(inst_targets))
            epsilon = 1e-10
            
            inst_error = np.sum(np.abs(inst_preds - inst_targets)) / max(inst_max, epsilon)
            total_error += inst_error
            total_samples += len(inst_targets)
        
        mape = (total_error / total_samples) * 100
        return mape

    def generate_scale_groups(self):
        """용량 그룹 정의"""
        max_target = np.max(self.targets_list)
        scale_groups = [("Small", lambda targets: (targets >= 0) & (targets < 30)),
                        ("Small-Medium", lambda targets: (targets >= 30) & (targets < 100))]

        # Generate 100kW intervals
        for i in range(1, int(max_target // 100) + 1):
            lower_bound = i * 100
            upper_bound = (i + 1) * 100
            scale_groups.append((f"{lower_bound}kW",
                                 lambda targets, lb=lower_bound, ub=upper_bound:
                                 (targets >= lb) & (targets < ub)))

        # Add the 1MW group
        scale_groups.append(("MW", lambda targets: targets >= 1000))
        return scale_groups

    def evaluate_scale_metrics(self):
        """용량 그룹별 metric 계산"""
        preds = np.concatenate(self.preds_list)
        targets = np.concatenate(self.targets_list)
        scale_groups = self.generate_scale_groups()

        results = []
        for scale_name, scale_func in scale_groups:
            mask = scale_func(targets)
            if isinstance(mask, torch.Tensor):
                mask = mask.numpy()
            
            if np.any(mask):
                masked_preds = preds[mask]
                masked_targets = targets[mask]
                metrics = self._calculate_metrics(masked_preds, masked_targets)
                results.append((scale_name, metrics))
            else:
                print(f"No data for scale {scale_name}")

        # 결과 저장
        with open(self.file_path, "w") as file:
            file.write("=" * 50 + "\n")
            file.write("Scale-Specific Evaluation Metrics\n")
            file.write("=" * 50 + "\n")
            
            for scale_name, (rmse, mae, mbe, r2) in results:
                file.write(f"Scale: {scale_name}\n")
                file.write(f"RMSE: {rmse:.4f} kW\n")
                file.write(f"MAE: {mae:.4f} kW\n")
                file.write(f"MBE: {mbe:.4f} kW\n")
                file.write(f"R2 Score: {r2:.4f}\n")
                file.write("=" * 50 + "\n")
            
            mape = self.calculate_mape()
            file.write(f"\nOverall MAPE: {mape:.4f}%\n")
            
        return results, mape