import numpy as np
import torch
from torchmetrics import R2Score


# - RMSE [kW] 규모 별 평가
# - nRMSE [%]
# - MAE [kW] 규모 별 평가
# - nMAE [%]
# - MAPE [%]
# - MBE [kW] 규모 별 평가
# - R²
# - SS (transfer)


import numpy as np
import torch
from sklearn.metrics import r2_score

class MetricEvaluator:
    def __init__(self, file_path):
        self.file_path = file_path
        self.preds_list = []
        self.targets_list = []
        self.installations_list = []


    def update(self, preds, targets, installations):
        """
        매 배치마다 전체 예측값과 실제값을 누적하여 사용
        """
        self.preds_list.append(preds)
        self.targets_list.append(targets)
        self.installations_list.append(installations)

    def calculate_metrics(self, preds, targets):
        """
        주어진 예측값과 실제값에 대한 지표 계산
        """
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
        epsilon = 1e-10
        mask = np.abs(targets) > epsilon
        if np.any(mask):
            mape = np.mean(np.abs((preds[mask] - targets[mask]) / targets[mask])) * 100
        else:
            mape = np.nan
        
        
        biases = [pred - act for pred, act in zip(preds, targets)]
        mbe = sum(biases) / len(biases)
            # R2 Score 계산
        r2 = r2_score(targets, preds)

        return (rmse, nrmse_range, nrmse_mean, mae, nmae, mape, mbe, r2)


    def evaluate(self, scale_groups, installations):
        """
        전체 데이터에 대한 지표를 %단위는 Installation, 나머지는 규모별로 계산하여 파일에 기록
        scale_groups: list of tuples [(scale_name, mask), ...]
        mask는 numpy 배열로 preds와 targets의 특정 요소를 필터링하는 조건을 나타냅니다.
        """
        preds = np.array(self.preds_list)
        targets = np.array(self.targets_list)

        results = []
        # 규모별 지표 계산 (RMSE, MAE, MBE, R2 )
        for scale_name, scale_func in scale_groups:
            
            mask = scale_func(preds, targets)
            if isinstance(mask, torch.Tensor):
                mask = mask.numpy()
            if np.any(mask):
                masked_preds = preds[mask]
                masked_targets = targets[mask]
                rmse = np.sqrt(np.mean((masked_preds - masked_targets) ** 2))
                mae = np.mean(np.abs(masked_preds - masked_targets))
                mbe = np.mean(masked_preds - masked_targets)
                r2 = r2_score(masked_targets, masked_preds)

                scale_metrics = (rmse, mae, mbe, r2)
           
                # metrics = self.calculate_metrics(masked_preds, masked_targets)
                results.append((scale_name, scale_metrics))
            else:
                print(f"No data for scale {scale_name}")
        
        # Installation별로 지표 계산 (nRMSE, nMAE, MAPE)
        unique_installations = np.unique(self.installations_list)
        installation_results = []
        # all_nrmse = []
        # all_nmae = []
        all_mape = []
        epsilon = 1e-10
        mask = np.abs(targets) > epsilon
        if np.any(mask):
            mape = np.mean(np.abs((preds[mask] - targets[mask]) / targets[mask])) * 100
        else:
            mape = np.nan
        all_mape.append(mape)
        installation_results.append((installation, nrmse, nmae, mape))

        # for installation in unique_installations:
        #     installation_mask = (installations == installation)
        #     installation_preds = preds[installation_mask]
        #     installation_targets = targets[installation_mask]
            
        #     if len(installation_targets) > 0:
                # # nRMSE 계산
                # rmse = np.sqrt(np.mean((installation_preds - installation_targets) ** 2))
                # target_mean = np.mean(installation_targets)
                # nrmse_mean = (rmse / target_mean) * 100  # 평균 기준 nRMSE

                # nRMSE (최대 기준)
                # targets_max = np.max(installation_targets)
                # nrmse = (rmse / (targets_max)) * 100  # (최대-최소) 기준 nRMSE
                # all_nrmse.append(nrmse)
                
                # nMAE 계산
                # targets_mean = np.mean(installation_targets)
                # mae = np.mean(np.abs(installation_preds - installation_targets))
                # nmae = (mae / targets_mean) * 100  # 평균 기준 nMAE
                # all_nmae.append(nmae)
                # MAPE 계산
        
        # avg_nrmse = np.mean(all_nrmse)
        # avg_nmae = np.mean(all_nmae)
        # avg_mape = np.mean(x for x in all_mape if not np.isnan(x))


        # 결과를 파일에 기록
        with open(self.file_path, "w") as file:
            file.write("=" * 50 + "\n")
            file.write("Installation-Specific Evaluation Metrics\n")
            file.write("=" * 50 + "\n")

            file.write("Average Metrics\n")
            file.write(f"Average nRMSE: {avg_nrmse:.4f}%\n")
            file.write(f"Average nMAE: {avg_nmae:.4f}%\n")
            file.write(f"Average MAPE: {avg_mape:.4f}%\n")

            for installation, nrmse, nmae, mape in installation_results:
                file.write(f"Installation: {installation}\n")
                file.write(f"nRMSE (Max): {nrmse:.4f}%\n")
                file.write(f"nMAE: {nmae:.4f}%\n")
                file.write(f"MAPE: {mape:.4f}%\n")
            

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

        return results



    def generate_scale_groups_for_dataset(self, dataset_type):

        if dataset_type == "Source":
            return [
                ("Small", lambda preds, targets: (targets >= 0) & (targets < 30)),
                ("Small-Medium", lambda preds, targets: (targets >= 30) & (targets < 100)),
                ("100kW", lambda preds, targets: (targets >= 100) & (targets < 200)),
                ("200kW", lambda preds, targets: (targets >= 200) & (targets < 300)),
                ("1mW", lambda preds, targets: targets >= 1000)
            ]
        
        elif dataset_type == "GIST":
            return [
                ("Small", lambda preds, targets: (targets >= 0) & (targets < 30)),
                ("Small-Medium", lambda preds, targets: (targets >= 30) & (targets < 100)),
                ("100kW", lambda preds, targets: (targets >= 100) & (targets < 200)),
                ("200kW", lambda preds, targets: (targets >= 200) & (targets < 300))
            ]
        
        elif dataset_type == "Miryang":
            return [
                ("Small", lambda preds, targets: (targets >= 0) & (targets < 30)),
                ("Small-Medium", lambda preds, targets: (targets >= 30) & (targets < 100)),
                ("600kW", lambda preds, targets: (targets >= 600) & (targets < 700)),
                ("900kW", lambda preds, targets: (targets >= 800) & (targets < 900))
            ]
        
        elif dataset_type == "OEDI_California":
            return [
                ("700kW", lambda preds, targets: (targets >= 0) & (targets < 800))
            ]
        elif dataset_type == "OEDI_Georgia":
            return[
                ("3mW", lambda preds, targets: (targets >= 0) & (targets < 4000))
            ]
        elif dataset_type == "UK":
            return[
                ('Not Specified', lambda preds, targets: (targets <= 0)),
                ("Small", lambda preds, targets: (targets >= 0) & (targets < 30)),
                ("Small-Medium", lambda preds, targets: (targets >= 30) & (targets < 100)),
                ("Large", lambda preds, targets: targets >= 100)

            ]
        elif dataset_type == "German":
            return[
                ("Small", lambda preds, targets: (targets >= 0) & (targets < 30)),
                ("Small-Medium", lambda preds, targets: (targets >= 30) & (targets < 100)),
                ("Large", lambda preds, targets: targets >= 100)
            ]
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
    

