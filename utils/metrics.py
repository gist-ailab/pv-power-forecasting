import numpy as np
import torch
import torchmetrics

def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))

def MAE(pred, true):
    return np.mean(np.abs(pred - true))

def MSE(pred, true):
    return np.mean((pred - true) ** 2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def nRMSE(pred, true, max_value):
    return (np.sqrt(MSE(pred, true)) / max_value) * 100

def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true)) * 100

def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true)) * 100

def R2Score(pred, true):
    r2score = torchmetrics.R2Score()
    return r2score(torch.tensor(pred), torch.tensor(true))

def metric(preds, targets, site_max_capacities, file_path="site_metrics.txt"):
    with open(file_path, "w") as file:
        file.write("Site-wise Evaluation Metrics\n")
        file.write("="*50 + "\n")
        
        # 각 지표의 합계를 저장할 변수 초기화
        mae_sum, mse_sum, rmse_sum, nrmse_sum, mape_sum, mspe_sum, rse_sum, r2_sum = 0, 0, 0, 0, 0, 0, 0, 0

        for i, site_max in enumerate(site_max_capacities):
            site_preds = preds[i]
            site_targets = targets[i]
            
            mae = MAE(site_preds, site_targets)
            mse = MSE(site_preds, site_targets)
            rmse = RMSE(site_preds, site_targets)
            nrmse = nRMSE(site_preds, site_targets, site_max)
            mape = MAPE(site_preds, site_targets)
            mspe = MSPE(site_preds, site_targets)
            rse = RSE(site_preds, site_targets)
            r2 = R2Score(site_preds, site_targets)

            # Write individual site metrics to the file
            file.write(f"Site {i+1} Metrics:\n")
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
            mae_sum += mae
            mse_sum += mse
            rmse_sum += rmse
            nrmse_sum += nrmse
            mape_sum += mape
            mspe_sum += mspe
            rse_sum += rse
            r2_sum += r2

        # 사이트별 지표의 평균 계산
        num_sites = len(site_max_capacities)
        avg_mae = mae_sum / num_sites
        avg_mse = mse_sum / num_sites
        avg_rmse = rmse_sum / num_sites
        avg_nrmse = nrmse_sum / num_sites
        avg_mape = mape_sum / num_sites
        avg_mspe = mspe_sum / num_sites
        avg_rse = rse_sum / num_sites
        avg_r2 = r2_sum / num_sites

        # 파일에 전체 평균 지표 출력
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

