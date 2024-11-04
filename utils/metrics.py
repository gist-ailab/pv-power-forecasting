import numpy as np


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    d += 1e-12
    return 0.01*(u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def nRMSE(pred, true, x_max, x_min, site_ap_max):
    return (np.sqrt(MSE(pred, true))/(x_max-x_min)) * 100
    # for site, ap_max in site_ap_max.items():
        
    # return (np.sqrt(MSE(pred, true))/(x_max-x_min)) * 100

# import numpy as np
# from sklearn.metrics import mean_squared_error

# def calculate_sitewise_nrmse(predictions, targets, site_indices, max_values):
#     """
#     사이트별 NRMSE 계산
    
#     predictions: 예측값 배열 (전체 샘플 수,)
#     targets: 실제값 배열 (전체 샘플 수,)
#     site_indices: 각 샘플에 해당하는 사이트 인덱스 배열 (전체 샘플 수,)
#     max_values: 각 사이트의 최대값 딕셔너리 {site_id: max_value}
    
#     returns: 사이트별 NRMSE 딕셔너리 {site_id: NRMSE}
#     """
#     site_nrmse = {}
#     unique_sites = np.unique(site_indices)
    
#     for site_id in unique_sites:
#         # 해당 사이트의 예측 및 실제값 추출
#         site_mask = site_indices == site_id
#         site_predictions = predictions[site_mask]
#         site_targets = targets[site_mask]
        
#         # RMSE 계산 및 NRMSE로 정규화
#         rmse = np.sqrt(mean_squared_error(site_targets, site_predictions))
#         nrmse = rmse / max_values[site_id]
#         site_nrmse[site_id] = nrmse
        
#     return site_nrmse

# # 예시 사용법
# predictions = np.array([...])  # 전체 예측값
# targets = np.array([...])      # 전체 실제값
# site_indices = np.array([...]) # 사이트 인덱스 배열
# max_values = {                 # 사이트별 최대값 딕셔너리
#     'site1': 1000,
#     'site2': 2000,
#     # ...
# }

# sitewise_nrmse = calculate_sitewise_nrmse(predictions, targets, site_indices, max_values)
# print("Site-wise NRMSE:", sitewise_nrmse)

def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def metric(pred, true, x_max=None, x_min=None):
    print(x_max, x_min)
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    # nrmse = nRMSE(pred, true, x_max, x_min)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    rse = RSE(pred, true)
    # corr = CORR(pred, true)

    # return mae, mse, rmse, mape, mspe, rse, corr
    # return mae, mse, rmse, nrmse, mape, mspe, rse
    return mae, mse, rmse, mape, mspe, rse

