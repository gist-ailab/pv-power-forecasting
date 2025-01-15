import torch

# Distributed 방식으로 저장된 모델 경로
# distributed_model_path = "/home/seongho_bak/Projects/PatchTST/checkpoints/250113_DKASC_day_PatchTST_sl240_pl24_ll0_nh16_el8_dm512_df2048_patch24(dist)/checkpoint.pth"
distributed_model_path = "/home/seongho_bak/Projects/PatchTST/checkpoints/250113_DKASC_day_PatchTST_sl240_pl24_ll0_nh16_el8_dm512_df2048_patch24(dist)/model_latest.pth"

# 단일 GPU용으로 변환 후 저장할 경로
# single_gpu_model_path = "/home/seongho_bak/Projects/PatchTST/checkpoints/250113_DKASC_day_PatchTST_sl240_pl24_ll0_nh16_el8_dm512_df2048_patch24/checkpoint.pth"
single_gpu_model_path = "/home/seongho_bak/Projects/PatchTST/checkpoints/250113_DKASC_day_PatchTST_sl240_pl24_ll0_nh16_el8_dm512_df2048_patch24/model_latest.pth"

# 모델 로드
state_dict = torch.load(distributed_model_path, map_location="cpu", weights_only=True)

# 'module.' 접두사 제거
new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

# 변환된 state_dict 저장
torch.save(new_state_dict, single_gpu_model_path)
print(f"Converted model saved to {single_gpu_model_path}")
