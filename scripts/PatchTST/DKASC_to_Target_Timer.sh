#!/bin/bash

# 사용자가 지정할 GPU ID (예: 0번 GPU)
GPU_ID=$1
num_freeze_layers=$2
data_name=$3

# GPU 사용률 확인 함수
check_gpu_usage() {
    usage=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $GPU_ID)    # GPU 사용률
    memory_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU_ID)  # GPU 메모리 사용량 (MB)
    current_time=$(date "+%Y-%m-%d %H:%M:%S") # 현재 시간

    echo "[$current_time] GPU $GPU_ID Utilization: $usage%, Memory Used: ${memory_used}MB"

    if [ "$usage" -eq "0" ] && [ "$memory_used" -le "50" ]; then
        return 0 # GPU가 사용 중이 아니며 메모리 사용량이 10MB 이하임
    else
        return 1 # 조건을 만족하지 않음
    fi
}

# GPU 사용률이 0%가 될 때까지 대기
echo "Waiting for GPU $GPU_ID to be available..."
while ! check_gpu_usage; do
    sleep 5 # 10초 간격으로 체크
done

echo "GPU $GPU_ID is now free. Starting training..."

# bash /home/seongho_bak/Projects/PatchTST/scripts/PatchTST/DKASC_to_Target_chronological.sh $GPU_ID $num_freeze_layers
bash /home/seongho_bak/Projects/PatchTST/scripts/PatchTST/DKASC_to_Target_nh16_el8_dm512_df2408_patch24.sh $GPU_ID $num_freeze_layers $data_name