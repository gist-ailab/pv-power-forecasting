#!/bin/bash

# 사용자가 지정할 GPU ID (예: 0번 GPU)
GPU_ID=$1

# GPU 사용률 확인 함수
check_gpu_usage() {
    usage=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $GPU_ID)
    echo $usage
    if [ "$usage" -eq "0" ]; then
        return 0 # GPU가 사용 중이 아님
    else
        return 1 # GPU가 사용 중임
    fi
}

# GPU 사용률이 0%가 될 때까지 대기
echo "Waiting for GPU $GPU_ID to be available..."
while ! check_gpu_usage; do
    sleep 10 # 10초 간격으로 체크
done

echo "GPU $GPU_ID is now free. Starting training..."

bash /home/seongho_bak/Projects/PatchTST/scripts/PatchTST/DKASC_test.sh $GPU_ID