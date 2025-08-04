#!/bin/bash
#SBATCH -p lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH -o train_%j.out
#SBATCH --exclude=lrz-hgx-h100-025

# Move into this script's directory
cd "$(dirname "$0")"

# Show GPU status
nvidia-smi

# Ensure Enroot image exists (skip if already)
enroot create /dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/messplus.sqsh || true

# Launch container and run training loop inside it
enroot start \
    --root \
    -e HF_TOKEN_PATH=/dss/dsshome1/06/go76xom2/.cache/huggingface/token \
    -e HF_HOME=/dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/go76xom2/.cache/hf/misc \
    -e HF_DATASETS_CACHE=/dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/go76xom2/.cache/datasets \
    -e HF_DATASETS_TRUST_REMOTE_CODE=True \
    -e VLLM_USE_V1=0 \
    --mount /dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/go76xom2/ \
    --mount /dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/ \
    --mount /dss/dsshome1/09/ge56heh2/ \
    --mount /sbin \
    --mount /usr/share/ \
    messplus /bin/bash << 'EOF'

# Go to the repo inside the container
cd /dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/go76xom2/RouterDC

# Hyperparameters
export top_k=4
export last_k=4
export training_steps=1000
export learning_rate="5e-5"
export temperature=1
export sample_loss_weight=0
export cluster_loss_weight=1
export seeds=(1)

# Prepare output directories
mkdir -p results logs

# Run experiments loop
for seed in "${seeds[@]}"; do
  EXP_NAME=qwen2_clw_${cluster_loss_weight}_slw_${sample_loss_weight}_tk_${top_k}_lk_${last_k}_lr_${learning_rate}_steps_${training_steps}_seed_${seed}_mixed
  SAVE_DIR=$(pwd)/results/${EXP_NAME}
  LOG_FILE=$(pwd)/logs/${EXP_NAME}.log

  mkdir -p "${SAVE_DIR}"
  echo "=== Experiment: ${EXP_NAME} ===" | tee -a "${LOG_FILE}"

  # Invoke the host venv's Python (mounted) inside the container
  /dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/mess-plus/venv/bin/python train_router_mdeberta.py \
    --data_paths \
      /dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/go76xom2/RouterDC/data/qwen2_mixed/arc_challenge/arc_challenge_qwen.json \
      /dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/go76xom2/RouterDC/data/qwen2_mixed/arc_easy/arc_easy_qwen.json \
      /dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/go76xom2/RouterDC/data/qwen2_mixed/boolq/boolq_qwen.json \
      /dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/go76xom2/RouterDC/data/qwen2_mixed/logiqa/logiqa_qwen.json \
      /dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/go76xom2/RouterDC/data/qwen2_mixed/piqa/piqa_qwen.json \
      /dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/go76xom2/RouterDC/data/qwen2_mixed/sciq/sciq_qwen.json \
      /dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/go76xom2/RouterDC/data/qwen2_mixed/social_iqa/social_iqa_qwen.json \
      /dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/go76xom2/RouterDC/data/qwen2_mixed/winogrande/winogrande_qwen.json \
    --batch_size 64 \
    --training_steps ${training_steps} \
    --learning_rate ${learning_rate} \
    --top_k ${top_k} \
    --last_k ${last_k} \
    --tempreture ${temperature} \
    --sample_loss_weight ${sample_loss_weight} \
    --cluster_loss_weight ${cluster_loss_weight} \
    --seed ${seed} \
    --save_path "${SAVE_DIR}" \
    --wandb_entity tum-i13 \
    --project_name routerdc-train 2>&1 | tee -a "${LOG_FILE}"

done

EOF

# End of enroot start
