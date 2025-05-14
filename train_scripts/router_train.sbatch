#!/bin/bash
#SBATCH -p lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH -o train_%j.out
#SBATCH --exclude=lrz-hgx-h100-025

# Move into the repo root
cd /dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/go76xom2/RouterDC

# Show GPU status
nvidia-smi

# Ensure Enroot image is created (skip if already exists)
enroot create /dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/messplus.sqsh || true

# Launch container and run training
enroot start \
    --root \
    -e HF_TOKEN_PATH=/dss/dsshome1/06/go76xom2/.cache/huggingface/token \
    -e HF_HOME=/dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/go76xom2/.cache/hf/misc \
    -e HF_DATASETS_CACHE=/dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/go76xom2/.cache/datasets \
    -e HF_DATASETS_TRUST_REMOTE_CODE=True \
    -e OPENAI_API_KEY=/dss/dsshome1/06/go76xom2/.cache/openai/token \
    -e VLLM_USE_V1=0 \
    --mount /dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ \
    --mount /dss/dsshome1/06/go76xom2/ \
    --mount /sbin \
    --mount /usr/share/ \
    messplus \
    bash -lc "
# Inside container now:

# Hyperparameters
top_k=3
last_k=3
training_steps=1000
learning_rate=\"5e-5\"
tempreture=1
sample_loss_weight=0
cluster_loss_weight=1
seeds=(1)

cd /dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/go76xom2/RouterDC
mkdir -p results logs

for seed in \"\${seeds[@]}\"; do
  EXP_NAME=clw_\${cluster_loss_weight}_slw_\${sample_loss_weight}_tk_\${top_k}_lk_\${last_k}_lr_\${learning_rate}_steps_\${training_steps}_seed_\${seed}
  SAVE_DIR=\$(pwd)/results/\${EXP_NAME}
  LOG_FILE=\$(pwd)/logs/\${EXP_NAME}.log

  mkdir -p \"\${SAVE_DIR}\"
  echo \"=== Experiment: \${EXP_NAME} ===\" | tee -a \"\${LOG_FILE}\"

  /dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/mess-plus/venv/bin/python \
    /dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/go76xom2/RouterDC/train_router_mdeberta.py \
    --data_paths \
      /dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/go76xom2/RouterDC/data/inference_outputs/arc_challenge/arc_challenge_train.json \
      /dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/go76xom2/RouterDC/data/inference_outputs/arc_easy/arc_easy_train.json \
      /dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/go76xom2/RouterDC/data/inference_outputs/boolq/boolq_train.json \
      /dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/go76xom2/RouterDC/data/inference_outputs/lambada_standard/lambada_standard_train.json \
      /dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/go76xom2/RouterDC/data/inference_outputs/logiqa/logiqa_train.json \
      /dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/go76xom2/RouterDC/data/inference_outputs/logiqa2/logiqa2_train.json \
      /dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/go76xom2/RouterDC/data/inference_outputs/mmlu_abstract_algebra/mmlu_abstract_algebra_train.json \
      /dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/go76xom2/RouterDC/data/inference_outputs/piqa/piqa_train.json \
      /dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/go76xom2/RouterDC/data/inference_outputs/sciq/sciq_train.json \
      /dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/go76xom2/RouterDC/data/inference_outputs/social_iqa/social_iqa_train.json \
      /dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/go76xom2/RouterDC/data/inference_outputs/winogrande/winogrande_train.json \
    --batch_size 64 \
    --training_steps \${training_steps} \
    --learning_rate \${learning_rate} \
    --top_k \${top_k} \
    --last_k \${last_k} \
    --tempreture \${tempreture} \
    --sample_loss_weight \${sample_loss_weight} \
    --cluster_loss_weight \${cluster_loss_weight} \
    --seed \${seed} \
    --save_path \"\${SAVE_DIR}\" \
  2>&1 | tee -a \"\${LOG_FILE}\"
done
"

# End of enroot start
