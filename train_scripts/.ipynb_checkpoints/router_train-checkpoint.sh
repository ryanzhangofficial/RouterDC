export CUDA_VISIBLE_DEVICES="6"

top_k=3
last_k=3
training_steps=1000
learning_rate="5e-5"
tempreture=1
similarity_function="cos"
sample_loss_weight=0
cluster_loss_weight=1
seeds=(1)

for seed in "${seeds[@]}"; do
    python -u train_router_mdeberta.py \
        --training_steps ${training_steps} \
        --top_k ${top_k} \
        --last_k ${last_k} \
        --learning_rate ${learning_rate} \
        --eval_steps 50 \
        --tempreture ${tempreture} \
        --similarity_function ${similarity_function} \
        --sample_loss_weight ${sample_loss_weight} \
        --cluster_loss_weight ${cluster_loss_weight} \
        --final_eval \
        --seed ${seed} \
        --batch_size 64 \
        --data_paths \
        "./data/inference_outputs/arc_challenge/arc_challenge_train.json" \
        "./data/inference_outputs/arc_easy/arc_easy_train.json" \
        "./data/inference_outputs/boolq/boolq_train.json" \
        "./data/inference_outputs/lambada_standard/lambada_standard_train.json" \
        "./data/inference_outputs/logiqa/logiqa_train.json" \
        "./data/inference_outputs/logiqa2/logiqa2_train.json" \
        "./data/inference_outputs/mmlu_abstract_algebra/mmlu_abstract_algebra_train.json" \
        "./data/inference_outputs/piqa/piqa_train.json" \
        "./data/inference_outputs/sciq/sciq_train.json" \
        "./data/inference_outputs/social_iqa/social_iqa_train.json" \
        "./data/inference_outputs/winogrande/winogrande_train.json" \
        --test_data_paths \
        "./data/inference_outputs/arc_challenge/arc_challenge_train.json" \
        "./data/inference_outputs/arc_easy/arc_easy_train.json" \
        "./data/inference_outputs/boolq/boolq_train.json" \
        "./data/inference_outputs/lambada_standard/lambada_standard_train.json" \
        "./data/inference_outputs/logiqa/logiqa_train.json" \
        "./data/inference_outputs/logiqa2/logiqa2_train.json" \
        "./data/inference_outputs/mmlu_abstract_algebra/mmlu_abstract_algebra_train.json" \
        "./data/inference_outputs/piqa/piqa_train.json" \
        "./data/inference_outputs/sciq/sciq_train.json" \
        "./data/inference_outputs/social_iqa/social_iqa_train.json" \
        "./data/inference_outputs/winogrande/winogrande_train.json" \
        --test_data_type \
        "probability" "probability" "probability" "probability" "probability" \
        "probability" "probability" "probability" "probability" "probability" "probability" \
        --save_path "/data/home/chensh/projects/LLM_router/logs/paper_result/RouterDC/clw_${cluster_loss_weight}/slw_${sample_loss_weight}_clw_${cluster_loss_weight}_${similarity_function}_tk_${top_k}_lk_${last_k}_lr_${learning_rate}_step_${training_steps}_t_${tempreture}_seed_${seed}" \
        > "./logs/paper_result/RouterDC/slw_${sample_loss_weight}_clw_${cluster_loss_weight}_${similarity_function}_tk_${top_k}_lk_${last_k}_lr_${learning_rate}_step_${training_steps}_t_${tempreture}_seed_${seed}.log" 2>&1
done
