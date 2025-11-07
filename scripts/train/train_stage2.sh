#!/bin/bash

LLM_VERSION="prolongvid/prolongvid_stage1_7B"
# for 7b model we recommend bs=1, accum=2, 16 nodes, 128 gpus, lr=1e-5, warmup=0.03
# for 72b model we recommend bs=1, accum=1, 32 nodes, 256 gpus, lr=1e-5, warmup=0.03
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

############### Pretrain ################

PROMPT_VERSION="qwen_1_5"

MID_RUN_NAME="llavanext-siglip-so400m-patch14-384-llava-onevision-qwen2.5-7b-256k-si_sft-grid-p12-svf32_sam-pmv-woCh-vidal100k-fps1_lv1f128-ytth07-3-5min60k-sum12mc013-Sing60k-woNE_a100_vflash_frevit-llm-lr1e-5_bz8_g32"
echo "MID_RUN_NAME: ${MID_RUN_NAME}"

CKPT_PATH=$LLM_VERSION # this could also be the previous stage checkpoint

if [ $RANK == 0 ];
then
ACCELERATE_CPU_AFFINITY=1 deepspeed --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT} \
    llava/train/train_mem.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path ${CKPT_PATH} \
    --version ${PROMPT_VERSION} \
    --data_path "data/stage2_yt8m_2-5min_train_data.json" \
    --image_folder data \
    --video_folder data \
    --mm_tunable_parts="mm_mlp_adapter,mm_language_model" \
    --mm_vision_tower_lr=0 \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres_max_9 \
    --image_grid_pinpoints "\(1x1\),...,\(6x6\)" \
    --mm_patch_merge_type spatial_unpad \
    --video_fps 0.5 --frames_upbound 128 --fixed_num_frames 128 \
    --mm_fixed_patch_length 12 \
    --bf16 True \
    --run_name $MID_RUN_NAME \
    --output_dir "output/${MID_RUN_NAME}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --per_device_eval_batch_size 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 256000 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --mm_newline_position grid \
    --enable_vision_tower_flash_attn True \
    --report_to wandb \
    --dataloader_drop_last True
fi

# You can delete the sdpa attn_implementation if you want to use flash attn
