export EXPERIMENT_NAME="ASPL_adv"
export MODEL_PATH="./stable-diffusion/stable-diffusion-2-1-base"
export CLEAN_TRAIN_DIR="outputs/ASPL/n000050_ADVERSARIAL/noise-ckpt/50" 
export CLEAN_ADV_DIR="outputs/ASPL/n000050_ADVERSARIAL/noise-ckpt/50"
export OUTPUT_DIR="outputs/$EXPERIMENT_NAME/n000050_ADVERSARIAL"
export CLASS_DIR="data/class-person"

# ------------------------- Add adv noise on ASPL-perturbed examples -------------------------
accelerate launch adv_before_train.py \
  --pretrained_model_name_or_path=$MODEL_PATH  \
  --enable_xformers_memory_efficient_attention \
  --instance_data_dir_for_train=$CLEAN_TRAIN_DIR \
  --instance_data_dir_for_adversarial=$CLEAN_ADV_DIR \
  --instance_prompt="a photo of sks person" \
  --class_data_dir=$CLASS_DIR \
  --num_class_images=200 \
  --class_prompt="a photo of person" \
  --output_dir=$OUTPUT_DIR \
  --center_crop \
  --with_prior_preservation \
  --prior_loss_weight=1.0 \
  --resolution=512 \
  --train_text_encoder \
  --train_batch_size=1 \
  --max_train_steps=10 \
  --max_f_train_steps=3 \
  --max_adv_train_steps=12 \
  --checkpointing_iterations=10 \
  --learning_rate=5e-7 \
  --pgd_alpha=2e-3 \
  --pgd_eps=2e-2 


# ------------------------- Train DreamBooth on adv examples -------------------------
export INSTANCE_DIR="$OUTPUT_DIR/noise-ckpt/10"
export DREAMBOOTH_OUTPUT_DIR="outputs/$EXPERIMENT_NAME/n000050_DREAMBOOTH"

accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_PATH  \
  --enable_xformers_memory_efficient_attention \
  --train_text_encoder \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$DREAMBOOTH_OUTPUT_DIR \
  --with_prior_preservation \
  --prior_loss_weight=1.0 \
  --instance_prompt="a photo of sks person" \
  --class_prompt="a photo of person" \
  --inference_prompt="a photo of sks person;a dslr portrait of sks person" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-7 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --max_train_steps=1000 \
  --checkpointing_steps=500 \
  --center_crop \
  --mixed_precision=bf16 \
  --prior_generation_precision=bf16 \
  --sample_batch_size=8 \
  --use_8bit_adam