# !/bin/bash -e

export nnUNet_raw_data_base="$PWD"
export RESULTS_FOLDER="/workspace/checkpoints/"

export CUDA_VISIBLE_DEVICES=0

nnUNet_predict -i "/workspace/inputs" -o "/workspace/outputs" -t Task501_FLARE2022 -tr "nnUNetTrainerV2" -m 3d_fullres --disable_tta -p "nnUNetPlansv2.1_bnf16_bs4" -f 5 --overwrite_existing --step_size 0.5 --num_threads_preprocessing 1 --num_threads_nifti_save 1 --output_folders "base_num_16" "fine-tune-200" --flare_mode --mode "fast"