#!/bin/bash hello

CUDA_VISIBLE_DEVICES=0 python main.py --block_size 128 --rts 0.5 --my_file_name llama3_diaq_error --max_rotation_step 256 --epochs 0 --wbits 4 --abits 4 --model /mnt/models/llama/llama-3/Llama-3-8B --lwc --alpha 0.6 --smooth --lac 0.9 --swc 0.8 --eval_ppl