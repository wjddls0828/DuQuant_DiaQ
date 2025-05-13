#!/bin/bash hello

CUDA_VISIBLE_DEVICES=3 python main.py --block_size 128 --rts 0.3 --my_file_name daq_0.3_w4a4 --max_rotation_step 256 --epochs 0 --wbits 4 --abits 4 --model /mnt/models/llama/llama-2/Llama-2-13b-hf --lwc --alpha 0.6 --smooth --lac 0.9 --swc 0.8 --eval_ppl --task arc_easy,arc_challenge,winogrande,boolq,piqa >>output.txt
CUDA_VISIBLE_DEVICES=2 python main.py --block_size 128 --rts 0.4 --my_file_name daq_0.4_w4a4 --max_rotation_step 256 --epochs 0 --wbits 4 --abits 4 --model /mnt/models/llama/llama-2/Llama-2-13b-hf --lwc --alpha 0.6 --smooth --lac 0.9 --swc 0.8 --eval_ppl --task arc_easy,arc_challenge,winogrande,boolq,piqa
CUDA_VISIBLE_DEVICES=2 python main.py --block_size 128 --rts 0.5 --my_file_name daq_0.5_w4a4 --max_rotation_step 256 --epochs 0 --wbits 4 --abits 4 --model /mnt/models/llama/llama-2/Llama-2-13b-hf --lwc --alpha 0.6 --smooth --lac 0.9 --swc 0.8 --eval_ppl --task arc_easy,arc_challenge,winogrande,boolq,piqa

