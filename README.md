# GA-LLM


This repository includes the implementation of paper "[Large Language Models for Next Point-of-Interest Recommendation](https://arxiv.org/pdf/2404.17591)".
# Install
1. Clone this repository to your local machine.
2. Install the enviroment by running
```
conda env create -f environment.yml
```
3. Download the model from (https://huggingface.co/Yukang/Llama-2-7b-longlora-32k-ft)
# Dataset
Download the datasets raw data from [datasets](https://www.dropbox.com/scl/fi/teo5pn8t296joue5c8pim/datasets.zip?rlkey=xvcgtdd9vlycep3nw3k17lfae&st=qd21069y&dl=0).
* Unzip datasets.zip to ./datasets
* Unzip datasets/nyc/raw.zip to datasets/nyc.
* Unzip datasets/tky/raw.zip to datasets/tky.
* Unzip datasets/ca/raw.zip to datasets/ca.
* run ```python preprocesssing/generate_ca_raw.py --dataset_name {dataset_name}```

# Preprocess
run ```python preprocessing/run.py```

run ```python preprocessing/traj_qk.py```

run ```python traj_sim --dataset_name {dataset_name} --model_path {your_model_path}```

run ```python preprocessing/to_nextpoi_qkt.py --dataset_name {dataset_name}```


transformers从4.34.0-->4.44.0
accelerate从0.21.0-->0.34.2
bitsandbytes从0.40.0-->0.43.3
torch从2.0.1-->2.4.0
flash-attn从2.3.2-->2.6.3
deepspeed从0.11.1-->0.15.1

# Main Performance
## train
run
```
nohup torchrun --nproc_per_node=7 supervised-fine-tune-llama3.1-qlora.py  \
--model_name_or_path model_llama3.1_storm/ \
--bf16 True \
--output_dir output_dir_llama3storm_4096_geo/ \
--model_max_length 4096 \
--use_flash_attn True \
--data_path datasets/ca/preprocessed/train_qa_pairs_kqt_geo.json \
--low_rank_training True \
--num_train_epochs 3  \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 2  \
--gradient_accumulation_steps 1  \
--evaluation_strategy "no"  \
--save_strategy "steps"  \
--save_steps 1000  \
--save_total_limit 2  \
--learning_rate 2e-5  \
--weight_decay 0.0  \
--warmup_steps 20  \
--lr_scheduler_type "constant_with_warmup"  \
--logging_steps 1  \
--deepspeed "ds_configs/stage2.json"  \
--tf32 True > runs/log_train_llama3.1storm_9_26_13.txt 2>&1 &
```

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 nohup torchrun --nproc_per_node=7 finetuning_v2.py  \
--model_name_or_path model/ \
--bf16 True \
--output_dir output_dir3/ \
--model_max_length 8192 \
--use_flash_attn True \
--data_path datasets/ca/preprocessed/train_qa_pairs_kqt_output.json \
--poi_embedding_path datasets/ca/preprocessed/POI_embeddings.npy \
--low_rank_training True \
--num_train_epochs 3  \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 1  \
--gradient_accumulation_steps 1  \
--evaluation_strategy "no"  \
--save_strategy "steps"  \
--save_steps 1000  \
--save_total_limit 2  \
--learning_rate 2e-5  \
--weight_decay 0.0  \
--warmup_steps 20  \
--lr_scheduler_type "constant_with_warmup"  \
--logging_steps 1  \
--deepspeed "ds_configs/stage2.json"  \
--tf32 True > runs/log_finetuning_9_7_18.txt 2>&1 &
```

```
CUDA_VISIBLE_DEVICES=0 nohup torchrun --nproc_per_node=1 finetuning_v4.py  \
--model_name_or_path /home/liuzhao/demo/LLM4POI/models/model \
--bf16 True \
--output_dir outputs/output_dir_embedding_32768_mtnetv1/ \
--model_max_length 32768 \
--use_flash_attn True \
--data_path /home/liuzhao/demo/LLM4POI/datasets/ca/preprocessed/train_qa_pairs_kqt_new2_poi.json \
--poi_embedding_path /home/liuzhao/demo/LLM4POI/datasets/ca/preprocessed/POI_embeddings_mtnet_4096.npy \
--low_rank_training True \
--num_train_epochs 3  \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 2  \
--gradient_accumulation_steps 1  \
--evaluation_strategy "no"  \
--save_strategy "steps"  \
--save_steps 1000  \
--save_total_limit 2  \
--learning_rate 2e-5  \
--weight_decay 0.0  \
--warmup_steps 20  \
--lr_scheduler_type "constant_with_warmup"  \
--logging_steps 1  \
--deepspeed "ds_configs/stage2.json"  \
--tf32 True > runs/log_finetuning_v4_poi_10_5_15.txt 2>&1 &
```

加上地理位置信息：
```
nohup torchrun --nproc_per_node=1 supervised-fine-tune-qlora.py  \
--model_name_or_path /root/autodl-tmp/models/model \
--bf16 True \
--output_dir /root/autodl-tmp/liuzhao_data/output_dir_geo_32768/ \
--model_max_length 32768 \
--use_flash_attn True \
--data_path /root/autodl-tmp/liuzhao_data/POI_dataset/ca/preprocessed/train_qa_pairs_kqt_geo.json \
--low_rank_training True \
--num_train_epochs 3  \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 2  \
--gradient_accumulation_steps 1  \
--evaluation_strategy "no"  \
--save_strategy "steps"  \
--save_steps 1000  \
--save_total_limit 2  \
--learning_rate 2e-5  \
--weight_decay 0.0  \
--warmup_steps 20  \
--lr_scheduler_type "constant_with_warmup"  \
--logging_steps 1  \
--deepspeed "ds_configs/stage2.json"  \
--tf32 True > runs/log_train_geo_9_26_18.txt 2>&1 &
```
启动nuhup torchrun之后可以用disown来彻底接触进程和当前会话的绑定


CUDA_VISIBLE_DEVICES=0 nohup torchrun --nproc_per_node=1 supervised-fine-tune-qlora.py  \
--model_name_or_path /home/liuzhao/demo/LLM4POI/models/model \
--bf16 True \
--output_dir outputs/nyc_32768 \
--model_max_length 32768 \
--use_flash_attn True \
--data_path /home/liuzhao/demo/LLM4POI/datasets/nyc/preprocessed/train_qa_pairs_kqt.json \
--low_rank_training True \
--num_train_epochs 3  \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 2  \
--gradient_accumulation_steps 1  \
--evaluation_strategy "no"  \
--save_strategy "steps"  \
--save_steps 1000  \
--save_total_limit 2  \
--learning_rate 2e-5  \
--weight_decay 0.0  \
--warmup_steps 20  \
--lr_scheduler_type "constant_with_warmup"  \
--logging_steps 1  \
--deepspeed "ds_configs/stage2.json"  \
--tf32 True > runs/log_train_nyc_32768_10_7_13.txt 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup torchrun --nproc_per_node=1 supervised-fine-tune-qlora.py  \
--model_name_or_path /home/liuzhao/demo/LLM4POI/models/model \
--bf16 True \
--output_dir outputs/nyc_32768_300items \
--model_max_length 32768 \
--use_flash_attn True \
--data_path /home/liuzhao/demo/LLM4POI/datasets/nyc/preprocessed/train_qa_pairs_kqt_300items.json \
--low_rank_training True \
--num_train_epochs 3  \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 2  \
--gradient_accumulation_steps 1  \
--evaluation_strategy "no"  \
--save_strategy "steps"  \
--save_steps 1000  \
--save_total_limit 2  \
--learning_rate 2e-5  \
--weight_decay 0.0  \
--warmup_steps 20  \
--lr_scheduler_type "constant_with_warmup"  \
--logging_steps 1  \
--deepspeed "ds_configs/stage2.json"  \
--tf32 True > runs/log_train_nyc_32768_300items_10_8_8.txt 2>&1 &


CUDA_VISIBLE_DEVICES=0 nohup torchrun --nproc_per_node=1 finetuning_v4.py  \
--model_name_or_path /home/liuzhao/demo/LLM4POI/models/model \
--bf16 True \
--output_dir outputs/ca/merged_poi/ \
--model_max_length 32768 \
--use_flash_attn True \
--data_path /home/liuzhao/demo/LLM4POI/datasets/ca/preprocessed/train_qa_pairs_kqt_200items_ca_merged_poi.json \
--poi_embedding_path /home/liuzhao/demo/LLM4POI/datasets/ca/preprocessed/ca_POI_embeddings_mtnet_4096.npy \
--low_rank_training True \
--num_train_epochs 3  \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 2  \
--gradient_accumulation_steps 1  \
--evaluation_strategy "no"  \
--save_strategy "steps"  \
--save_steps 1000  \
--save_total_limit 2  \
--learning_rate 2e-5  \
--weight_decay 0.0  \
--warmup_steps 20  \
--lr_scheduler_type "constant_with_warmup"  \
--logging_steps 1  \
--deepspeed "ds_configs/stage2.json"  \
--tf32 True > runs/ca/train_merged_poi_10_16_2.txt 2>&1 &



CUDA_VISIBLE_DEVICES=0 nohup torchrun --nproc_per_node=1 supervised-fine-tune-qlora.py  \
--model_name_or_path /home/liuzhao/demo/LLM4POI/models/model \
--bf16 True \
--output_dir outputs/ca_200items_geo \
--model_max_length 32768 \
--use_flash_attn True \
--data_path /home/liuzhao/demo/LLM4POI/datasets/ca/preprocessed/train_qa_pairs_kqt_200items_geo.json \
--low_rank_training True \
--num_train_epochs 3  \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 2  \
--gradient_accumulation_steps 1  \
--evaluation_strategy "no"  \
--save_strategy "steps"  \
--save_steps 1000  \
--save_total_limit 2  \
--learning_rate 2e-5  \
--weight_decay 0.0  \
--warmup_steps 20  \
--lr_scheduler_type "constant_with_warmup"  \
--logging_steps 1  \
--deepspeed "ds_configs/stage2.json"  \
--tf32 True > runs/ca_200items_geo_10_13_14.txt 2>&1 &


CUDA_VISIBLE_DEVICES=0 nohup torchrun --nproc_per_node=1 finetune_projector.py  \
--model_name_or_path /home/liuzhao/demo/LLM4POI/models/model \
--bf16 True \
--output_dir outputs/ca/projector_wo_sim \
--model_max_length 32768 \
--use_flash_attn True \
--data_path /home/liuzhao/demo/LLM4POI/datasets/ca/preprocessed/train_qa_pairs_kqt_200items_ca_projector_wo_sim.json \
--poi_embedding_path /home/liuzhao/demo/LLM4POI/datasets/ca/preprocessed/ca_POI_embeddings_128.npy \
--low_rank_training True \
--num_train_epochs 3  \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 2  \
--gradient_accumulation_steps 1  \
--evaluation_strategy "no"  \
--save_strategy "steps"  \
--save_steps 1000  \
--save_total_limit 2  \
--learning_rate 2e-5  \
--weight_decay 0.0  \
--warmup_steps 20  \
--lr_scheduler_type "constant_with_warmup"  \
--logging_steps 1  \
--deepspeed "ds_configs/stage2.json"  \
--tf32 True > runs/ca/train_projector_wo_sim_10_22_16.txt 2>&1 &


CUDA_VISIBLE_DEVICES=0 nohup torchrun --nproc_per_node=1 finetune_projector_nyc.py  \
--model_name_or_path /home/liuzhao/demo/LLM4POI/models/model \
--bf16 True \
--output_dir outputs/nyc/projector_demo \
--model_max_length 32768 \
--use_flash_attn True \
--data_path /home/liuzhao/demo/LLM4POI/datasets/nyc/preprocessed/train_qa_pairs_kqt_200items_nyc_projector.json \
--poi_embedding_path /home/liuzhao/demo/LLM4POI/datasets/nyc/preprocessed/nyc_POI_embeddings_mtnet_128.npy \
--low_rank_training True \
--num_train_epochs 3  \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 2  \
--gradient_accumulation_steps 1  \
--evaluation_strategy "no"  \
--save_strategy "steps"  \
--save_steps 1000  \
--save_total_limit 2  \
--learning_rate 2e-5  \
--weight_decay 0.0  \
--warmup_steps 20  \
--lr_scheduler_type "constant_with_warmup"  \
--logging_steps 1  \
--deepspeed "ds_configs/stage2.json"  \
--tf32 True > runs/nyc/train_projector_demo_10_26_14.txt 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup torchrun --nproc_per_node=1  --master_port=29501 finetune_projector.py  \
--model_name_or_path /home/liuzhao/demo/LLM4POI/models/model \
--bf16 True \
--output_dir outputs/ca/projector_fixed_without_sim_change_lora_10_28_16 \
--model_max_length 32768 \
--use_flash_attn True \
--data_path /home/liuzhao/demo/LLM4POI/datasets/ca/preprocessed/train_qa_pairs_kqt_200items_ca_projector_without_sim.json \
--poi_embedding_path /home/liuzhao/demo/LLM4POI/datasets/ca/preprocessed/ca_POI_embeddings_mtnet_128.npy \
--low_rank_training True \
--num_train_epochs 3  \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 2  \
--gradient_accumulation_steps 1  \
--evaluation_strategy "no"  \
--save_strategy "steps"  \
--save_steps 1000  \
--save_total_limit 2  \
--learning_rate 2e-5  \
--weight_decay 0.0  \
--warmup_steps 20  \
--lr_scheduler_type "constant_with_warmup"  \
--logging_steps 1  \
--deepspeed "ds_configs/stage2.json"  \
--tf32 True > runs/ca/train_projector_without_sim_change_lora_10_28_16.txt 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python finetune_projector_2layer.py \
--model_name_or_path /home/liuzhao/demo/LLM4POI/models/model \
--output_dir outputs/nyc/projector_2layer_r32_alpha64_without_sim_11_4_13 \
--lora_r 32 \
--lora_alpha 64 \
--model_max_length 32768 \
--use_flash_attn True \
--data_path /home/liuzhao/demo/LLM4POI/datasets/nyc/preprocessed/train_qa_pairs_kqt_200items_nyc_projector_without_sim_new.json \
--poi_embedding_path /home/liuzhao/demo/LLM4POI/datasets/nyc/preprocessed/nyc_POI_embeddings_mtnet_128.npy \
--low_rank_training True \
--num_train_epochs 3 \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 2 \
--gradient_accumulation_steps 1 \
--evaluation_strategy "no" \
--save_strategy "steps" \
--save_steps 1000 \
--save_total_limit 2 \
--learning_rate 2e-5 \
--weight_decay 0.0 \
--warmup_steps 20 \
--lr_scheduler_type "constant_with_warmup" \
--logging_steps 1 \
--tf32 True > runs/nyc/train_projector_2layer_incremental_r32_alpha64_without_sim_11_6_12.txt 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python finetune_projector_2layer_incremental.py \
--model_name_or_path /home/liuzhao/demo/LLM4POI/models/model \
--peft_dir /home/liuzhao/demo/LLM4POI/outputs/nyc_32768_300items/checkpoint-33066 \
--output_dir outputs/nyc/projector_2layer_incremental_r32_alpha64_without_sim_11_6_12 \
--projector_path /home/liuzhao/demo/LLM4POI/outputs/nyc/projector_2layer_r32_alpha64_without_sim_11_4_13/checkpoint-33066/linear_mapping_layer_combined.pt \
--lora_r 32 \
--lora_alpha 64 \
--model_max_length 32768 \
--use_flash_attn True \
--data_path /home/liuzhao/demo/LLM4POI/datasets/nyc/preprocessed/train_qa_pairs_kqt_200items_nyc_projector_without_sim_new.json \
--poi_embedding_path /home/liuzhao/demo/LLM4POI/datasets/nyc/preprocessed/nyc_POI_embeddings_mtnet_128.npy \
--low_rank_training True \
--num_train_epochs 3 \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 2 \
--gradient_accumulation_steps 1 \
--evaluation_strategy "no" \
--save_strategy "steps" \
--save_steps 1000 \
--save_total_limit 2 \
--learning_rate 2e-5 \
--weight_decay 0.0 \
--warmup_steps 20 \
--lr_scheduler_type "constant_with_warmup" \
--logging_steps 1 \
--tf32 True > runs/nyc/train_projector_2layer_incremental_r32_alpha64_without_sim_11_6_13.txt 2>&1 &




CUDA_VISIBLE_DEVICES=0 nohup torchrun --nproc_per_node=1 finetune_LLM.py \
--model_name_or_path /home/liuzhao/demo/LLM4POI/models/model \
--bf16 True \
--output_dir outputs/ca/projector_wo_sim_llm \
--model_max_length 32768 \
--use_flash_attn True \
--data_path /home/liuzhao/demo/LLM4POI/datasets/ca/preprocessed/train_qa_pairs_kqt_200items_ca_llm.json \
--poi_embedding_path /home/liuzhao/demo/LLM4POI/datasets/ca/preprocessed/ca_POI_embeddings_mtnet_128.npy \
--projector_path /home/liuzhao/demo/LLM4POI/outputs/ca/projector_wo_sim/linear_mapping_layer.pt \
--low_rank_training True \
--num_train_epochs 3 \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 2 \
--gradient_accumulation_steps 1 \
--evaluation_strategy "no" \
--save_strategy "steps" \
--save_steps 1000 \
--save_total_limit 2 \
--learning_rate 2e-5 \
--weight_decay 0.0 \
--warmup_steps 20 \
--lr_scheduler_type "constant_with_warmup" \
--logging_steps 1 \
--deepspeed "ds_configs/stage2.json" \
--tf32 True > runs/ca/train_llm_with_projector_wo_sim_10_25_16.txt 2>&1 &



CUDA_VISIBLE_DEVICES=0 nohup torchrun --nproc_per_node=1 finetune_LLM.py \
--model_name_or_path /home/liuzhao/demo/LLM4POI/models/model \
--bf16 True \
--output_dir outputs/nyc/llm_r16_alpha64_incremental_without_sim_11_3_12 \
--model_max_length 32768 \
--use_flash_attn True \
--data_path /home/liuzhao/demo/LLM4POI/datasets/nyc/preprocessed/train_qa_pairs_kqt_200items_nyc_llm_without_sim_new.json \
--poi_embedding_path /home/liuzhao/demo/LLM4POI/datasets/nyc/preprocessed/nyc_POI_embeddings_mtnet_128.npy \
--projector_path /home/liuzhao/demo/LLM4POI/outputs/nyc/projector_incremental_without_sim_r16_alpha64_11_2_15/linear_mapping_layer_combined.pt \
--low_rank_training True \
--num_train_epochs 3 \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 2 \
--gradient_accumulation_steps 1 \
--evaluation_strategy "no" \
--save_strategy "steps" \
--save_steps 1000 \
--save_total_limit 2 \
--learning_rate 2e-5 \
--weight_decay 0.0 \
--warmup_steps 20 \
--lr_scheduler_type "constant_with_warmup" \
--logging_steps 1 \
--deepspeed "ds_configs/stage2.json" \
--tf32 True > runs/nyc/train_llm_incremental_r16_alpha32_without_sim_11_3_12.txt 2>&1 &


CUDA_VISIBLE_DEVICES=0 nohup torchrun --nproc_per_node=1 finetune_projector_incremental.py  \
--model_name_or_path /home/liuzhao/demo/LLM4POI/models/model \
--peft_dir /home/liuzhao/demo/LLM4POI/outputs/nyc_32768_300items/checkpoint-33066 \
--lora_r 16 \
--lora_alpha 64 \
--bf16 True \
--output_dir outputs/nyc/projector_incremental_without_sim_r16_alpha64_11_2_15 \
--model_max_length 32768 \
--use_flash_attn True \
--data_path /home/liuzhao/demo/LLM4POI/datasets/nyc/preprocessed/train_qa_pairs_kqt_200items_nyc_projector_without_sim_new.json \
--poi_embedding_path /home/liuzhao/demo/LLM4POI/datasets/nyc/preprocessed/nyc_POI_embeddings_mtnet_128.npy \
--low_rank_training True \
--num_train_epochs 3  \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 2  \
--gradient_accumulation_steps 1  \
--evaluation_strategy "no"  \
--save_strategy "steps"  \
--save_steps 1000  \
--save_total_limit 2  \
--learning_rate 2e-5  \
--weight_decay 0.0  \
--warmup_steps 20  \
--lr_scheduler_type "constant_with_warmup"  \
--logging_steps 1  \
--deepspeed "ds_configs/stage2.json"  \
--tf32 True > runs/nyc/train_projector_incremental_without_sim_r16_alpha64_11_2_15.txt 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python finetune_projector_incremental.py \
--model_name_or_path /home/liuzhao/demo/LLM4POI/models/model \
--peft_dir /home/liuzhao/demo/LLM4POI/outputs/nyc_32768/checkpoint-33066 \
--output_dir outputs/nyc/projector_change_trainer \
--model_max_length 32768 \
--use_flash_attn True \
--data_path /home/liuzhao/demo/LLM4POI/datasets/nyc/preprocessed/train_qa_pairs_kqt_200items_nyc_projector.json \
--poi_embedding_path /home/liuzhao/demo/LLM4POI/datasets/nyc/preprocessed/nyc_POI_embeddings_mtnet_128.npy \
--low_rank_training True \
--num_train_epochs 3 \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 2 \
--gradient_accumulation_steps 1 \
--evaluation_strategy "no" \
--save_strategy "steps" \
--save_steps 1000 \
--save_total_limit 2 \
--learning_rate 2e-5 \
--weight_decay 0.0 \
--warmup_steps 20 \
--lr_scheduler_type "constant_with_warmup" \
--logging_steps 1 \
--tf32 True > runs/nyc/train_projector_change_trainer.txt 2>&1 &



CUDA_VISIBLE_DEVICES=0 nohup torchrun --nproc_per_node=1 finetune_pre_LLM.py \
--model_name_or_path /home/liuzhao/demo/LLM4POI/models/model \
--bf16 True \
--output_dir /home/liuzhao/demo/LLM4POI/outputs/nyc/llm_r128_without_sim_incremental_11_1_13 \
--model_max_length 32768 \
--use_flash_attn True \
--data_path /home/liuzhao/demo/LLM4POI/datasets/nyc/preprocessed/train_qa_pairs_kqt_200items_nyc_llm_without_sim.json \
--poi_embedding_path /home/liuzhao/demo/LLM4POI/datasets/nyc/preprocessed/nyc_POI_embeddings_mtnet_128.npy \
--projector_path /home/liuzhao/demo/LLM4POI/outputs/nyc/projector_incremental_without_sim_10_30_15/linear_mapping_layer_combined.pt \
--low_rank_training True \
--num_train_epochs 6 \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 2 \
--gradient_accumulation_steps 1 \
--evaluation_strategy "no" \
--save_strategy "steps" \
--save_steps 1000 \
--save_total_limit 2 \
--learning_rate 2e-5 \
--weight_decay 0.0 \
--warmup_steps 20 \
--lr_scheduler_type "constant_with_warmup" \
--logging_steps 1 \
--deepspeed "ds_configs/stage2.json" \
--tf32 True > runs/nyc/train_llm_with_projector_r128_without_sim_incremental_11_1_13.txt 2>&1 &


CUDA_VISIBLE_DEVICES=0 nohup torchrun --nproc_per_node=1 finetune_LLM_2layer.py \
--model_name_or_path /home/liuzhao/demo/LLM4POI/models/model \
--bf16 True \
--output_dir outputs/nyc/llm_2layer_r32_alpha64_without_sim_11_5_13 \
--model_max_length 32768 \
--use_flash_attn True \
--data_path /home/liuzhao/demo/LLM4POI/datasets/nyc/preprocessed/train_qa_pairs_kqt_200items_nyc_llm_without_sim_new.json \
--poi_embedding_path /home/liuzhao/demo/LLM4POI/datasets/nyc/preprocessed/nyc_POI_embeddings_mtnet_128.npy \
--projector_path /home/liuzhao/demo/LLM4POI/outputs/nyc/projector_2layer_r32_alpha64_without_sim_11_4_13/checkpoint-33066/linear_mapping_layer_combined.pt \
--low_rank_training True \
--num_train_epochs 3 \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 2 \
--gradient_accumulation_steps 1 \
--evaluation_strategy "no" \
--save_strategy "steps" \
--save_steps 1000 \
--save_total_limit 2 \
--learning_rate 2e-5 \
--weight_decay 0.0 \
--warmup_steps 20 \
--lr_scheduler_type "constant_with_warmup" \
--logging_steps 1 \
--deepspeed "ds_configs/stage2.json" \
--tf32 True > runs/nyc/train_llm_2layer_r32_alpha64_without_sim_11_5_13.txt 2>&1 &


CUDA_VISIBLE_DEVICES=0 nohup torchrun --nproc_per_node=1 finetune_LLM_2layer.py \
--model_name_or_path /home/liuzhao/demo/LLM4POI/models/model \
--bf16 True \
--output_dir outputs/nyc/llm_2layer_r32_alpha64_without_sim_11_5_13 \
--model_max_length 32768 \
--use_flash_attn True \
--data_path /home/liuzhao/demo/LLM4POI/datasets/nyc/preprocessed/train_qa_pairs_kqt_200items_nyc_llm_without_sim_new.json \
--poi_embedding_path /home/liuzhao/demo/LLM4POI/datasets/nyc/preprocessed/nyc_POI_embeddings_mtnet_128.npy \
--projector_path /home/liuzhao/demo/LLM4POI/outputs/nyc/projector_2layer_r32_alpha64_without_sim_11_4_13/checkpoint-33066/linear_mapping_layer_combined.pt \
--low_rank_training True \
--num_train_epochs 3 \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 2 \
--gradient_accumulation_steps 1 \
--evaluation_strategy "no" \
--save_strategy "steps" \
--save_steps 1000 \
--save_total_limit 2 \
--learning_rate 2e-5 \
--weight_decay 0.0 \
--warmup_steps 20 \
--lr_scheduler_type "constant_with_warmup" \
--logging_steps 1 \
--deepspeed "ds_configs/stage2.json" \
--tf32 True > runs/nyc/train_llm_2layer_r32_alpha64_without_sim_11_5_13.txt 2>&1 &


CUDA_VISIBLE_DEVICES=0 nohup torchrun --nproc_per_node=1 finetune_LLM_2layer.py \
--model_name_or_path /home/liuzhao/demo/LLM4POI/models/model \
--bf16 True \
--output_dir outputs/nyc/llm_2layer_incremental_r32_alpha64_without_sim_11_7_14 \
--model_max_length 32768 \
--use_flash_attn True \
--data_path /home/liuzhao/demo/LLM4POI/datasets/nyc/preprocessed/train_qa_pairs_kqt_200items_nyc_llm_without_sim_new.json \
--poi_embedding_path /home/liuzhao/demo/LLM4POI/datasets/nyc/preprocessed/nyc_POI_embeddings_mtnet_128.npy \
--projector_path /home/liuzhao/demo/LLM4POI/outputs/nyc/projector_2layer_incremental_r32_alpha64_without_sim_11_6_12/checkpoint-33066/linear_mapping_layer_combined.pt \
--low_rank_training True \
--num_train_epochs 3 \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 2 \
--gradient_accumulation_steps 1 \
--evaluation_strategy "no" \
--save_strategy "steps" \
--save_steps 1000 \
--save_total_limit 2 \
--learning_rate 2e-5 \
--weight_decay 0.0 \
--warmup_steps 20 \
--lr_scheduler_type "constant_with_warmup" \
--logging_steps 1 \
--deepspeed "ds_configs/stage2.json" \
--tf32 True > runs/nyc/train_llm_2layer_incremental_r32_alpha64_without_sim_11_7_14.txt 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python finetune_projector_2layer_incremental.py \
--model_name_or_path /home/liuzhao/demo/LLM4POI/models/model \
--peft_dir /home/liuzhao/demo/LLM4POI/outputs/nyc_32768_300items/checkpoint-33066 \
--output_dir outputs/nyc/projector_2layer_incremental_r32_alpha64_without_sim_11_7_16 \
--projector_path /home/liuzhao/demo/LLM4POI/outputs/nyc/projector_2layer_r32_alpha64_without_sim_11_4_13/checkpoint-33066/linear_mapping_layer_combined.pt \
--lora_r 32 \
--lora_alpha 64 \
--model_max_length 32768 \
--use_flash_attn True \
--data_path /home/liuzhao/demo/LLM4POI/datasets/nyc/preprocessed/train_qa_pairs_kqt_200items_nyc_projector_without_sim_new.json \
--poi_embedding_path /home/liuzhao/demo/LLM4POI/datasets/nyc/preprocessed/nyc_POI_embeddings_mtnet_128.npy \
--low_rank_training True \
--num_train_epochs 3 \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 2 \
--gradient_accumulation_steps 1 \
--evaluation_strategy "no" \
--save_strategy "steps" \
--save_steps 1000 \
--tf32 True > runs/nyc/train_projector_2layer_incremental_r32_alpha64_without_sim_11_7_16.txt 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup torchrun --nproc_per_node=1 finetune_LLM_2layer.py \
--model_name_or_path /home/liuzhao/demo/LLM4POI/models/model \
--bf16 True \
--output_dir outputs/nyc/llm_2layer_incremental_r32_alpha64_without_sim_11_8_23 \
--model_max_length 32768 \
--use_flash_attn True \
--data_path /home/liuzhao/demo/LLM4POI/datasets/nyc/preprocessed/train_qa_pairs_kqt_200items_nyc_llm_without_sim_new.json \
--poi_embedding_path /home/liuzhao/demo/LLM4POI/datasets/nyc/preprocessed/nyc_POI_embeddings_mtnet_128.npy \
--projector_path /home/liuzhao/demo/LLM4POI/outputs/nyc/projector_2layer_incremental_r32_alpha64_without_sim_11_7_16/checkpoint-33066/linear_mapping_layer_combined.pt \
--low_rank_training True \
--num_train_epochs 3 \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 2 \
--gradient_accumulation_steps 1 \
--evaluation_strategy "no" \
--save_strategy "steps" \
--save_steps 1000 \
--save_total_limit 2 \
--learning_rate 2e-5 \
--weight_decay 0.0 \
--warmup_steps 20 \
--lr_scheduler_type "constant_with_warmup" \
--logging_steps 1 \
--deepspeed "ds_configs/stage2.json" \
--tf32 True > runs/nyc/train_llm_2layer_incremental_r32_alpha64_without_sim_11_8_23.txt 2>&1 &


CUDA_VISIBLE_DEVICES=0 nohup python finetune_projector_2layer_poi_gps.py \
--model_name_or_path /home/liuzhao/demo/LLM4POI/models/model \
--output_dir outputs/nyc/projector_poi_gps_2layer_r32_alpha64_without_sim_11_8_19 \
--lora_r 32 \
--lora_alpha 64 \
--model_max_length 32768 \
--use_flash_attn True \
--data_path /home/liuzhao/demo/LLM4POI/datasets/nyc/preprocessed/train_qa_pairs_kqt_200items_nyc_projector_gps_poi_without_sim_new.json \
--poi_embedding_path /home/liuzhao/demo/LLM4POI/datasets/nyc/preprocessed/nyc_POI_embeddings_mtnet_128.npy \
--gps_embedding_path /home/liuzhao/demo/LLM4POI/datasets/nyc/preprocessed/nyc_gps_embeddings.npy \
--low_rank_training True \
--num_train_epochs 3 \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 2 \
--gradient_accumulation_steps 1 \
--evaluation_strategy "no" \
--save_strategy "steps" \
--save_steps 1000 \
--save_total_limit 2 \
--learning_rate 2e-5 \
--weight_decay 0.0 \
--warmup_steps 20 \
--lr_scheduler_type "constant_with_warmup" \
--logging_steps 1 \
--tf32 True > runs/nyc/train_projector_poi_gps_2layer_incremental_r32_alpha64_without_sim_11_8_19.txt 2>&1 &



CUDA_VISIBLE_DEVICES=0 nohup python finetune_projector_stage2.py \
--model_name_or_path /home/liuzhao/demo/LLM4POI/models/model \
--pre_poi_mapping_layer /home/liuzhao/demo/LLM4POI/outputs/nyc/projector_incremental_without_sim_r16_alpha64_11_2_15/linear_mapping_layer_combined.pt \
--output_dir outputs/nyc/projector_stage2_r32_alpha64_without_sim_11_9_13 \
--lora_r 32 \
--lora_alpha 64 \
--model_max_length 32768 \
--use_flash_attn True \
--data_path /home/liuzhao/demo/LLM4POI/datasets/nyc/preprocessed/train_qa_pairs_kqt_200items_nyc_projector_without_sim_new.json \
--poi_embedding_path /home/liuzhao/demo/LLM4POI/datasets/nyc/preprocessed/nyc_POI_embeddings_mtnet_128.npy \
--low_rank_training True \
--num_train_epochs 3 \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 2 \
--gradient_accumulation_steps 1 \
--evaluation_strategy "no" \
--save_strategy "steps" \
--save_steps 1000 \
--tf32 True > runs/nyc/train_projector_stage2_r32_alpha64_without_sim_11_9_13.txt 2>&1 &



CUDA_VISIBLE_DEVICES=0 nohup torchrun --nproc_per_node=1 finetune_GeoEncoder.py  \
--model_name_or_path /home/liuzhao/demo/LLM4POI/models/model \
--bf16 True \
--output_dir outputs/nyc/geo_encoder_11_10_17 \
--gps_mapping_path /home/liuzhao/demo/LLM4POI/datasets/nyc/preprocessed/gps_mapping.csv \
--model_max_length 32768 \
--use_flash_attn True \
--data_path /home/liuzhao/demo/LLM4POI/datasets/nyc/preprocessed/train_qa_pairs_kqt_200items_nyc_projector_gps_without_sim_stage1.json \
--low_rank_training True \
--num_train_epochs 3  \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 2  \
--gradient_accumulation_steps 1  \
--evaluation_strategy "no"  \
--save_strategy "steps"  \
--save_steps 1000  \
--save_total_limit 2  \
--learning_rate 2e-5  \
--weight_decay 0.0  \
--warmup_steps 20  \
--lr_scheduler_type "constant_with_warmup"  \
--logging_steps 1  \
--deepspeed "ds_configs/stage2.json"  \
--tf32 True > runs/nyc/train_geo_encoder_stage1_11_10_17.txt 2>&1 &


CUDA_VISIBLE_DEVICES=0 nohup torchrun --nproc_per_node=1 --master_port=29501 finetune_FusionGeoEncoder.py \
--model_name_or_path /home/liuzhao/demo/LLM4POI/models/model \
--bf16 True \
--lora_r 32 \
--lora_alpha 64 \
--output_dir outputs/nyc/fusion_finetune_r256_512_12_15_14 \
--gps_mapping_path /home/liuzhao/demo/LLM4POI/datasets/nyc/preprocessed/gps_mapping.csv \
--model_max_length 32768 \
--use_flash_attn True \
--data_path /home/liuzhao/demo/LLM4POI/datasets/nyc/preprocessed/train_qa_pairs_kqt_200items_nyc_projector_gps_without_sim_stage1.json \
--low_rank_training True \
--num_train_epochs 3  \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 2  \
--gradient_accumulation_steps 1  \
--evaluation_strategy "no"  \
--save_strategy "steps"  \
--save_steps 1000  \
--save_total_limit 2  \
--learning_rate 2e-5  \
--weight_decay 0.0  \
--warmup_steps 20  \
--lr_scheduler_type "constant_with_warmup"  \
--logging_steps 1  \
--deepspeed "ds_configs/stage2.json"  \
--tf32 True > runs/nyc/train_fusion_geo_encoder_r256_512_12_15_14.txt 2>&1 &



CUDA_VISIBLE_DEVICES=0 nohup torchrun --nproc_per_node=1 finetune_LLM_with_GeoEncoder.py \
--model_name_or_path /home/liuzhao/demo/LLM4POI/models/model \
--bf16 True \
--output_dir outputs/nyc2/llm_stage2_geoencoder_r8_alpha32_without_sim_11_23_20 \
--model_max_length 32768 \
--use_flash_attn True \
--data_path /home/liuzhao/demo/LLM4POI/datasets/nyc/preprocessed/train_qa_pairs_kqt_200items_nyc_llm_without_sim_gps.json \
--gps_mapping_path /home/liuzhao/demo/LLM4POI/datasets/nyc/preprocessed/gps_mapping.csv \
--geoencoder_path /home/liuzhao/demo/LLM4POI/outputs/nyc2/stage2_r8_alpha32_geo_encoder_11_22_23/33066/geo_encoder_merged.pth \
--low_rank_training True \
--num_train_epochs 3 \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 2 \
--gradient_accumulation_steps 1 \
--evaluation_strategy "no" \
--save_strategy "steps" \
--save_steps 1000 \
--save_total_limit 2 \
--learning_rate 2e-5 \
--weight_decay 0.0 \
--warmup_steps 20 \
--lr_scheduler_type "constant_with_warmup" \
--logging_steps 1 \
--deepspeed "ds_configs/stage2.json" \
--tf32 True > runs/nyc/train_llm_stage2_geoencoder_r8_alpha32_without_sim_11_23_20.txt 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup torchrun --nproc_per_node=1 finetune_LLM_with_FusionGeoEncoder.py \
  --model_name_or_path /home/liuzhao/demo/LLM4POI/models/model \
  --bf16 True \
  --output_dir outputs/nyc/llm_fusiongeo_r8_alpha16_12_15_14 \
  --model_max_length 32768 \
  --use_flash_attn True \
  --data_path /home/liuzhao/demo/LLM4POI/datasets/nyc/preprocessed/train_qa_pairs_kqt_200items_nyc_llm_without_sim_gps.json \
  --gps_mapping_path /home/liuzhao/demo/LLM4POI/datasets/nyc/preprocessed/gps_mapping.csv \
  --fusion_geoencoder_path /home/liuzhao/demo/LLM4POI/outputs/nyc/fusion_finetune_r256_512_12_15_13/2/fusion_geo_encoder_merged.pth \
  --low_rank_training True \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps 1 \
  --evaluation_strategy no \
  --save_strategy steps \
  --save_steps 1000 \
  --save_total_limit 2 \
  --learning_rate 2e-5 \
  --weight_decay 0.0 \
  --warmup_steps 20 \
  --lr_scheduler_type constant_with_warmup \
  --logging_steps 1 \
  --deepspeed ds_configs/stage2.json \
  --tf32 True > runs/nyc/train_llm_fusion_12_15_14.txt 2>&1 &



CUDA_VISIBLE_DEVICES=0 nohup torchrun --nproc_per_node=1 finetune_LLM_with_both.py \
--model_name_or_path /home/liuzhao/demo/LLM4POI/models/model \
--bf16 True \
--output_dir outputs/nyc/llm_both_without_sim_new2_11_21_13 \
--model_max_length 32768 \
--use_flash_attn True \
--data_path /home/liuzhao/demo/LLM4POI/datasets/nyc/preprocessed/train_qa_pairs_kqt_200items_nyc_llm_without_sim_gps_poi_new2.json \
--poi_embedding_path /home/liuzhao/demo/LLM4POI/datasets/nyc/preprocessed/nyc_POI_embeddings_mtnet_128.npy \
--projector_path /home/liuzhao/demo/LLM4POI/outputs/nyc/pre_both_11_18_14/33066/linear_mapping_layer_combined.pt \
--gps_mapping_path /home/liuzhao/demo/LLM4POI/datasets/nyc/preprocessed/gps_mapping.csv \
--geoencoder_path /home/liuzhao/demo/LLM4POI/outputs/nyc/pre_both_11_18_14/33066/geo_encoder_merged.pth \
--low_rank_training True \
--num_train_epochs 3 \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 2 \
--gradient_accumulation_steps 1 \
--evaluation_strategy "no" \
--save_strategy "steps" \
--save_steps 1000 \
--save_total_limit 2 \
--learning_rate 2e-5 \
--weight_decay 0.0 \
--warmup_steps 20 \
--lr_scheduler_type "constant_with_warmup" \
--logging_steps 1 \
--deepspeed "ds_configs/stage2.json" \
--tf32 True > runs/nyc/train_llm_both_without_sim_new2_11_21_13.txt 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup torchrun --nproc_per_node=1 finetune_pre_both_incremental.py  \
--model_name_or_path /home/liuzhao/demo/LLM4POI/models/model \
--peft_dir /home/liuzhao/demo/LLM4POI/outputs/nyc_32768_300items/checkpoint-33066 \
--bf16 True \
--output_dir outputs/nyc2/pre_both_incremental_stage2_r8_32_128_256_11_25_16 \
--linear_mapping_path /home/liuzhao/demo/LLM4POI/outputs/nyc/projector_incremental_without_sim_10_30_15/linear_mapping_layer_combined.pt \
--geo_encoder_path /home/liuzhao/demo/LLM4POI/outputs/nyc2/stage2_r8_alpha32_geo_encoder_11_22_23/33066/geo_encoder_merged.pth \
--poi_embedding_path /home/liuzhao/demo/LLM4POI/datasets/nyc/preprocessed/nyc_POI_embeddings_mtnet_128.npy \
--gps_mapping_path /home/liuzhao/demo/LLM4POI/datasets/nyc/preprocessed/gps_mapping.csv \
--lora_r_main_model 8 \
--lora_alpha_main_model 32 \
--lora_r_geo_encoder 128 \
--lora_alpha_geo_encoder 256 \
--model_max_length 32768 \
--use_flash_attn True \
--data_path /home/liuzhao/demo/LLM4POI/datasets/nyc/preprocessed/train_qa_pairs_kqt_200items_nyc_llm_without_sim_gps_poi_wocoor.json \
--low_rank_training True \
--num_train_epochs 3  \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 2  \
--gradient_accumulation_steps 1  \
--evaluation_strategy "no"  \
--save_strategy "steps"  \
--save_steps 1000  \
--save_total_limit 2  \
--learning_rate 2e-5  \
--weight_decay 0.0  \
--warmup_steps 20  \
--lr_scheduler_type "constant_with_warmup"  \
--logging_steps 1  \
--deepspeed "ds_configs/stage2.json"  \
--tf32 True > runs/nyc/train_pre_both_incremental_stage2_r8_32_128_256_11_25_16.txt 2>&1 &


CUDA_VISIBLE_DEVICES=0 nohup torchrun --nproc_per_node=1 finetune_pre_both_incremental_fusion.py  \
--model_name_or_path /home/liuzhao/demo/LLM4POI/models/model \
--peft_dir /home/liuzhao/demo/LLM4POI/outputs/nyc_32768_300items/checkpoint-33066 \
--bf16 True \
--output_dir outputs/nyc2/pre_both_incremental_fusion_r8_32_128_256_12_23_16 \
--linear_mapping_path /home/liuzhao/demo/LLM4POI/outputs/nyc/projector_incremental_without_sim_10_30_15/linear_mapping_layer_combined.pt \
--geo_encoder_path /home/liuzhao/demo/LLM4POI/outputs/nyc/fusion_finetune_r64_128_12_15_14/19000/fusion_geo_encoder_merged.pth \
--poi_embedding_path /home/liuzhao/demo/LLM4POI/datasets/nyc/preprocessed/nyc_POI_embeddings_mtnet_128.npy \
--gps_mapping_path /home/liuzhao/demo/LLM4POI/datasets/nyc/preprocessed/gps_mapping.csv \
--lora_r_main_model 8 \
--lora_alpha_main_model 32 \
--lora_r_geo_encoder 128 \
--lora_alpha_geo_encoder 256 \
--model_max_length 32768 \
--use_flash_attn True \
--data_path /home/liuzhao/demo/LLM4POI/datasets/nyc/preprocessed/train_qa_pairs_kqt_200items_nyc_llm_without_sim_gps_poi_wocoor.json \
--low_rank_training True \
--num_train_epochs 3  \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 2  \
--gradient_accumulation_steps 1  \
--evaluation_strategy "no"  \
--save_strategy "steps"  \
--save_steps 1  \
--save_total_limit 2  \
--learning_rate 2e-5  \
--weight_decay 0.0  \
--warmup_steps 20  \
--lr_scheduler_type "constant_with_warmup"  \
--logging_steps 1  \
--deepspeed "ds_configs/stage2.json"  \
--tf32 True > runs/nyc/train_pre_both_incremental_fusion_r8_32_128_256_12_23_16.txt 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup torchrun --nproc_per_node=1 finetune_LLM_with_both_fusion.py \
--model_name_or_path /home/liuzhao/demo/LLM4POI/models/model \
--bf16 True \
--output_dir outputs/nyc2/llm_pre_both_incremental_fusion_r8_32_128_256_without_sim_wocoor_12_23_16 \
--model_max_length 32768 \
--use_flash_attn True \
--data_path /home/liuzhao/demo/LLM4POI/datasets/nyc/preprocessed/train_qa_pairs_kqt_200items_nyc_llm_without_sim_gps_poi_wocoor.json \
--poi_embedding_path /home/liuzhao/demo/LLM4POI/datasets/nyc/preprocessed/nyc_POI_embeddings_mtnet_128.npy \
--linear_mapping_path /home/liuzhao/demo/LLM4POI/outputs/nyc2/pre_both_incremental_fusion_r8_32_128_256_12_23_16/23/linear_mapping_layer_combined.pt \
--gps_mapping_path /home/liuzhao/demo/LLM4POI/datasets/nyc/preprocessed/gps_mapping.csv \
--fusion_geoencoder_path /home/liuzhao/demo/LLM4POI/outputs/nyc2/pre_both_incremental_fusion_r8_32_128_256_12_23_16/23/geo_encoder_merged.pth \
--low_rank_training True \
--num_train_epochs 3 \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 2 \
--gradient_accumulation_steps 1 \
--evaluation_strategy "no" \
--save_strategy "steps" \
--save_steps 1000 \
--save_total_limit 2 \
--learning_rate 2e-5 \
--weight_decay 0.0 \
--warmup_steps 20 \
--lr_scheduler_type "constant_with_warmup" \
--logging_steps 1 \
--deepspeed "ds_configs/stage2.json" \
--tf32 True > runs/nyc/train_llm_pre_both_incremental_fusion_r8_32_128_256_without_sim_wocoor_12_23_16.txt 2>&1 &



























































































## test
run
```
python eval_next_poi.py --model_path {your_model_path}--dataset_name {DATASET_NAME} --output_dir {your_finetuned_model} --test_file "test_qa_pairs_kqt.txt"
```

```
nohup torchrun --nproc_per_node=7 eval_5\&10.py > runs/log_eval_510_9_14_9.txt &

c
```
CUDA_VISIBLE_DEVICES=0 nohup python eval_new.py \
--batch_size 8 \
--base_model /home/liuzhao/demo/LLM4POI/models/model \
--cache_dir ./cache \
--seq_len 32768 \
--context_size 32768 \
--peft_model None \
--flash_attn True \
--model_path /home/liuzhao/demo/LLM4POI/models/model \
--data_path /home/liuzhao/demo/LLM4POI/datasets \
--output_dir /home/liuzhao/demo/LLM4POI/outputs/output_dir_embedding_32768_mtnetv1/checkpoint-109122 \
--dataset_name ca \
--test_file test_qa_pairs_kqt_new2_poi.txt > runs/log_eval_new2poi_10_7_10.txt 2>&1 &


CUDA_VISIBLE_DEVICES=0 nohup python eval_next_poi.py \
--batch_size 8 \
--base_model /home/liuzhao/demo/LLM4POI/models/model \
--cache_dir ./cache \
--seq_len 32768 \
--context_size 32768 \
--peft_model None \
--flash_attn True \
--model_path /home/liuzhao/demo/LLM4POI/models/model \
--data_path /home/liuzhao/demo/LLM4POI/datasets \
--output_dir /home/liuzhao/demo/LLM4POI/outputs/nyc_32768/checkpoint-33066 \
--dataset_name nyc \
--test_file test_qa_pairs_kqt_200items.txt > runs/log_eval_nyc_32768_10_10_13.txt 2>&1 &


CUDA_VISIBLE_DEVICES=0 nohup python test_next_poi.py \
--batch_size 8 \
--base_model /home/liuzhao/demo/LLM4POI/models/model \
--cache_dir ./cache \
--seq_len 32768 \
--context_size 32768 \
--flash_attn True \
--data_path /home/liuzhao/demo/LLM4POI/datasets \
--dataset_name nyc \
--test_file test_qa_pairs_kqt_200items.txt \
--test_type base > runs/log_test_nyc_32768_base.txt 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python test_next_poi.py \
--batch_size 8 \
--base_model /home/liuzhao/demo/LLM4POI/models/model \
--cache_dir ./cache \
--seq_len 32768 \
--context_size 32768 \
--flash_attn True \
--data_path /home/liuzhao/demo/LLM4POI/datasets \
--dataset_name nyc \
--test_file test_qa_pairs_kqt_200items.txt \
--test_type projector \
--poi_embedding_path /home/liuzhao/demo/LLM4POI/datasets/nyc/preprocessed/nyc_POI_embeddings_mtnet_128.npy \
--projector_path /home/liuzhao/demo/LLM4POI/outputs/nyc/projector/linear_mapping_layer.pt \
> runs/log_test_nyc_32768_projector.txt 2>&1 &



CUDA_VISIBLE_DEVICES=0 nohup python eval_next_poi.py \
--batch_size 8 \
--base_model /home/liuzhao/demo/LLM4POI/models/model \
--cache_dir ./cache \
--seq_len 32768 \
--context_size 32768 \
--peft_model None \
--flash_attn True \
--model_path /home/liuzhao/demo/LLM4POI/models/model \
--data_path /home/liuzhao/demo/LLM4POI/datasets \
--output_dir /home/liuzhao/demo/LLM4POI/outputs/nyc_32768_300items/checkpoint-33066 \
--dataset_name nyc \
--test_file test_qa_pairs_kqt_300items.txt > runs/log_eval_nyc_300items_10_10_11.txt 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python eval_new.py \
--batch_size 8 \
--base_model /home/liuzhao/demo/LLM4POI/models/model \
--cache_dir ./cache \
--seq_len 32768 \
--context_size 32768 \
--peft_model None \
--flash_attn True \
--model_path /home/liuzhao/demo/LLM4POI/models/model \
--data_path /home/liuzhao/demo/LLM4POI/datasets \
--output_dir /home/liuzhao/demo/LLM4POI/outputs/nyc_output_dir_v4_200items_geo/checkpoint-33066 \
--dataset_name nyc \
--test_file test_qa_pairs_kqt_200items_geo.txt > runs/log_eval_nyc_200items_geo_10_10_11.txt 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python eval_new.py \
--batch_size 8 \
--base_model /home/liuzhao/demo/LLM4POI/models/model \
--cache_dir ./cache \
--seq_len 32768 \
--context_size 32768 \
--peft_model None \
--flash_attn True \
--model_path /home/liuzhao/demo/LLM4POI/models/model \
--data_path /home/liuzhao/demo/LLM4POI/datasets \
--output_dir /home/liuzhao/demo/LLM4POI/outputs/nyc_output_dir_v4_200items_geo_poi/checkpoint-33066 \
--dataset_name nyc \
--test_file test_qa_pairs_kqt_200items_geo_poi.txt > runs/log_eval_nyc_200items_geo_poi_10_11_12.txt 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python eval_new.py \
--batch_size 8 \
--base_model /home/liuzhao/demo/LLM4POI/models/model \
--cache_dir ./cache \
--seq_len 32768 \
--context_size 32768 \
--peft_model None \
--flash_attn True \
--model_path /home/liuzhao/demo/LLM4POI/models/model \
--data_path /home/liuzhao/demo/LLM4POI/datasets \
--output_dir /home/liuzhao/demo/LLM4POI/outputs/ca/merged_poi/checkpoint-109122 \
--dataset_name ca \
--test_file test_qa_pairs_kqt_200items_ca_merged_poi.txt > runs/ca/eval_merged_poi_10_21_22.txt 2>&1 &



CUDA_VISIBLE_DEVICES=0 nohup python test_next_poi.py \
--batch_size 8 \
--base_model /home/liuzhao/demo/LLM4POI/models/model \
--cache_dir ./cache \
--seq_len 32768 \
--context_size 32768 \
--flash_attn True \
--model_path /home/liuzhao/demo/LLM4POI/models/model \
--data_path /home/liuzhao/demo/LLM4POI/datasets \
--dataset_name ca \
--test_file test_qa_pairs_kqt_200items_ca_llm.txt \
--test_type base > runs/ca/eval_base_10_26_16.log 2>&1 &


CUDA_VISIBLE_DEVICES=0 nohup python eval_test.py \
--batch_size 8 \
--base_model /home/liuzhao/demo/LLM4POI/models/model \
--cache_dir ./cache \
--seq_len 32768 \
--context_size 32768 \
--flash_attn True \
--model_path /home/liuzhao/demo/LLM4POI/models/model \
--data_path /home/liuzhao/demo/LLM4POI/datasets \
--dataset_name ca \
--test_file test_qa_pairs_kqt_200items_ca_llm.txt \
--test_type projector \
--poi_embedding_path /home/liuzhao/demo/LLM4POI/datasets/ca/preprocessed/ca_POI_embeddings_mtnet_128.npy \
--projector_path /home/liuzhao/demo/LLM4POI/outputs/ca/projector_wo_sim/linear_mapping_layer.pt > runs/ca/eval_projector_demo_10_26_16.log 2>&1 &



CUDA_VISIBLE_DEVICES=0 nohup python eval_test.py \
--batch_size 8 \
--base_model /home/liuzhao/demo/LLM4POI/models/model \
--cache_dir ./cache \
--seq_len 32768 \
--context_size 32768 \
--flash_attn True \
--model_path /home/liuzhao/demo/LLM4POI/models/model \
--data_path /home/liuzhao/demo/LLM4POI/datasets \
--output_dir /home/liuzhao/demo/LLM4POI/outputs/ca/merged_poi/checkpoint-109122 \
--dataset_name ca \
--test_file test_qa_pairs_kqt_200items_ca_llm.txt \
--test_type llm \
--poi_embedding_path /home/liuzhao/demo/LLM4POI/datasets/ca/preprocessed/ca_POI_embeddings_mtnet_128.npy \
--projector_path /home/liuzhao/demo/LLM4POI/outputs/ca/projector_wo_sim/linear_mapping_layer.pt > runs/ca/eval_llm_10_26_16.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python test_next_poi_2layer.py \
--batch_size 8 \
--base_model /home/liuzhao/demo/LLM4POI/models/model \
--output_dir /home/liuzhao/demo/LLM4POI/outputs/nyc/llm_2layer_incremental_r32_alpha64_without_sim_11_8_23/checkpoint-33066 \
--cache_dir ./cache \
--seq_len 32768 \
--context_size 32768 \
--flash_attn True \
--model_path /home/liuzhao/demo/LLM4POI/models/model \
--data_path /home/liuzhao/demo/LLM4POI/datasets \
--dataset_name nyc \
--test_file test_qa_pairs_kqt_200items_nyc_llm_without_sim_new.txt \
--test_type llm \
--poi_embedding_path /home/liuzhao/demo/LLM4POI/datasets/nyc/preprocessed/nyc_POI_embeddings_mtnet_128.npy \
--projector_path /home/liuzhao/demo/LLM4POI/outputs/nyc/projector_2layer_incremental_r32_alpha64_without_sim_11_7_16/checkpoint-33066/linear_mapping_layer_combined.pt \
--use_random_projector False > runs/nyc/eval_llm_2layer_incremental_r32_alpha64_without_sim_11_10_14.log 2>&1 &



CUDA_VISIBLE_DEVICES=0 nohup python test_next_poi.py \
--batch_size 8 \
--base_model /home/liuzhao/demo/LLM4POI/models/model \
--output_dir /home/liuzhao/demo/LLM4POI/outputs/nyc/llm_r16_alpha64_incremental_without_sim_11_3_12/checkpoint-33066 \
--cache_dir ./cache \
--seq_len 32768 \
--context_size 32768 \
--flash_attn True \
--model_path /home/liuzhao/demo/LLM4POI/models/model \
--data_path /home/liuzhao/demo/LLM4POI/datasets \
--dataset_name nyc \
--test_file test_qa_pairs_kqt_200items_nyc_llm_without_sim_new.txt \
--test_type llm \
--poi_embedding_path /home/liuzhao/demo/LLM4POI/datasets/nyc/preprocessed/nyc_POI_embeddings_mtnet_128.npy \
--projector_path /home/liuzhao/demo/LLM4POI/outputs/nyc/projector_incremental_without_sim_r16_alpha64_11_2_15/linear_mapping_layer_combined.pt \
--use_random_projector False > runs/nyc/eval_llm_incremental_r16_alpha64_without_sim_11_4_12.txt 2>&1 &



CUDA_VISIBLE_DEVICES=0 nohup python test_next_gps.py \
--batch_size 8 \
--base_model /home/liuzhao/demo/LLM4POI/models/model \
--output_dir /home/liuzhao/demo/LLM4POI/outputs/nyc/llm_geoencoder_r8_alpha32_without_sim_11_12_12/checkpoint-33066 \
--cache_dir ./cache \
--seq_len 32768 \
--context_size 32768 \
--flash_attn True \
--model_path /home/liuzhao/demo/LLM4POI/models/model \
--data_path /home/liuzhao/demo/LLM4POI/datasets \
--dataset_name nyc \
--test_file test_qa_pairs_kqt_200items_nyc_llm_without_sim_gps.txt \
--test_type llm \
--gps_mapping_path /home/liuzhao/demo/LLM4POI/datasets/nyc/preprocessed/gps_mapping.csv \
--geoencoder_path /home/liuzhao/demo/LLM4POI/outputs/nyc/geo_encoder_11_10_17/33066/geo_encoder_merged.pth \
--use_random_projector False > runs/nyc/eval_llm_gps_r8_alpha32_without_sim_11_14_9.txt 2>&1 &


CUDA_VISIBLE_DEVICES=0 nohup python test_next_both.py \
--batch_size 8 \
--base_model /home/liuzhao/demo/LLM4POI/models/model \
--output_dir /home/liuzhao/demo/LLM4POI/outputs/nyc/llm_both_without_sim_new2_11_21_13/checkpoint-33066 \
--cache_dir ./cache \
--seq_len 32768 \
--context_size 32768 \
--flash_attn True \
--model_path /home/liuzhao/demo/LLM4POI/models/model \
--data_path /home/liuzhao/demo/LLM4POI/datasets \
--dataset_name nyc \
--test_file test_qa_pairs_kqt_200items_nyc_llm_without_sim_gps_poi_new2.txt \
--test_type llm \
--poi_embedding_path /home/liuzhao/demo/LLM4POI/datasets/nyc/preprocessed/nyc_POI_embeddings_mtnet_128.npy \
--projector_path /home/liuzhao/demo/LLM4POI/outputs/nyc/pre_both_11_18_14/33066/linear_mapping_layer_combined.pt \
--gps_mapping_path /home/liuzhao/demo/LLM4POI/datasets/nyc/preprocessed/gps_mapping.csv \
--geoencoder_path /home/liuzhao/demo/LLM4POI/outputs/nyc/pre_both_11_18_14/33066/geo_encoder_merged.pth \
--use_random_projector False > runs/nyc/eval_llm_pre_both_test_without_sim_new2_11_22_22.txt 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python test_next_fusion_gps.py \
  --batch_size 8 \
  --base_model /home/liuzhao/demo/LLM4POI/models/model \
  --output_dir /home/liuzhao/demo/LLM4POI/outputs/nyc/llm_fusiongeo_r8_alpha32_12_15 \
  --cache_dir ./cache \
  --seq_len 32768 \
  --context_size 32768 \
  --flash_attn True \
  --model_path /home/liuzhao/demo/LLM4POI/models/model \
  --data_path /home/liuzhao/demo/LLM4POI/datasets \
  --dataset_name nyc \
  --test_file test_qa_pairs_kqt_200items_nyc_llm_without_sim_gps.txt \
  --test_type llm \
  --gps_mapping_path /home/liuzhao/demo/LLM4POI/datasets/nyc/preprocessed/gps_mapping.csv \
  --fusion_geoencoder_path /home/liuzhao/demo/LLM4POI/outputs/nyc/fusion_finetune_r256_512_12_15_13/2/fusion_geo_encoder_merged.pth \
  --use_random_projector False > runs/nyc/eval_llm_fusion_geo.txt 2>&1 &

## Acknowledgement
This code is developed based on [STHGCN](https://github.com/ant-research/Spatio-Temporal-Hypergraph-Model) and [LongLoRA](https://github.com/dvlab-research/LongLoRA?tab=readme-ov-file).
## Citation
If you find our work useful, please consider cite our paper with following:
```
@inproceedings{li-2024-large,
author = {Li, Peibo and de Rijke, Maarten and Xue, Hao and Ao, Shuang and Song, Yang and Salim, Flora D.},
booktitle = {SIGIR 2024: 47th international ACM SIGIR Conference on Research and Development in Information Retrieval},
date-added = {2024-03-26 23:47:40 +0000},
date-modified = {2024-03-26 23:48:47 +0000},
month = {July},
publisher = {ACM},
title = {Large Language Models for Next Point-of-Interest Recommendation},
year = {2024}}
```
