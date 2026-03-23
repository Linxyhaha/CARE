DATASET=Video_Games

export CUDA_LAUNCH_BLOCKING=1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH 

tokenizer=TIGER # backbone model

model_class=Qwen2.5-0.5B

base_model=/your/path/to/pretrained/model/${model_class} # TODO

# NOTE: for customized query numbers and progressive attention enabling - current example with item identifier length = 4
query_list_list=("1 1 1 1") # explanation: "1" 
progressive_list_list=("1 1 1 1") # explanation: "1" means using progressive attention mask, "0" means using standard causal attention mask; each entry refers to a reasoning stage

progressive_attn="--progressive_attn"
only_train_response="--only_train_response"

(
mkdir -p ./log/train
mkdir -p ./log/test

for lr in 5e-5 
do
    for wd in 0.01  
    do
        for idx in "${!query_list_list[@]}"
        do
            query_list="${query_list_list[$idx]}"
            progressive_list="${progressive_list_list[$idx]}"

            querys=$(echo $query_list | tr ' ' '_')
            adaptives=$(echo $progressive_list | tr ' ' '_')

            for div_query_scale in 0.3
            do  
                suffix=${tokenizer}-${querys}querys-${adaptives}progressive
                suffix=${DATASET}-${tokenizer}-${model_class}-${lr}lr-${wd}wd-${div_query_scale}divquery_${suffix}${progressive_attn#--}
                logfile=./log/train/${suffix}-train.log
                OUTPUT_DIR=./ckpt/${DATASET}_${suffix}

                mkdir -p ${OUTPUT_DIR}

                echo "start running..."
                accelerate launch --config_file zero2-8gpu.yaml \
                --main_process_port $((RANDOM % 9999 + 20000)) train.py \
                    --base_model ${base_model} \
                    --output_dir $OUTPUT_DIR \
                    --dataset $DATASET \
                    --query_div_scale ${div_query_scale} \
                    ${progressive_attn} \
                    --query_list ${query_list} \
                    --progressive_list ${progressive_list} \
                    --per_device_batch_size 2 \
                    --learning_rate $lr \
                    --epochs 15 \
                    --weight_decay $wd \
                    --save_and_eval_strategy steps \
                    --save_and_eval_steps 200 \
                    --valid_sample -1 \
                    --warmup_steps 100 \
                    --test_batch_size 32 \
                    --num_beams 20 \
                    --special_token_for_answer "|start_of_answer|" \
                    ${only_train_response} \
                    --index_file .${tokenizer}-index.json \
                    #&> >(tee -a ${logfile}) 
                wait


                # inference
                CKPT_PATH=${OUTPUT_DIR}
                logfile=./log/indepth/test/${suffix}-test.log
                
                echo "Running inference for checkpoint: ${CKPT_PATH}"
                echo "Saving logs to: ${logfile}"

                CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=1 --master_port=19324 inference.py \
                    --dataset $DATASET \
                    --base_model /your/path/to/pretrained/model/${model_class} \
                    --ckpt_path $CKPT_PATH \
                    --query_list ${query_list} \
                    --progressive_list ${progressive_list} \
                    ${progressive_attn} \
                    --test_batch_size 50 \
                    --num_beams 20 \
                    --index_file .${tokenizer}-index.json \
                    --filter_items \
                    > ${logfile}

            done
        done
    done
done
) #&
