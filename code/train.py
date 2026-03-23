import argparse
import os
import sys
import torch

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    print(gpu_name)

import transformers
from transformers import Qwen2Tokenizer, Qwen2Config, Qwen2ForCausalLM, EarlyStoppingCallback
from utils import *
from collator import Collator_DecoderOnly_manual, TestCollator, Collator_Reasoning_Training
from torch.utils.data import DataLoader
import ipdb
from tqdm import tqdm
from models import CARE

def train(args):

    set_seed(args.seed)
    ensure_dir(args.output_dir)

    args.progressive_list = [bool(_) for _ in args.progressive_list]

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    local_rank = int(os.environ.get("LOCAL_RANK") or 0)
    if local_rank == 0:
        print(vars(args))

    if ddp:
        device_map = {"": local_rank}

    # Check if parameter passed or if set within environ
    use_wandb = len(args.wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(args.wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = args.wandb_project
    if len(args.wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = args.wandb_watch
    # os.environ["WANDB_DISABLED"] = "true" 

    config = Qwen2Config.from_pretrained(args.base_model)
    tokenizer = Qwen2Tokenizer.from_pretrained(args.base_model,model_max_length=args.model_max_length, padding_side="left",)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    train_data, valid_data = load_datasets(args)

    if args.valid_sample > -1:
        valid_data.slice_data(args.valid_sample)

    add_num = tokenizer.add_tokens(train_data.datasets[0].get_new_tokens())
    config.vocab_size = len(tokenizer)
    config.query_list = args.query_list
    config.progressive_list = args.progressive_list


    if local_rank == 0:
        print("add {} new token.".format(add_num))
        print("data num:", len(train_data))
        tokenizer.save_pretrained(args.output_dir)
        config.save_pretrained(args.output_dir) 

    collator = Collator_Reasoning_Training(args, tokenizer) 

    dtype = torch.bfloat16 
    bf16=True

    model = CARE.from_pretrained(args.base_model, query_list=args.query_list, query_div_scale=args.query_div_scale, progressive_attn = args.progressive_attn, progressive_list=args.progressive_list, attention_strategy=args.attention_strategy, torch_dtype=dtype, device_map=device_map)

    # update model other args 
    model.update_config(args.query_list, args.progressive_list)
    
    # update model training forward
    model.forward = model.forward_training 

    model.resize_token_embeddings(len(tokenizer))

    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True
    
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=valid_data,
        args=transformers.TrainingArguments(
            seed=args.seed,
            per_device_train_batch_size=args.per_device_batch_size,
            per_device_eval_batch_size=args.per_device_batch_size,
            warmup_steps=args.warmup_steps,
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            lr_scheduler_type=args.lr_scheduler_type,
            run_name=args.wandb_run_name,
            bf16=bf16,
            logging_steps=args.logging_step,
            optim=args.optim,
            gradient_checkpointing=False,
            evaluation_strategy=args.save_and_eval_strategy,
            save_strategy=args.save_and_eval_strategy,
            eval_steps=args.save_and_eval_steps,
            save_steps=args.save_and_eval_steps,
            output_dir=args.output_dir,
            save_total_limit=1,
            load_best_model_at_end=True,
            ddp_find_unused_parameters=False if ddp else None,
            save_safetensors=False
        ),
        tokenizer=tokenizer,
        data_collator=collator,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=5)],
    )

    model.config.use_cache = False

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    # trainer.train(resume_from_checkpoint=args.resume_from_checkpoint,)

    if args.epochs:
        if int(os.environ.get("LOCAL_RANK"))==0: # real exp
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            print(f"The best model is saved at {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LLMRec')
    parser = parse_global_args(parser)
    parser = parse_train_args(parser)
    parser = parse_test_args(parser)
    parser = parse_dataset_args(parser)

    args = parser.parse_args()

    train(args)
