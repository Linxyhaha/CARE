
import argparse
import json
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
import sys
from typing import List, Dict

import torch
import ipdb
import time 

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    print(gpu_name)

import transformers
transformers.logging.set_verbosity_error()
from peft import PeftModel 
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Qwen2Tokenizer, Qwen2Config, Qwen2ForCausalLM
from utils import *
from collator import TestCollator
from evaluate import get_topk_results, get_metrics_results
from generation_trie import Trie

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

from models import Qwen2WithReasoning, Qwen2WithReasoning_Query, Qwen2WithReasoning_ParallelQuery_Custom, Qwen2Model_AdaptiveAttn

from collections import Counter

def compute_div_or_ratio(all_pred_list, Ks=[5,10,20],top=3):
    """
    all_pred_list: List[List[item]]，每个用户一个长度为20的推荐结果
    """
    results = {}

    for K in Ks:
        # flatten 前 K 的推荐结果
        flat_preds = []
        for pred in all_pred_list:
            flat_preds.extend(pred[:K])  # 取 top-K
        
        total = len(flat_preds)
        
        # ===== DivRatio@K =====
        unique_items = len(set(flat_preds))
        divratio = unique_items / total

        # ===== ORRatio@K =====
        counter = Counter(flat_preds)
        top3_count = sum([cnt for item, cnt in counter.most_common(top)])
        orratio = top3_count / total

        results[f"DivRatio@{K}"] = round(divratio, 4)
        results[f"ORRatio@{K}"] = round(orratio, 4)

    return results

def print_div_or_results(div_or_results):
    """
    div_or_results: dict like {
        'DivRatio@5': x, 'DivRatio@10': y, 'DivRatio@20': z,
        'ORRatio@5': a, 'ORRatio@10': b, 'ORRatio@20': c
    }
    """
    div_keys = [f"DivRatio@5", f"DivRatio@10", f"DivRatio@20"]
    or_keys  = [f"ORRatio@5", f"ORRatio@10", f"ORRatio@20"]

    div_values = [str(div_or_results[k]) for k in div_keys]
    or_values  = [str(div_or_results[k]) for k in or_keys]

    print("[Test]: DivRatio: {} ORRatio: {}".format(
        '-'.join(div_values),
        '-'.join(or_values)
    ))

class Trie(object):
    def __init__(self, sequences: List[List[int]] = []):
        self.trie_dict = {}
        self.len = 0
        if sequences:
            for sequence in sequences:
                Trie._add_to_trie(sequence, self.trie_dict)
                self.len += 1

        self.append_trie = None
        self.bos_token_id = None

    def append(self, trie, bos_token_id):
        self.append_trie = trie
        self.bos_token_id = bos_token_id

    def add(self, sequence: List[int]):
        Trie._add_to_trie(sequence, self.trie_dict)
        self.len += 1

    def get(self, prefix_sequence: List[int]):
        return Trie._get_from_trie(
            prefix_sequence, self.trie_dict, self.append_trie, self.bos_token_id
        )

    @staticmethod
    def load_from_dict(trie_dict):
        trie = Trie()
        trie.trie_dict = trie_dict
        trie.len = sum(1 for _ in trie)
        return trie

    @staticmethod
    def _add_to_trie(sequence: List[int], trie_dict: Dict):
        if sequence:
            if sequence[0] not in trie_dict:
                trie_dict[sequence[0]] = {}
            Trie._add_to_trie(sequence[1:], trie_dict[sequence[0]])

    @staticmethod
    def _get_from_trie(
        prefix_sequence: List[int],
        trie_dict: Dict,
        append_trie=None,
        bos_token_id: int = None,
    ):
        if len(prefix_sequence) == 0:
            output = list(trie_dict.keys())
            if append_trie and bos_token_id in output:
                output.remove(bos_token_id)
                output += list(append_trie.trie_dict.keys())
            return output
        elif prefix_sequence[0] in trie_dict:
            return Trie._get_from_trie(
                prefix_sequence[1:],
                trie_dict[prefix_sequence[0]],
                append_trie,
                bos_token_id,
            )
        else:
            if append_trie:
                return append_trie.get(prefix_sequence)
            else:
                return []

    def __iter__(self):
        def _traverse(prefix_sequence, trie_dict):
            if trie_dict:
                for next_token in trie_dict:
                    yield from _traverse(
                        prefix_sequence + [next_token], trie_dict[next_token]
                    )
            else:
                yield prefix_sequence

        return _traverse([], self.trie_dict)

    def __len__(self):
        return self.len

    def __getitem__(self, value):
        return self.get(value)


def prefix_allowed_tokens_fn(candidate_trie, tokenizer, special_token_for_answer="|start_of_answer|"):

    sep = tokenizer(special_token_for_answer)["input_ids"]
    bos = [1]
    def prefix_allowed_tokens(batch_id, sentence):
        for i in range(len(sentence),-1,-1):
            if sentence[i-len(sep):i].tolist() == sep:
                # sentence_ = sentence[i:].tolist() 
                if i == len(sentence):
                    sentence_ = bos
                else:
                    sentence_ = [1] + sentence[i:].tolist()
        trie_out = candidate_trie.get(sentence_)
        return trie_out

    return prefix_allowed_tokens

def get_greedy_prefix_allowed_tokens_fn(indices, tokenizer, special_token_for_answer="|start_of_answer|"):

    allowed_tokens = {}
    for index in indices.values(): # 对于每个item
        for i, token in enumerate(index): # 对于每个item 的每个位置的code
            token_id = tokenizer(token)["input_ids"][0] # qwen没有bos，拿到code 的token id
            if i not in allowed_tokens.keys(): # 所有第i个位置的dictionary
                allowed_tokens[i] = set() # 给第i个位置建个dict
            allowed_tokens[i].add(token_id) # 生成的第i个位置的有效token id加进来
    allowed_tokens[len(allowed_tokens.keys())] = set([tokenizer.eos_token_id]) # 末尾添加eos

    sep = tokenizer(special_token_for_answer)["input_ids"][1:]
    # sep=[13]

    def prefix_allowed_tokens_fn(batch_id, sentence):
        sentence = sentence.tolist()
        reversed_sent = sentence[::-1]
        for i in range(len(reversed_sent)):
            if reversed_sent[i:i + len(sep)] == sep[::-1]:
                # print(list(self.allowed_tokens[i]))
                try:
                    return list(allowed_tokens[i])
                except:
                     print("problem here")

    return prefix_allowed_tokens_fn

def get_topk_results(predictions, scores, targets, k, all_items=None, special_token_for_answer="|start_of_answer|"):
    results = []
    B = len(targets)
    predictions = [_.split(special_token_for_answer)[-1] for _ in predictions]
    # predictions = [_.strip().replace(" ","") for _ in predictions]
    # print(predictions)##################
    if all_items is not None:
        cnt = 0
        for i, seq in enumerate(predictions):
            if seq not in all_items:
                scores[i] = -1000
                cnt += 1
        # print(f"*** invalid items for a user: {cnt}") 
    batch_pred = []
    # print(scores)
    for b in range(B):
        batch_seqs = predictions[b * k: (b + 1) * k]
        batch_scores = scores[b * k: (b + 1) * k]

        pairs = [(a, b) for a, b in zip(batch_seqs, batch_scores)]
        # print(pairs)
        results = sorted(pairs, key=lambda x: x[1], reverse=True)
        pred = [r[0] for r in results]
        batch_pred.append(pred)
        # ipdb.set_trace()
    return batch_pred


def test(args):
    set_seed(args.seed)
    print(vars(args))

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK") or 0)
    torch.cuda.set_device(local_rank)
    if local_rank == 0:
        print(vars(args))

    dist.init_process_group(backend="nccl", world_size=world_size, rank=local_rank)

    device_map = {"": local_rank}
    device = torch.device("cuda",local_rank)

    tokenizer = Qwen2Tokenizer.from_pretrained(args.ckpt_path)
    # tokenizer = Qwen2Tokenizer.from_pretrained("/storage/xylin/LLM4Rec/reasoning/ckpt/debug")
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = tokenizer.eos_token_id


    if not args.lora:
        args.base_model = args.ckpt_path

    config = Qwen2Config.from_pretrained(args.ckpt_path)
    if args.autoregression_steps==0:
        config.autoregression_steps = args.autoregression_steps
    

    if args.adaptive_attn:
        config.adaptive_attn = True
        config.attention_strategy = args.attention_strategy
        config.test = True
    # update model attention mask function if needed

    model = Qwen2WithReasoning_ParallelQuery_Custom.from_pretrained(  # Qwen2ForCausalLM
        args.ckpt_path,
        config=config,
        # autoregression_steps=args.autoregression_step,
        # load_in_8bit=load_8bit,
        torch_dtype=torch.bfloat16,
        # attn_implementation='flash_attention_2',
        low_cpu_mem_usage=True,
        device_map=device_map,
    )
    
    # model.autoregression_steps = config.autoregression_steps
    model.identifier_len = 4
    # model.steps_per_code = model.autoregression_steps + 1
    
    # print("*** model reasoning step:", model.autoregression_steps)
    
    model.resize_token_embeddings(len(tokenizer))

    model.config.pad_token_id = tokenizer.pad_token_id

    if args.lora:
        model = PeftModel.from_pretrained(
            model,
            args.ckpt_path,
            torch_dtype=torch.bfloat16,
            # torch_dtype=torch.float16,
            device_map=device_map,
        )

    model = DistributedDataParallel(model, device_ids=[local_rank])
    model.eval()

    prompt_ids = [0]

    test_data = load_test_dataset(args)
    if args.subset_test:
        args.sample_num = 2000
        test_data = load_test_dataset(args)
    ddp_sampler = DistributedSampler(test_data, num_replicas=world_size, rank=local_rank, shuffle=False, drop_last=False)

    collator = TestCollator(args, tokenizer)
    all_items = test_data.get_all_items()

    if args.greedy_trie:
        prefix_allowed_tokens = get_greedy_prefix_allowed_tokens_fn(test_data.indices,tokenizer)
    else:
        candidate_trie = Trie(
            [
                [1] + 
                tokenizer.encode(candidate)
                + [tokenizer.eos_token_id]
                for candidate in all_items
            ]
        )
        prefix_allowed_tokens = prefix_allowed_tokens_fn(candidate_trie, tokenizer, args.special_token_for_answer)

    test_loader = DataLoader(test_data, batch_size=args.test_batch_size, collate_fn=collator,
                             sampler=ddp_sampler, shuffle=False, num_workers=1, pin_memory=True)
    if local_rank == 0:
        print("data num:", len(test_data))
    # all performance
    with torch.no_grad(): 
        for prompt_id in prompt_ids:
            total = 0
            all_pred_list = []
            all_gold_list = []
            st_all = time.time()
            for step, batch in enumerate(tqdm(test_loader)):
                inputs = batch[0].to(device)
                targets = batch[1]
                total += len(targets)
                output = model.module.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=args.max_new_token, #10
                    prefix_allowed_tokens_fn=prefix_allowed_tokens,
                    num_beams=args.num_beams,
                    num_return_sequences=args.num_beams,
                    output_scores=True,
                    return_dict_in_generate=True,
                    early_stopping=True,
                    do_sample=False
                )
                
                output_ids = output["sequences"]
                scores = output["sequences_scores"]
                output = tokenizer.batch_decode(
                    output_ids, skip_special_tokens=True
                )

                topk_res = get_topk_results(output,scores,targets,args.num_beams,
                                            all_items=all_items if args.filter_items else None)

                res_gather_list = [None for _ in range(world_size)]
                dist.all_gather_object(obj=topk_res, object_list=res_gather_list)
                target_gather_list = [None for _ in range(world_size)]
                dist.all_gather_object(obj=targets, object_list=target_gather_list)

                if local_rank == 0:
                    for ga_res in res_gather_list:
                        all_pred_list += ga_res
                    for ga_tar in target_gather_list:
                        all_gold_list += ga_tar

            if local_rank == 0:
                print("=== End ===")
                test_results = computeTopNAccuracy(all_gold_list, all_pred_list, topN=[5, 10, 20],rank=local_rank)
                print_results(None, None, test_results)
            dist.barrier()
        dist.barrier()
            # print results
        if local_rank == 0:
            test_results = computeTopNAccuracy(all_gold_list, all_pred_list, topN=[5, 10, 20])
            print("=== End ===")
            print("=== All performance")
            print_results(None, None, test_results)
            print(f"All time costs: {round(time.time()-st_all, 2)}s")

            # ---- 新增 diversity 和 over-recommendation 指标 ----
            div_or_results = compute_div_or_ratio(all_pred_list, Ks=[5,10,20],top=10)
            print_div_or_results(div_or_results)
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLMRec_test")
    parser = parse_global_args(parser)
    parser = parse_dataset_args(parser)
    parser = parse_train_args(parser)
    parser = parse_test_args(parser)
    # parser = parse_llama_args(parser)

    args = parser.parse_args()
    print(args.ckpt_path)
    test(args)
