from transformers import Qwen2ForCausalLM, BeamSearchScorer
import torch
import torch.nn.functional as F
from transformers.models.qwen2.modeling_qwen2 import *
from transformers.models.qwen2.modeling_qwen2 import _CONFIG_FOR_DOC
from transformers.generation.utils import *
from transformers.generation.utils import _split_model_inputs
import ipdb

@dataclass
class CausalLMOutputWithPastReason(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    context_input_embeds: torch.FloatTensor = None
    context_attention: torch.LongTensor = None
    prefix_ids: torch.LongTensor = None

@dataclass
class CausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    cache_position: Optional[torch.LongTensor] = None
    position_ids: Optional[torch.LongTensor] = None
    attention_mask: Optional[torch.LongTensor] = None
    inputs_embeds: Optional[torch.FloatTensor] = None


@dataclass
class CausalLMOutputAnalysis(ModelOutput):
    logits: torch.FloatTensor = None
    target: torch.Tensor=None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None

   
class Qwen2Model_AdaptiveAttn_Custom(Qwen2PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Qwen2DecoderLayer`]

    Args:
        config: Qwen2Config

    Notes: this class is compatible with the 
    1) customized reasoning steps (reasoning steps for each code can be different)
    2) customized adaptive attention (whether each code attend adaptive information)
    """
    

    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2RotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(QWEN2_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_mask_dict: dict = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, attention_mask_dict, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **flash_attn_kwargs,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        return output if return_dict else output.to_tuple()

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        attention_mask_dict: dict, 
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):  

        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)
        using_sliding_window_cache = isinstance(past_key_values, SlidingWindowCache)


        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        # SlidingWindowCache or StaticCache
        if using_sliding_window_cache or using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        # DynamicCache or no cache
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            attention_mask_dict,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
            config=self.config,
            past_key_values=past_key_values,
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu"]
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    def update_attention_mask_stragety(self, mode, test=False):
        if mode == "hard":
            self._build_stage_attention_mask_across_items_fast = self._build_stage_attention_mask_across_items_fast_V2
        else:
            raise NotImplementedError

        self.test = test
      
    def _build_stage_attention_mask_across_items_fast_V2(
        self,
        batch_size: int,
        input_len: int,
        len_identifier: int,
        generation_code_idx_start_from_1: int,
        query_list: list,
        progressive_list: list,
        dtype: torch.dtype,
        device: torch.device,
    ):
        """
        Build a stage-restricted attention mask. "hard"

        For each stage k:
        - Query vectors: can see selected item tokens (<=k-th code per item),
            previous reasoning tokens (all previous stages), and the special token.
        - Gold code: standard causal mask (can see everything before it).
        """

        if not self.test: # train
            stage_lens = [q + 1 for q in query_list]
            total_reason_tokens = sum(stage_lens)
            total_len = input_len + total_reason_tokens

            min_val = torch.finfo(dtype).min
            mask = torch.full((batch_size, 1, total_len, total_len),
                fill_value=min_val, dtype=dtype, device=device)

            # (1) Historical item region: normal causal mask
            base_causal = torch.tril(torch.ones(input_len, input_len, device=device, dtype=torch.bool))
            mask[:, :, :input_len, :input_len] = torch.where(
                base_causal,
                torch.zeros_like(mask[:, :, :input_len, :input_len]),
                mask[:, :, :input_len, :input_len],
            )
            # (2) Item info (split items + special token)
            num_items = input_len // len_identifier
            special_token_idx = input_len - 1
            code_pos = torch.arange(len_identifier, device=device).repeat(num_items)
            
            visible_mask = code_pos.unsqueeze(0) < torch.arange(
                1, generation_code_idx_start_from_1 + 1, device=device
            ).unsqueeze(1)
            
            reasoning_start = input_len  # where reasoning tokens start
            cur_ptr = input_len

            # (3) Construct stage-specific visibility
            for k in range(generation_code_idx_start_from_1):
                n_query = query_list[k] # n_query for current stage
                start = cur_ptr
                end = start + n_query + 1  # +1 for gold token

                # ---- (a) Queries: selective or full visibility ----
                q_start = start
                q_end = start + n_query

                if progressive_list[k]:
                    # selective visibility
                    visible_cols = visible_mask[k].nonzero(as_tuple=True)[0].tolist()
                    visible_cols.append(special_token_idx)
                    visible_cols = torch.tensor(visible_cols, device=device, dtype=torch.long)
                else:
                    # full visibility over all item tokens
                    visible_cols = torch.arange(input_len, device=device, dtype=torch.long)

                # set attention mask to 0 for visible tokens in historical items
                mask[:, :, q_start:q_end, visible_cols] = 0

                # queries attend all previous reasoning tokens
                if q_start > reasoning_start:
                    mask[:, :, q_start:q_end, reasoning_start:q_start] = 0

                # query block is causal mask
                local_q_causal = torch.tril(torch.ones(n_query, n_query, device=device, dtype=torch.bool))
                mask[:, :, q_start:q_end, q_start:q_end] = torch.where(
                    local_q_causal,
                    torch.zeros_like(mask[:, :, q_start:q_end, q_start:q_end]),
                    mask[:, :, q_start:q_end, q_start:q_end],
                )

                # ---- (b) Gold code: standard causal ----
                gold_pos = end - 1  # last token in this stage
                mask[:, :, gold_pos, :gold_pos] = 0  # can attend to everything before

                # move to next stage
                cur_ptr = end
                
        else: # test
            stage_idx = generation_code_idx_start_from_1 - 1
            total_len = input_len + sum(q + 1 for q in query_list[:stage_idx]) + query_list[stage_idx] # no gold token
            
            min_val = torch.finfo(dtype).min
            mask = torch.full((batch_size, 1, total_len, total_len),
                            fill_value=min_val, dtype=dtype, device=device)

            # (1) Historical item region: normal causal mask
            base_causal = torch.tril(torch.ones(input_len, input_len, device=device, dtype=torch.bool))
            mask[:, :, :input_len, :input_len] = torch.where(
                base_causal,
                torch.zeros_like(mask[:, :, :input_len, :input_len]),
                mask[:, :, :input_len, :input_len],
            )

            # (2) Item info (split items + special token)
            num_items = input_len // len_identifier
            special_token_idx = input_len - 1
            code_pos = torch.arange(len_identifier, device=device).repeat(num_items)
            visible_mask = code_pos.unsqueeze(0) < torch.arange(
                1, generation_code_idx_start_from_1 + 1, device=device
            ).unsqueeze(1)

            reasoning_start = input_len  # where reasoning tokens start

            # (3) Construct stage-specific visibility
            for k in range(generation_code_idx_start_from_1):
                start = input_len + sum(q + 1 for q in query_list[:k])
                n_query = query_list[k] 
                end = start + (n_query + 1) 

                # ---- (a) Queries: selective visibility ----
                q_start = start
                q_end = start + n_query

                if progressive_list[k]:
                    # selective visibility: only attend partial items + special token
                    visible_cols = visible_mask[k].nonzero(as_tuple=True)[0].tolist()
                    visible_cols.append(special_token_idx)
                    visible_cols = torch.tensor(visible_cols, device=device, dtype=torch.long)
                else:
                    # full visibility: attend all item tokens
                    visible_cols = torch.arange(input_len, device=device, dtype=torch.long)

                # set attention mask to 0 for selective tokens in hitorical items
                mask[:, :, q_start:q_end, visible_cols] = 0

                # queries attend all reasoning tokens + gold tokens
                if q_start > reasoning_start:
                    mask[:, :, q_start:q_end, reasoning_start:q_start] = 0

                # query block is causal mask 
                local_q_causal = torch.tril(torch.ones(n_query, n_query, device=device, dtype=torch.bool))
                mask[:, :, q_start:q_end, q_start:q_end] = torch.where(
                    local_q_causal,
                    torch.zeros_like(mask[:, :, q_start:q_end, q_start:q_end]),
                    mask[:, :, q_start:q_end, q_start:q_end],
                )

                if k < (generation_code_idx_start_from_1 - 1): # not the last code
                    # ---- (b) Gold code: standard causal ----
                    gold_pos = end - 1  # last token in this stage
                    mask[:, :, gold_pos, :gold_pos] = 0  # can attend to everything before  
                
        return mask     
       

    
    def _prepare_4d_causal_attention_mask_with_cache_position(
        self,
        attention_mask: torch.Tensor,
        attention_mask_dict: torch.Tensor, 
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        config: Qwen2Config,
        past_key_values: Cache,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache, to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            device (`torch.device`):
                The device to plcae the 4D attention mask on.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
            config (`Qwen2Config`):
                The model's configuration class
            past_key_values (`Cache`):
                The cache class that is being used currently to generate
        """

        # ---- our custom adpative attention ----
        if isinstance(attention_mask_dict, dict) and "input_len" in attention_mask_dict:
            spec = attention_mask_dict
            causal_mask = self._build_stage_attention_mask_across_items_fast(
                batch_size=batch_size,
                input_len=spec["input_len"],
                len_identifier=spec["len_identifier"],
                generation_code_idx_start_from_1=spec["generation_code_idx_start_from_1"],
                # n_query=spec["n_query"],
                query_list=spec["query_list"],
                progressive_list=spec["progressive_list"],
                dtype=dtype,
                device=device,
            )
            
        # ---- After causal_mask is built, add the original mask (pad)----
        if isinstance(attention_mask, torch.Tensor) and attention_mask.dim() == 2:
            # attention_mask: [bs, total_len]  (1 for valid token, 0 for pad)
            valid = attention_mask.to(dtype=torch.bool)  # [bs, total_len]
            # Mask out invalid keys (columns)
            causal_mask = causal_mask.masked_fill(~valid[:, None, None, :], torch.finfo(causal_mask.dtype).min)
            # # Mask out invalid queries (rows)
            # causal_mask = causal_mask.masked_fill(~valid[:, None, :, None], torch.finfo(causal_mask.dtype).min)

        if self.test and spec["generation_code_idx_start_from_1"]>1:
            causal_mask = causal_mask[:,:,-(spec["query_list"][spec["generation_code_idx_start_from_1"]-1]+1):, :]
            

        return causal_mask


class CARE(Qwen2ForCausalLM):
    '''
        model use multiple quries to do parallel reasoning to generate each code. the key difference lies in forward and the attention mask
    '''
    def __init__(self, config, query_list=[1,1,1,1], identifier_len=4, query_div_scale=0, progressive_attn=None, progressive_list=[True, True, True, True], attention_strategy=None, use_cache=True):
        super().__init__(config)

        if not hasattr(config, "progressive_attn"): # training
            if progressive_attn: 
                self.model = Qwen2Model_AdaptiveAttn_Custom(config)
                self.model.update_attention_mask_stragety(attention_strategy)
            else:
                pass
        else: # inference
            if config.progressive_attn:
                self.model = Qwen2Model_AdaptiveAttn_Custom(config) 
                self.model.update_attention_mask_stragety(config.attention_strategy, config.test)
            else:
                pass

        self.identifier_len = identifier_len
        self.train_use_cache = use_cache

        # self.n_query = reasoning_steps * (self.identifier_len+1) # EOS
        if hasattr(config, "query_list"): # test
            self.n_query = sum(config.query_list)
            self.query_list = config.query_list
            self.progressive_list = config.progressive_list
        else: # train
            self.n_query = sum(query_list)
            self.query_list = query_list
            self.progressive_list = progressive_list
        
        if self.n_query:
            self.query_vector = nn.Embedding(self.n_query, self.model.config.hidden_size)
        if self.n_query == 0:
            zeros = torch.zeros(1, self.model.config.hidden_size)
            self.query_vector = nn.Embedding.from_pretrained(zeros, freeze=True)
        
        self.query_div_scale = query_div_scale

    def fixed_cross_entropy(self, source, target, num_items_in_batch: int = None, ignore_index: int = -100, **kwargs):
        # Upcast to float if we need to compute the loss to avoid potential precision issues
        source = source.float()
        target = target.to(source.device)
        
        reduction = "sum" if num_items_in_batch is not None else "mean"
        loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
        if reduction == "sum":
            loss = loss / num_items_in_batch
        return loss

    def update_query_embedding(self):
        raise NotImplementedError

    def update_config(self, query_list, progressive_list):
        self.n_query = sum(query_list)
        self.progressive_list = progressive_list
        self.config.progressive_list = progressive_list
        self.config.query_list = query_list

    def forward_training(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # 1. convert input_ids into input_embeds
        inputs_embeds = self.model.get_input_embeddings()(input_ids)  # (bs, seq_len, dim)
        loss = 0
        
        for code_idx in range(self.identifier_len):
            reasoning_step = self.query_list[code_idx]
            # 2. concat the learnable querys 
            lookup_idx = torch.arange(sum(self.query_list[:code_idx]), sum(self.query_list[:code_idx]) + reasoning_step) # index only from query_vectors (exclude code position)
            lookup_idx = lookup_idx.to(inputs_embeds.device).unsqueeze(0).repeat(inputs_embeds.shape[0], 1) # (bs, n_reason)
            query_vector = self.query_vector(lookup_idx) # (bs, n_reason, dim)
            inputs_embeds = torch.cat([inputs_embeds, query_vector], dim=1) # (bs, seq_len+n_reason, dim)

            # update (1) sequence input (2) attention mask - concat with the gold code
            code_idx_tensor = torch.LongTensor([code_idx]).to(labels.device)
            gold_code_idx = labels[:, code_idx_tensor] # (bs, 1)
            gold_code_emb = self.model.get_input_embeddings()(gold_code_idx) # (bs, dim)
            inputs_embeds = torch.cat([inputs_embeds, gold_code_emb], dim=1)

            # 3. update attention mask (add # reasoning steps)
            attention_mask = torch.cat([attention_mask, attention_mask[:,-1:].repeat(1, reasoning_step+1)], dim=1) # (bs, seq_len+n_reason+gold code)
        
        attention_mask_dict = {
            "input_len": input_ids.size(1),
            "len_identifier": self.identifier_len,
            "generation_code_idx_start_from_1": self.identifier_len,
            "query_list": self.query_list,
            "progressive_list": self.progressive_list
        }

        forward_inputs_embeds = inputs_embeds
        
        # 1. forward once
        outputs = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            attention_mask_dict=attention_mask_dict,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=forward_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]  # (bs, total_seq_len, hidden_dim)
        logits = self.lm_head(hidden_states)  # (bs, total_seq_len, vocab_size)

        bs = input_ids.size(0)
        input_len = input_ids.size(1)

        # 2. get predictions
        pred_positions = []
        for code_idx in range(self.identifier_len):
            pred_pos = input_len + sum(self.query_list[:code_idx]) + code_idx + self.query_list[code_idx] - 1
            pred_positions.append(pred_pos)
        pred_positions = torch.tensor(pred_positions, device=logits.device)  # (identifier_len,)

        selected_logits = logits[:, pred_positions, :]  # (bs, identifier_len, vocab)

        # 3. get targets (only identifier, not include EOS)
        targets = labels[:,:-1]  # (bs, identifier_len)

        # 4. compute loss
        logits_flat = selected_logits.view(-1, self.config.vocab_size)
        targets_flat = targets.reshape(-1)

        loss = self.fixed_cross_entropy(
            source=logits_flat,
            target=targets_flat,
            num_items_in_batch=len(targets_flat)
        )

        # query diversity loss
        qv = self.query_vector.weight  # (n_query, dim)
        qv = F.normalize(qv, dim=1)     # cosine normalization

        sim_matrix = torch.matmul(qv, qv.T)  # (n_query, n_query), cosine similarity
        mask = torch.eye(sim_matrix.size(0), device=qv.device).bool()
        sim_matrix = sim_matrix.masked_fill(mask, 0.0)  # remove diagonal

        div_loss = sim_matrix.mean()
        loss += self.query_div_scale * div_loss

        if not return_dict:
            output = (logits_flat,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=selected_logits,  # 只返回参与loss的logits
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


    def forward_inference(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        code_idx: Optional[int] = None, 
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        

        # if EOS
        code_idx = min(code_idx, self.identifier_len-1)

        # 1. convert input_ids into input_embeds
        if input_ids is not None:
            assert code_idx == 0
            
        n_query = self.query_list[code_idx]
 
        if code_idx == 0:
            inputs_embeds = self.model.get_input_embeddings()(input_ids)  # (bs, seq_len, dim)
            cache_position = torch.cat([cache_position, torch.arange(cache_position.shape[0], cache_position.shape[0]+n_query).to(cache_position.device)], dim=0) 
            add_tensor = torch.arange(1, n_query+1).to(position_ids.device)
            position_ids = torch.cat([position_ids, add_tensor+position_ids[:,-1:]], dim=1)
        else:
            position_ids = position_ids.repeat(1, n_query+1)
            for idx in range(n_query+1):
                position_ids[:,idx] = position_ids[:,idx] + (idx+1)

            # position_ids = torch.cat([position_ids, temp])  
            past = past_key_values.get_seq_length()
            cache_position = torch.arange(past, past + 1+n_query, device=inputs_embeds.device)

        # 2. concat the learnable querys
        # lookup_idx = torch.arange(code_idx * reasoning_steps, (code_idx+1)*reasoning_steps) 
        lookup_idx = torch.arange(sum(self.query_list[:code_idx]), sum(self.query_list[:code_idx])+self.query_list[code_idx])
        lookup_idx = lookup_idx.to(inputs_embeds.device).unsqueeze(0).repeat(inputs_embeds.shape[0], 1) # (bs, n_reason)

        query_vector = self.query_vector(lookup_idx) # (bs, n_reason, dim)
        
        inputs_embeds = torch.cat([inputs_embeds, query_vector], dim=1) # (bs, seq_len+n_reason, dim)

        # input_len = attention_mask.size(1) - code_idx*(reasoning_steps+1)
        input_len = attention_mask.size(1) - (sum(self.query_list[:code_idx])+code_idx)

        # 3. update attention mask (add # reasoning steps) and cache position
        attention_mask = torch.cat([attention_mask, attention_mask[:,-1:].repeat(1, self.query_list[code_idx])], dim=1) # (bs, seq_len+n_reason)

        attention_mask_dict = {
            "input_len": input_len,
            "len_identifier": self.identifier_len,
            "generation_code_idx_start_from_1": code_idx + 1, # 只开放到当前阶段
            "query_list": self.query_list,
            "progressive_list": self.progressive_list
        }

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)

        outputs = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            attention_mask_dict=attention_mask_dict,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        # 4. take last output logits (get the final logits) 
        hidden_states = outputs[0] # (bs, seq_len+n_reason, dim)

        reason_states = hidden_states[:, -1:, :] # (bs, 1, dim)
        logits = self.lm_head(reason_states) # (bs, 1, dim)
        logits = torch.mean(logits, dim=1, keepdim=True) # (bs, dim)

        loss = None        
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cache_position=cache_position,
            position_ids=position_ids, 
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
        )


    def _beam_search(
        self,
        input_ids: torch.LongTensor,
        beam_scorer: BeamScorer,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool,
        **model_kwargs,
    ) -> Union[GenerateBeamOutput, torch.LongTensor]:

        # init values
        pad_token_id = generation_config._pad_token_tensor
        eos_token_id = generation_config._eos_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        sequential = generation_config.low_memory
        do_sample = generation_config.do_sample

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape
        model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
            )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        beam_indices = (
            tuple(() for _ in range(batch_beam_size)) if (return_dict_in_generate and output_scores) else None
        )
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # initialise score of first beam with 0 and the rest with -1e9. This makes sure that only tokens
        # of the first beam are considered to avoid sampling the exact same tokens across all beams.
        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))

        this_peer_finished = False

        decoder_prompt_len = input_ids.shape[-1]  # record the prompt length of decoder
        import ipdb
        code_idx = 0
        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):

            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            
            # prepare variable output controls (note: some models won't accept all output controls)
            model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
            model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

            # if sequential is True, split the input to batches of batch_size and run sequentially
            if sequential:
                if any(
                    model_name in self.__class__.__name__.lower()
                    for model_name in [
                        "fsmt",
                        "reformer",
                        "ctrl",
                        "gpt_bigcode",
                        "transo_xl",
                        "xlnet",
                        "cpm",
                        "jamba",
                    ]
                ):
                    raise RuntimeError(
                        f"Currently generation for {self.__class__.__name__} is not supported "
                        f"for `low_memory beam_search`. Please open an issue on GitHub if you need this feature."
                    )

                inputs_per_sub_batches = _split_model_inputs(
                    model_inputs,
                    split_size=batch_size,
                    full_batch_size=batch_beam_size,
                    config=self.config.get_text_config(),
                )
                outputs_per_sub_batch = [
                    self.forward_inference(**inputs_per_sub_batch, return_dict=True, code_idx=code_idx) for inputs_per_sub_batch in inputs_per_sub_batches
                ]

                outputs = stack_model_outputs(outputs_per_sub_batch, self.config.get_text_config())

            else:  # Unchanged original behavior
                outputs = self.forward_inference(**model_inputs, return_dict=True, code_idx=code_idx)

            code_idx += 1 # update code idx
            
            # update cache position and position ids
            model_kwargs["cache_position"] = outputs.cache_position
            model_kwargs["position_ids"] = outputs.position_ids
            model_kwargs["attention_mask"] = outputs.attention_mask
            model_kwargs['inputs_embeds'] = outputs.inputs_embeds

            # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
                num_new_tokens=self.query_list[min(code_idx, self.identifier_len-1)]+1,
            )
            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue
            
            # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
            # (the clone itself is always small)
            # .float() is needed to retain precision for later logits manipulations
            next_token_logits = outputs.logits[:, -1, :].clone().float()
            next_token_logits = next_token_logits.to(input_ids.device)
            next_token_scores = nn.functional.log_softmax(
                next_token_logits, dim=-1
            )  # (batch_size * num_beams, vocab_size)
            
            next_token_scores_processed = logits_processor(input_ids, next_token_scores)
            next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(
                next_token_scores_processed
            )

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores_processed,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)
                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            # Beam token selection: pick 1 + eos_token_id.shape[0] next tokens for each beam so we have at least 1
            # non eos token per beam.
            n_eos_tokens = eos_token_id.shape[0] if eos_token_id is not None else 0
            n_tokens_to_keep = max(2, 1 + n_eos_tokens) * num_beams
            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=n_tokens_to_keep)
                next_token_scores = torch.gather(next_token_scores, -1, next_tokens)
                next_token_scores, _indices = torch.sort(next_token_scores, descending=True, dim=1)
                next_tokens = torch.gather(next_tokens, -1, _indices)
            else:
                next_token_scores, next_tokens = torch.topk(
                    next_token_scores, n_tokens_to_keep, dim=1, largest=True, sorted=True
                )

            next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                beam_indices=beam_indices,
                decoder_prompt_len=decoder_prompt_len,
            )

            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

            beam_next_tokens_embeds = self.model.get_input_embeddings()(beam_next_tokens.unsqueeze(-1).to(input_ids.device)) 
            model_kwargs["inputs_embeds"] = torch.cat([model_kwargs["inputs_embeds"][beam_idx, ...], beam_next_tokens_embeds], dim=1) # 这个是把full_input_embeds按照beam idx更新交换顺序

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            # IMPORTANT: Note that this should appear BEFORE the call to _reorder_cache() to save the maximum memory
            # (that way the memory peak does not include outputs.logits)
            del outputs

            if model_kwargs.get("past_key_values", None) is not None:
                model_kwargs["past_key_values"] = self._temporary_reorder_cache(
                    model_kwargs["past_key_values"], beam_idx
                )

            if return_dict_in_generate and output_scores:
                beam_indices = tuple((beam_indices[beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices))))

            # increase cur_len
            cur_len = cur_len + 1

            if beam_scorer.is_done or all(stopping_criteria(input_ids, scores)):
                this_peer_finished = True

        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
            beam_indices=beam_indices,
            decoder_prompt_len=decoder_prompt_len,
        )

        if return_dict_in_generate:
            if not output_scores:
                sequence_outputs["sequence_scores"] = None

            if self.config.is_encoder_decoder:
                return GenerateBeamEncoderDecoderOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    logits=raw_logits,
                    beam_indices=sequence_outputs["beam_indices"],
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateBeamDecoderOnlyOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    logits=raw_logits,
                    beam_indices=sequence_outputs["beam_indices"],
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return sequence_outputs["sequences"]


    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        # 1. Handle BC:
        model_inputs = {}
        # - some models don't have `Cache` support (which implies they don't expect `cache_position` in `forward`)
        if self._supports_cache_class:
            model_inputs["cache_position"] = cache_position
        elif cache_position is None:
            raise NotImplementedError
        
        # 2. Generic cache-dependent input preparation
        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        # Exception 3: with synced GPUs cache_position may go out of bounds, but we only want dummy token in that case.
        #              (we can't check exception 3 while compiling)
        # Excpetion 4: If input_embeds are passed then slice it through `cache_position`, to keep only the unprocessed tokens and
        # generate the first token for each sequence. Later use the generated Input ids for continuation.
        if past_key_values is not None:
            
            model_inputs["past_key_values"] = past_key_values
            if inputs_embeds is not None and cache_position is not None: # and input_ids.shape[1] == 0:  # Exception 4
                # inputs_embeds = inputs_embeds[:, -cache_position.shape[0] :]
                inputs_embeds = inputs_embeds[:, -1:] # get the last one 

        # 3. Prepare base model inputs
        input_ids_key = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step for every prompt.
        if not self.config.is_encoder_decoder:
            if inputs_embeds is not None and len(cache_position) == inputs_embeds.shape[1]:
                model_inputs[input_ids_key] = None
                model_inputs["inputs_embeds"] = inputs_embeds
                model_inputs["inputs_ids"] = input_ids
            else:
                assert input_ids is not None, "input_ids must be provided"
                model_inputs["input_ids"] = input_ids
                    
        else:
            raise NotImplementedError

        # 4. Create missing `position_ids` on the fly
        attention_mask = (
            kwargs.pop("decoder_attention_mask", None) if self.config.is_encoder_decoder else attention_mask
        )
        attention_mask_key = "decoder_attention_mask" if self.config.is_encoder_decoder else "attention_mask"
        position_ids_key = "decoder_position_ids" if self.config.is_encoder_decoder else "position_ids"
        if (
            attention_mask is not None
            and kwargs.get(position_ids_key) is None
            and position_ids_key in set(inspect.signature(self.forward).parameters.keys())
        ):
            if attention_mask.shape[1] == input_ids.shape[1]:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                kwargs[position_ids_key] = position_ids  # placed in kwargs for further processing (see below)
            else:
                pass

        # 5. Slice model inputs if it's an input that should have the same length as `input_ids`
        for model_input_name in ["position_ids", "token_type_ids", "decoder_position_ids"]:
            model_input = kwargs.get(model_input_name)
            if model_input is not None:
                if past_key_values is not None:
                    current_input_length = (
                        model_inputs["inputs_embeds"].shape[1]
                        if model_inputs.get("inputs_embeds") is not None
                        else model_inputs[input_ids_key].shape[1]
                    )
                    model_input = model_input[:, -current_input_length:]
                    model_input = model_input.clone(memory_format=torch.contiguous_format)
                model_inputs[model_input_name] = model_input
        # 
        # 6. Create 4D attention mask is we are using a `StaticCache` (important for performant compiled forward pass)
        if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
            if model_inputs["inputs_embeds"] is not None:
                batch_size, sequence_length, _ = model_inputs["inputs_embeds"].shape
                device = model_inputs["inputs_embeds"].device
            else:
                batch_size, sequence_length = model_inputs[input_ids_key].shape
                device = model_inputs[input_ids_key].device

            # Create the causal mask with fixed shape in advance, to reduce recompilations. If the function to create
            # the 4D causal mask exists, it should be present in the base model (XXXModel class).
            base_model = getattr(self, self.base_model_prefix, None)
            if base_model is None:
                causal_mask_creation_function = getattr(
                    self, "_prepare_4d_causal_attention_mask_with_cache_position", None
                )
            else:
                causal_mask_creation_function = getattr(
                    base_model, "_prepare_4d_causal_attention_mask_with_cache_position", None
                )
            if causal_mask_creation_function is None:
                logger.warning_once(
                    f"{self.__class__.__name__} has no `_prepare_4d_causal_attention_mask_with_cache_position` method "
                    "defined in its base modeling class. Compiled forward passes will be sub-optimal. If you're "
                    "writing code, see Llama for an example implementation. If you're a user, please report this "
                    "issue on GitHub."
                )
            else:
                attention_mask = causal_mask_creation_function(
                    attention_mask,
                    sequence_length=sequence_length,
                    target_length=past_key_values.get_max_cache_shape(),
                    dtype=self.dtype,
                    device=device,
                    cache_position=cache_position,
                    batch_size=batch_size,
                    config=self.config,
                    past_key_values=past_key_values,
                )
        if attention_mask is not None:
            model_inputs[attention_mask_key] = attention_mask

        # 7. Forward ALL kwargs that are uninitialized (e.g. `use_cache`).
        for key, value in kwargs.items():
            if key not in model_inputs:
                model_inputs[key] = value

        # 8. Remove unexpected `generate` inputs (TODO @joao: fix trainer and examples)
        model_inputs.pop("labels", None)
        return model_inputs
    
    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ) -> Dict[str, Any]:
        # update past_key_values keeping its naming used in model code
        for possible_cache_name in ALL_CACHE_NAMES:
            if possible_cache_name in outputs:
                # TODO (joao): remove output/input mismatch when these old models (xlnet, reformer) are deprecated
                if possible_cache_name in ("past_buckets_states", "mems"):
                    cache_name = "past_key_values"
                else:
                    cache_name = possible_cache_name
                model_kwargs[cache_name] = getattr(outputs, possible_cache_name)
                break

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

        if not is_encoder_decoder:
            # update attention mask
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                model_kwargs["attention_mask"] = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )
        else:
            # update decoder attention mask
            if "decoder_attention_mask" in model_kwargs:
                decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                model_kwargs["decoder_attention_mask"] = torch.cat(
                    [decoder_attention_mask, decoder_attention_mask.new_ones((decoder_attention_mask.shape[0], 1))],
                    dim=-1,
                )

        if model_kwargs.get("use_cache", True):
            model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + num_new_tokens
        else:
            past_positions = model_kwargs.pop("cache_position")
            new_positions = torch.arange(
                past_positions[-1] + 1, past_positions[-1] + num_new_tokens + 1, dtype=past_positions.dtype
            ).to(past_positions.device)
            model_kwargs["cache_position"] = torch.cat((past_positions, new_positions))
        return model_kwargs