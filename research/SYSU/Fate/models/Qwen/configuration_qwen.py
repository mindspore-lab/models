from mindnlp.transformers.configuration_utils import PretrainedConfig

class Qwen2MoeConfig(PretrainedConfig):

    model_type = "qwen2_moe"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=2048,
        intermediate_size=5632,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_key_value_heads=16,
        hidden_act="silu",
        max_position_embeddings=8192,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        # tie_word_embeddings=False,
        rope_theta=1000000.0,
        use_sliding_window=False,
        sliding_window=32768,
        max_window_layers=21,
        attention_dropout=0.0,
        decoder_sparse_step=1,
        moe_intermediate_size=1408,
        shared_expert_intermediate_size=5632,
        num_experts_per_tok=4,
        num_experts=60,
        norm_topk_prob=False,
        output_router_logits=False,
        router_aux_loss_coef=0.001,
        mlp_only_layers=None,
        eos_token_id=151643,
        bos_token_id=151643,
        pad_token_id=151643,
        # **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window if use_sliding_window else None
        self.max_window_layers = max_window_layers

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout
        self.eos_token_id = eos_token_id
        self.bos_token_id = bos_token_id
        self.pad_token_id = pad_token_id

        # MoE arguments
        self.decoder_sparse_step = decoder_sparse_step
        self.moe_intermediate_size = moe_intermediate_size
        self.shared_expert_intermediate_size = shared_expert_intermediate_size
        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts
        self.norm_topk_prob = norm_topk_prob
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.mlp_only_layers = [] if mlp_only_layers is None else mlp_only_layers

        # super().__init__(
        #     tie_word_embeddings=tie_word_embeddings,
        #     **kwargs,
        # )

def get_Qwen_config(name):
    if "/" in name:
        name = name.split("/")[1]
    name = name.lower()

    if name == "qwen1.5-moe-a2.7b":
        config = Qwen2MoeConfig()
    else:
        raise ValueError(f"Invalid model name: {name}")
    return config

if __name__ == "__main__":
   config = get_Qwen_config("qwen1.5-moe-a2.7b")
   config.offload = True
   print(config.d_ff)