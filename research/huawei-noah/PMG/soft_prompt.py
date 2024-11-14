import mindspore as ms

class PrefixEncoder(ms.nn.Cell):
    def __init__(self, num_hidden_layers, hidden_size, pre_seq_len, prefix_projection=False, prefix_hidden_size=4096):
        super().__init__()
        self.prefix_projection = prefix_projection
        if self.prefix_projection:
            # Use a two-layer MLP to encode the prefix
            self.embedding = ms.nn.Embedding(pre_seq_len, hidden_size)
            self.trans = ms.nn.Sequential(
                ms.nn.Linear(hidden_size, prefix_hidden_size),
                ms.nn.Tanh(),
                ms.nn.Linear(prefix_hidden_size, num_hidden_layers * 2 * hidden_size)
            )
        else:
            self.embedding = ms.nn.Embedding(pre_seq_len, num_hidden_layers * 2 * hidden_size)

    def forward(self, prefix):
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.trans(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)
        return past_key_values
    
class InferenceModel(ms.nn.Cell):
    def __init__(self, layer_num, num_image_prompt, num_prefix_prompt, emb_dim=4096, sd_hidden_state_dim=768):
        super().__init__()
        self.layer_num = layer_num
        self.num_image_prompt = num_image_prompt
        self.num_prefix_prompt = num_prefix_prompt
        self.emb_dim = emb_dim
        self.mapping_layer = ms.nn.Linear(emb_dim, sd_hidden_state_dim)
        self.trainable_prompt = ms.nn.Parameter(ms.randn((1, num_image_prompt, emb_dim), requires_grad=True))
        
        self.prefix_tokens = ms.arange(num_prefix_prompt).long()
        self.prefix_encoder = PrefixEncoder(layer_num, 4096, num_prefix_prompt)
    
    def forward(self, llama_tokenizer, llama_model, token, token_len):
        bsz = token.shape[0]
        attention_mask = token!=llama_tokenizer.pad_token_id
        emb = llama_model.model.embed_tokens(token)
        for i in range(bsz):
            l = token_len[i].item()
            emb[i, l:l+self.num_image_prompt] = self.trainable_prompt
            attention_mask[i, l:l+self.num_image_prompt] = 1
        attention_mask = ms.ops.Concat(1)([ms.ones((bsz, self.num_prefix_prompt), device=attention_mask.device), attention_mask])
        
        num_head = llama_model.model.layers[0].self_attn.num_heads
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(bsz, -1).to(token.device)
        past_key_values = self.prefix_encoder(prefix_tokens) # [bsz, num_prefix_prompt, layer_num * 2 * hidden_size]
        past_key_values = past_key_values.view(bsz, self.num_prefix_prompt, self.layer_num, 2, num_head, -1) # [bsz, num_prefix_prompt, layer_num, 2, num_head, head_dim]
        past_key_values = past_key_values.permute(2, 3, 0, 4, 1, 5) # [layer_num, 2, bsz, num_head, num_prefix_prompt, head_dim]
        
        outputs = llama_model.model.forward(inputs_embeds=emb,
                                            output_hidden_states=True,
                                            attention_mask=attention_mask,
                                            past_key_values=past_key_values,
                                            )
        encoder_hidden_states = []
        for i in range(bsz):
            l = token_len[i].item()
            encoder_hidden_states.append(outputs.last_hidden_state[i, l:l+self.num_image_prompt])
        encoder_hidden_states = self.mapping_layer(ms.ops.Stack()(encoder_hidden_states))
        
        return encoder_hidden_states
    
# model = InferenceModel(layer_num=len(llama_model.model.layers), num_image_prompt=args.num_image_prompt, num_prefix_prompt=args.num_prefix_prompt, emb_dim=4096, sd_hidden_state_dim=768)
# model = accelerator.prepare(model)