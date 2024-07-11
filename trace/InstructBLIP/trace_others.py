from lavis.models.eva_vit import create_eva_vit_g


class visual_encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual_encoder = create_eva_vit_g(
                                img_size=224, 
                                drop_path_rate=0,
                                use_grad_checkpoint=False,
                                vit_precision="fp32"  # fp32 or fp16 precision
                            )
        self.ln_vision = LayerNorm(visual_encoder.num_features)
    def forward(x):
        return self.ln_vision(self.visual_encoder(x))

def trace_visual_encoder():
    traced = torch.jit.trace(visual_encoder(), torch.randn(1, 3, 224, 224))
    jit.save(traced, "visual_encoder_jit.pt")
    torch.export_onnx(traced, "visual_encoder_onnx.onnx", input_names=["input"], output_names=["output"], opset_version=12)

class Qformer(nn.Module):
    def __init__(self):
        super().__init__()
        num_query_token = 32
        vision_width = 1408 # visual_encoder.num_features
        cross_attention_freq = 2

        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        self.Qformer = BertLMHeadModel.from_pretrained(
            "bert-base-uncased", config=encoder_config
        )
        self.query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        self.query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)

        self.llm_proj = nn.Linear(
            self.Qformer.config.hidden_size, 4096 # self.llm_model.config.hidden_size
        )

    def forward(text_Qformer_input_ids, Qformer_atts, query_tokens, image_embeds, image_atts):
        query_output = self.Qformer.bert(
                    text_Qformer_input_ids, # [1, 8]
                    attention_mask=Qformer_atts, # [1, 40]
                    query_embeds=query_tokens, # [1, 32, 768]
                    encoder_hidden_states=image_embeds, # [1, 257, 1408]
                    encoder_attention_mask=image_atts, # [1, 257]
                    return_dict=False)
        breakpoint()
        inputs_llm = self.llm_proj(query_output.last_hidden_state[:,:query_tokens.size(1),:])
        return inputs_llm

def trace_Qformer(max_txt_len = 128):
    qformer = Qformer()
    traced = torch.jit.trace(qformer, [
        torch.randint(0, 1000, (1, max_txt_len)),
        torch.ones(1, max_txt_len+qformer.query_tokens.data.shape[1], dtype=torch.long),
        torch.randn(qformer.query_tokens.data.shape),
        torch.randn(1, 257, 1408),
        torch.ones(1, 257, dtype=torch.long),
    ])
    jit.save(traced, "Qformer_jit.pt")
    torch.export_onnx(traced, "Qformer_onnx.onnx", input_names=["input"], output_names=["output"], opset_version=12)
