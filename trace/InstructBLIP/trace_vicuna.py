load_path = '/workspace/vicuna-7b-1.1'

llm_tokenizer = LlamaTokenizer.from_pretrained(load_path, use_fast=False, truncation_side="left")

llm_model = LlamaForCausalLM.from_pretrained(
    load_path, torch_dtype=torch.float32
)