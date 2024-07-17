"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""


"""
Requires Transformer 4.28 and above, implementation may change according the Llama implementation
"""
import logging
import string
from packaging import version

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
import numpy as np
import transformers

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train
from lavis.models.blip2_models.Qformer import BertConfig

from npuengine import EngineOV
from transformers import LlamaTokenizer

import sys
module_path = "/workspace/InstructBLIP-TPU/lavis/models/blip2_models/instructblip_cpp"
if module_path not in sys.path:
    sys.path.append(module_path)

@registry.register_model("blip2_vicuna_instruct_tpu")
class Blip2VicunaInstruct_TPU(Blip2Base):
    """
    BLIP2 Vicuna model.
    Supported model types:
        - vicuna7b
        - vicuna13b
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2_vicuna_instruct", "vicuna7b")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "vicuna7b": "configs/models/blip2/blip2_instruct_vicuna7b.yaml",
        "vicuna13b": "configs/models/blip2/blip2_instruct_vicuna13b.yaml",
    }

    def __init__(
        self,
        vit_model="../../trace/InstructBLIP/bmodel/visual_encoder_F32.bmodel",
        qformer_model="../../trace/InstructBLIP/bmodel/Qformer_F16.bmodel",
        img_size=224,
        drop_path_rate=0,
        llm_model='/workspace/vicuna-7b-1.1',
        gpt_bmodel_path='../../trace/InstructBLIP/bmodel/llama2-7b_int4_1dev_512.bmodel',
        num_query_token=32,
        prompt="",
        max_txt_len=128,
        max_output_txt_len=256,
        apply_lemmatizer=False,
        qformer_text_input=True,
        device_id=0
    ):
        super().__init__()
        
        self.tokenizer = self.init_tokenizer(truncation_side="left")
        self.visual_encoder_with_ln_vision = EngineOV(model_path=vit_model, batch=1, device_id=device_id)

        self.QformerBert_with_llm_proj = EngineOV(model_path=qformer_model, batch=1, device_id=device_id)

        # init query tokens
        num_query_token = 32
        vision_width = 1408 # visual_encoder.num_features
        cross_attention_freq = 2

        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token

        self.query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        self.query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)

        self.llm_tokenizer = LlamaTokenizer.from_pretrained(llm_model, use_fast=False, truncation_side="left")
        self.llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.llm_tokenizer.add_special_tokens({'bos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'eos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'unk_token': '</s>'})
        # self.llm_tokenizer.pad_token = self.llm_tokenizer.unk_token
        self.eos_token_id = self.llm_tokenizer(
                    self.llm_tokenizer.eos_token, add_special_tokens=False
                ).input_ids[0]
        import llama
        self.llm_model = llama.llama()
        self.llm_model.init([device_id], gpt_bmodel_path)
        self.llm_model.max_new_tokens = 256

        self.max_txt_len = max_txt_len
        self.max_output_txt_len = max_output_txt_len
        self.prompt = prompt
        prompt_tokens = self.llm_tokenizer(self.prompt, return_tensors="pt")
        self.prompt_length = prompt_tokens.attention_mask.sum(1)

        self._lemmatizer = None

        self.qformer_text_input = qformer_text_input

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=256,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1,
        num_captions=1,
        temperature=1,
    ):
        self.llm_tokenizer.padding_side = "left"

        if "prompt" in samples.keys():
            prompt = samples["prompt"]
        else:
            prompt = self.prompt

        image = samples["image"]

        bs = image.size(0)

        if isinstance(prompt, str):
            prompt = [prompt] * bs
        else:
            assert len(prompt) == bs, "The number of prompts must be equal to the batch size."

        # For TextCaps
        if "ocr_tokens" in samples.keys() and "{}" in prompt[0]:
            prompt = [p.format(', '.join(samples['ocr_tokens'][i][:30])) for i, p in enumerate(prompt)]

        query_tokens = self.query_tokens.expand(bs, -1, -1)
        if self.qformer_text_input:
            # remove ocr tokens in q_former (for eval textvqa)
            # qformer_prompt = prompt
            # qformer_prompt = ['Question: ' + qp.split(' Question: ')[1] for qp in qformer_prompt]

            text_Qformer = self.tokenizer(
                prompt,
                padding='longest',
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
            Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask], dim=1)

        # For video data
        if image.dim() == 5:
            inputs_llm, atts_llm = [], []
            for j in range(image.size(2)):
                this_frame = image[:,:,j,:,:]
                with self.maybe_autocast():
                    breakpoint()
                    frame_embeds = torch.from_numpy(self.visual_encoder_with_ln_vision([this_frame.numpy()])[0])
                    # frame_embeds = self.ln_vision(self.visual_encoder(this_frame))
                frame_atts = torch.ones(frame_embeds.size()[:-1], dtype=torch.long).to(image.device)
                if self.qformer_text_input:
                    breakpoint()
                    frame_inputs_llm = self.QformerBert_with_llm_proj([
                        text_Qformer.input_ids.numpy(), # [1, 8]
                        Qformer_atts.numpy(), # [1, 40]
                        query_tokens.numpy(), # [1, 32, 768]
                        image_embeds.numpy(), # [1, 257, 1408]
                        image_atts.numpy()],
                        )[0]
                    frame_inputs_llm = torch.from_numpy(frame_inputs_llm)
                else:
                    breakpoint()
                # if self.qformer_text_input:
                #     breakpoint()
                #     frame_query_output = self.Qformer.bert(
                #         text_Qformer.input_ids,
                #         attention_mask=Qformer_atts,
                #         query_embeds=query_tokens,
                #         encoder_hidden_states=frame_embeds,
                #         encoder_attention_mask=frame_atts,
                #         return_dict=True,
                #     )
                # else:
                #     breakpoint()
                #     frame_query_output = self.Qformer.bert(
                #         query_embeds=query_tokens,
                #         encoder_hidden_states=frame_embeds,
                #         encoder_attention_mask=frame_atts,
                #         return_dict=True,
                #     )
                # frame_inputs_llm = self.llm_proj(frame_query_output.last_hidden_state[:,:query_tokens.size(1),:])
                frame_atts_llm = torch.ones(frame_inputs_llm.size()[:-1], dtype=torch.long).to(image.device)
                inputs_llm.append(frame_inputs_llm)
                atts_llm.append(frame_atts_llm)
            inputs_llm = torch.cat(inputs_llm, dim=1)
            atts_llm = torch.cat(atts_llm, dim=1)
        else:
            with self.maybe_autocast():
                breakpoint()
                # image_embeds = self.ln_vision(self.visual_encoder(image)) # [1, 3, 224, 224]
                image_embeds = torch.from_numpy(self.visual_encoder_with_ln_vision([image.numpy()])[0])
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

            if self.qformer_text_input:
                breakpoint()
                # todo: padding to [1,128] remove magic value 
                if text_Qformer.input_ids.shape[1] > self.max_txt_len:
                    text_Qformer.input_ids = text_Qformer.input_ids[:, :self.max_txt_len]
                else:
                    text_Qformer.input_ids = torch.cat([text_Qformer.input_ids, torch.zeros([1, self.max_txt_len - text_Qformer.input_ids.shape[1]], dtype=torch.int32)], dim=1)
                if Qformer_atts.shape[1] > self.max_txt_len+32:
                    Qformer_atts = Qformer_atts[:, :self.max_txt_len+32]
                else:
                    Qformer_atts = torch.cat([Qformer_atts, torch.zeros([1, self.max_txt_len+32 - Qformer_atts.shape[1]], dtype=torch.int32)], dim=1)
                
                inputs_llm = self.QformerBert_with_llm_proj([
                    text_Qformer.input_ids.numpy().astype(np.int32), # [1, 8]
                    Qformer_atts.numpy().astype(np.int32), # [1, 40]
                    query_tokens.numpy().astype(np.float32), # [1, 32, 768]
                    image_embeds.numpy().astype(np.float32), # [1, 257, 1408]
                    image_atts.numpy().astype(np.int32)], # [ 1 257 ]
                    )[0]
                inputs_llm = torch.from_numpy(inputs_llm)
                breakpoint()
            else:
                breakpoint()
            #     # 没有文本prompt 可以直接传入固定的query_embeds，不需要像if里面那样计算prompt text的embedding，与固定的query_tokens拼接作为实际的embedding）
            #     breakpoint()
            #     query_output = self.Qformer.bert(
            #         query_embeds=query_tokens,
            #         encoder_hidden_states=image_embeds,
            #         encoder_attention_mask=image_atts,
            #         return_dict=True,
            #     )
            # inputs_llm = self.llm_proj(query_output.last_hidden_state[:,:query_tokens.size(1),:])
            # breakpoint()
            
            # atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(image.device)

        llm_tokens = self.llm_tokenizer(
            prompt,
            padding="longest",
            return_tensors="pt"
        ).to(image.device)

        with self.maybe_autocast():
            breakpoint()
            # inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens.input_ids) # Embedding(32001, 4096)
            outputs = self.llm_model.generate(
                llm_tokens.input_ids.flatten().tolist(),
                self.eos_token_id, # EOS_TOKEN_ID
                inputs_llm.numpy().flatten().tolist(), # [1, 32, 4096]
            )
            breakpoint()
            outputs = torch.tensor(outputs, dtype=torch.long)

            # outputs = self.llm_model.generate(
            #     inputs_embeds=inputs_embeds,
            #     attention_mask=attention_mask,
            #     do_sample=False, # use_nucleus_sampling,
            #     # top_p=top_p,
            #     # temperature=temperature,
            #     num_beams=1, #num_beams,
            #     max_length=max_length,
            #     min_length=min_length,
            #     # eos_token_id=self.eos_token_id,
            #     # repetition_penalty=repetition_penalty,
            #     # length_penalty=length_penalty,
            #     num_return_sequences=num_captions,
            # )

        outputs[outputs == 0] = 2 # convert output id 0 to 2 (eos_token_id)
        output_text = self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        output_text = [text.strip() for text in output_text]

        return output_text

    def predict_answers(
        self,
        samples,
        num_beams=5,
        inference_method="generate",
        max_len=10,
        min_len=1,
        num_ans_candidates=128,
        answer_list=None,
        prompt="",
        length_penalty=0,
        **kwargs
    ):
        if isinstance(samples["text_input"], str):
            samples["text_input"] = [samples["text_input"]]

        if prompt:
            if prompt.count("{}") == 2:
                if 'ocr_tokens' in samples:
                    text_input = [
                        prompt.format(', '.join(samples['ocr_tokens'][i][:30]), samples["text_input"][i])
                    for i in range(len(samples["text_input"]))]
                elif 'choices' in samples:
                    text_input = []
                    for i in range(len(samples["text_input"])):
                        this_choices = [f"({string.ascii_lowercase[j]}) {ch}" for j, ch in enumerate(samples["choices"][i])]
                        this_choices = " ".join(this_choices)
                        text_input.append(prompt.format(samples["text_input"][i], this_choices))
            else:
                text_input = [prompt.format(question) for question in samples["text_input"]]
        else:
            text_input = samples["text_input"]

        samples["prompt"] = text_input

        output_text = self.generate(
            samples,
            num_beams=num_beams,
            max_length=max_len,
            min_length=min_len,
            length_penalty=length_penalty
        )

        if "apply_lemmatizer" in samples.keys() and samples["apply_lemmatizer"]:
            output_text = self._lemmatize(output_text)

        return output_text

    def _lemmatize(self, answers):
        def apply(answer):
            doc = self.lemmatizer(answer)

            words = []
            for token in doc:
                if token.pos_ in ["NOUN", "VERB"]:
                    words.append(token.lemma_)
                else:
                    words.append(token.text)
            answer = " ".join(words)

            return answer

        return [apply(answer) for answer in answers]

    @property
    def lemmatizer(self):
        if self._lemmatizer is None:
            try:
                import spacy

                self._lemmatizer = spacy.load("en_core_web_sm")
            except ImportError:
                logging.error(
                    """
                    Please install spacy and en_core_web_sm model to apply lemmatization.
                    python -m spacy download en_core_web_sm
                    OR
                    import spacy.cli
                    spacy.cli.download("en_core_web_sm")
                    """
                )
                exit(1)

        return self._lemmatizer


    @classmethod
    def from_config(cls, cfg):
        breakpoint()
        return cls(device_id=0)

    # @classmethod
    # def from_config(cls, cfg):
    #     vit_model = cfg.get("vit_model", "eva_clip_g")
    #     img_size = cfg.get("image_size")
    #     num_query_token = cfg.get("num_query_token")
    #     llm_model = cfg.get("llm_model")

    #     drop_path_rate = cfg.get("drop_path_rate", 0)
    #     use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
    #     vit_precision = cfg.get("vit_precision", "fp16")
    #     freeze_vit = cfg.get("freeze_vit", True)

    #     prompt = cfg.get("prompt", "")
    #     max_txt_len = cfg.get("max_txt_len", 128)
    #     max_output_txt_len = cfg.get("max_output_txt_len", 256)

    #     apply_lemmatizer = cfg.get("apply_lemmatizer", False)

    #     qformer_text_input = cfg.get("qformer_text_input", True)
    #     breakpoint()
    #     model = cls(
    #         vit_model=vit_model,
    #         img_size=img_size,
    #         drop_path_rate=drop_path_rate,
    #         use_grad_checkpoint=use_grad_checkpoint,
    #         vit_precision=vit_precision,
    #         freeze_vit=freeze_vit,
    #         num_query_token=num_query_token,
    #         llm_model=llm_model,
    #         prompt=prompt,
    #         max_txt_len=max_txt_len,
    #         max_output_txt_len=max_output_txt_len,
    #         apply_lemmatizer=apply_lemmatizer,
    #         qformer_text_input=qformer_text_input,
    #     )

    #     # if qformer_text_input:
    #     #     # Hard-coded to load from BLIP-2 stage-1 pre-trained model (not ideal)
    #     #     model.load_from_pretrained(
    #     #         url_or_filename="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained.pth"
    #     #     )

    #     model.load_checkpoint_from_config(cfg)

    #     return model
