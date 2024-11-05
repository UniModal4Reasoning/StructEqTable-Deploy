import torch
from torch import nn

from transformers import AutoTokenizer
try:
    from lmdeploy import pipeline, GenerationConfig, PytorchEngineConfig, ChatTemplateConfig
except:
    print("\033[93mimport lmdeploy failed, if do not use lmdeploy, ignore this message\033[0m")


class InternVL_LMDeploy(nn.Module):
    def __init__(self, model_path='U4R/StructTable-InternVL2-1B', max_new_tokens=1024, batch_size=4, **kwargs):
        super().__init__()
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.max_batch_size = batch_size

        # init model and tokenizer from ckpt path
        self.init_tokenizer(model_path)
        self.init_model(model_path)

        self.prompt_template = {
            'latex': '<latex>',
            'html': '<html>',
            'markdown': '<markdown>',
        }
        # support output format
        self.supported_output_format = ['latex', 'html', 'markdown']
    
    def init_tokenizer(self, tokenizer_path):
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
            use_fast=False,
        )

    def init_model(self, model_path):
        engine_config = PytorchEngineConfig(
            dtype='bfloat16',
            max_batch_size=self.max_batch_size,
            cache_max_entry_count=0.1
        )
        self.pipeline = pipeline(
            model_path,
            backend_config=engine_config,
            chat_template_config=ChatTemplateConfig(model_name='internvl2-internlm2')
        )

    def forward(self, images, output_format='latex', **kwargs):
        # process image to tokens
        if not isinstance(images, list):
            images = [images] 
        
        prompts = [self.prompt_template[output_format]] * len(images)
        generation_config = GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            temperature=1.0,
            stop_token_ids=[self.tokenizer.eos_token_id],
        )
        
        responses = self.pipeline(
            [(x, y) for x, y in zip(prompts, images)],
            gen_config=generation_config,
        )
        batch_decode_texts = [responce.text for responce in responses]
        return batch_decode_texts
    

