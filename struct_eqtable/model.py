import torch

from torch import nn
from transformers import AutoModelForVision2Seq, AutoProcessor


class StructTable(nn.Module):
    def __init__(self, model_path, max_new_tokens=2048, max_time=60):
        super().__init__()
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.max_generate_time = max_time

        # init model and image processor from ckpt path
        self.init_image_processor(model_path)
        self.init_model(model_path)
    
    def init_model(self, model_path):
        self.model = AutoModelForVision2Seq.from_pretrained(model_path)
        self.model.eval()
    
    def init_image_processor(self, image_processor_path):
        self.data_processor = AutoProcessor.from_pretrained(image_processor_path)

    def forward(self, image):
        # process image to tokens
        image_tokens = self.data_processor.image_processor(
            images=image,
            return_tensors='pt',
        )

        # generate text from image tokens
        model_output = self.model.generate(
            flattened_patches=image_tokens['flattened_patches'],
            attention_mask=image_tokens['attention_mask'], 
            max_new_tokens=self.max_new_tokens,
            max_time=self.max_generate_time
        )
        latex_codes = self.data_processor.batch_decode(model_output, skip_special_tokens=True)

        return latex_codes
