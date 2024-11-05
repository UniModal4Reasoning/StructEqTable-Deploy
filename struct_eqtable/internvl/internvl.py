import torch

from torch import nn
from transformers import AutoModel, AutoTokenizer, AutoImageProcessor, GenerationConfig

from .conversation import get_conv_template

class InternVL(nn.Module):
    def __init__(self, model_path='U4R/StructTable-InternVL2-1B', max_new_tokens=1024, max_time=30, flash_attn=True, **kwargs):
        super().__init__()
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.max_generate_time = max_time
        self.flash_attn = flash_attn

        # init model and image processor from ckpt path
        self.init_tokenizer(model_path)
        self.init_image_processor(model_path)
        self.init_model(model_path)

        self.prompt_template = {
            'latex': '<latex>',
            'html': '<html>',
            'markdown': '<markdown>',
        }
        # support output format
        self.supported_output_format = ['latex', 'html', 'markdown']

    def init_model(self, model_path):
        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=self.flash_attn,
        )
        self.model.eval()
    
    def init_image_processor(self, image_processor_path):
        self.image_processor = AutoImageProcessor.from_pretrained(
            image_processor_path,
            trust_remote_code=True,
        )

    def init_tokenizer(self, tokenizer_path):
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
            use_fast=False,
        )

        self.image_context_token = '<IMG_CONTEXT>'
        self.image_token_num = 256
        self.image_start_token = '<img>'
        self.image_end_token = '</img>'
        self.img_context_token_id = self.tokenizer.convert_tokens_to_ids(self.image_context_token)
    
    def format_image_tokens(self, path_num):
        return f'{self.image_start_token}{self.image_context_token* self.image_token_num * path_num}{self.image_end_token}'

    def forward(self, images, output_format='latex', **kwargs):
        # process image to tokens
        if not isinstance(images, list):
            images = [images] 
        
        pixel_values_list = []
        for image in images:
            path_images = self.dynamic_preprocess(
                image, image_size=448, max_num=12
            )
            pixel_values = self.image_processor(
                path_images, 
                return_tensors='pt'
            )['pixel_values'].to(torch.bfloat16)
            pixel_values_list.append(pixel_values)
        
        batch_size = len(pixel_values_list)
        conversation_list = []
        for bs_idx in range(batch_size):
            pixel_values= pixel_values_list[bs_idx].to(torch.bfloat16)

            image_tokens = self.format_image_tokens(pixel_values.shape[0])
            question = '<image>\n' + self.prompt_template[output_format]
            answer = None
        
            template = get_conv_template(self.model.config.template)
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], answer)
            conversation = template.get_prompt()
            conversation = conversation.replace('<image>', image_tokens, 1)
            conversation_list.append(conversation)

        device = next(self.parameters()).device
        self.tokenizer.padding_side = 'left'
        model_inputs = self.tokenizer(
            conversation_list, 
            return_tensors='pt', 
            padding=True,
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        ).to(device)
        pixel_values = torch.cat(pixel_values_list, axis=0).to(device)

        # generation config
        generation_config = dict(
            max_new_tokens=self.max_new_tokens,
            max_time=self.max_generate_time,
            img_context_token_id=self.img_context_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            do_sample=False,
            no_repeat_ngram_size=20,
        )

        # generate text from image tokens
        model_output = self.model.generate(
            pixel_values=pixel_values,
            input_ids=model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask, 
            **generation_config,
            # **kwargs
        )

        batch_decode_texts = self.tokenizer.batch_decode(
            model_output,
            skip_special_tokens=True
        )
        return batch_decode_texts
    
    def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def dynamic_preprocess(self, image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # calculate the existing image aspect ratio
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # find the closest aspect ratio to the target
        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)

        # calculate the target width and height
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            # split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images
