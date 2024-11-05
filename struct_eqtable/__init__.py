from .pix2s import Pix2Struct, Pix2StructTensorRT
from .internvl import InternVL, InternVL_LMDeploy

from transformers import AutoConfig


__ALL_MODELS__ = {
    'Pix2Struct': Pix2Struct,
    'Pix2StructTensorRT': Pix2StructTensorRT,
    'InternVL': InternVL,
    'InternVL_LMDeploy': InternVL_LMDeploy,
}


def get_model_name(model_path):
    model_config = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    if 'Pix2Struct' in model_config.architectures[0]:
        model_name = 'Pix2Struct'
    elif 'InternVL' in model_config.architectures[0]:
        model_name = 'InternVL'
    else:
        raise ValueError(f"Unsupported model type: {model_config.architectures[0]}")

    return model_name


def build_model(model_ckpt='U4R/StructTable-InternVL2-1B', **kwargs):
    model_name = get_model_name(model_ckpt)
    if model_name == 'InternVL' and kwargs.get('lmdeploy', False):
        model_name = 'InternVL_LMDeploy'
    elif model_name == 'Pix2Struct' and kwargs.get('tensorrt_path', None):
        model_name = 'Pix2StructTensorRT'

    model = __ALL_MODELS__[model_name](
        model_ckpt, 
        **kwargs
    )

    return model