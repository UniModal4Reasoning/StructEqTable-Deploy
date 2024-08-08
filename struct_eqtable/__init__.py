from .model import StructTable

def build_model(model_ckpt, **kwargs):
    tensorrt_path = kwargs.get('tensorrt_path', None)
    if tensorrt_path is not None:
        from .model_trt import StructTableTensorRT
        model = StructTableTensorRT(model_ckpt, **kwargs)
    else:
        model = StructTable(model_ckpt, **kwargs)

    return model