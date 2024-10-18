import os
import time
import json

import torch
import torch.nn as nn

try:
    import tensorrt_llm
    import tensorrt as trt
    import tensorrt_llm.profiler as profiler

    from tensorrt_llm._utils import str_dtype_to_trt, torch_to_numpy
    from tensorrt_llm.lora_manager import LoraManager
    from tensorrt_llm.runtime import Session, TensorInfo, ModelConfig, SamplingConfig
except:
    print("\033[93mimport tensorrt_llm failed, if do not use tensorrt, ignore this message\033[0m")

from typing import List
from transformers import AutoProcessor, AutoTokenizer, AutoConfig


def trt_dtype_to_torch(dtype):
    if dtype == trt.float16:
        return torch.float16
    elif dtype == trt.float32:
        return torch.float32
    elif dtype == trt.int32:
        return torch.int32
    elif dtype == trt.bfloat16:
        return torch.bfloat16
    else:
        raise TypeError("%s is not supported" % dtype)


class Pix2StructTensorRT(nn.Module):

    def __init__(self, model_path, tensorrt_path, batch_size=1, max_new_tokens=4096, **kwargs):
        
        self.model_ckpt_path = model_path
        self.tensorrt_path = tensorrt_path
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens

        self.llm_engine_path = os.path.join(tensorrt_path, 'llm_engines')
        self.visual_engine_path = os.path.join(tensorrt_path, 'visual_engines')
        
        device_id = torch.cuda.current_device() % torch.cuda.device_count()
        self.device_id = device_id
        self.device = "cuda:%d" % (device_id)
        
        self.stream = torch.cuda.Stream(torch.cuda.current_device())
        torch.cuda.set_stream(self.stream)

        # parse model type from visual engine config
        with open(os.path.join(self.visual_engine_path, "config.json"),
                  "r") as f:
            config = json.load(f)
        self.model_type = config['builder_config']['model_type']
        self.vision_precision = config['builder_config']['precision']

        self.vision_precision = 'float16'
        self.decoder_llm = not (
            't5' in self.model_type
            or self.model_type in ['nougat', 'pix2struct', 'StructEqTable']
        )  # BLIP2-T5, pix2struct and Nougat are using encoder-decoder models as LLMs

        self.profiling_iterations = 20

        self.init_image_encoder()
        self.init_tokenizer()
        self.init_llm()
        self.init_image_processor()

        self.special_str_list = ['\\midrule', '\\hline']
        self.supported_output_format = ['latex']

    def postprocess_latex_code(self, code):
        for special_str in self.special_str_list:
            code = code.replace(special_str, special_str + ' ')
        return code

    def init_image_processor(self):
        self.data_processor = AutoProcessor.from_pretrained(
            self.model_ckpt_path)

    def init_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_ckpt_path, use_fast=True, use_legacy=False)
        # self.tokenizer.padding_side = "right"

    def init_image_encoder(self):
        vision_encoder_path = os.path.join(self.visual_engine_path,
                                           'visual_encoder.engine')
        with open(vision_encoder_path, 'rb') as f:
            engine_buffer = f.read()
        self.visual_encoder_session = Session.from_serialized_engine(
            engine_buffer)

    def init_llm(self):

        self.model = TRTLLMEncDecModel.from_engine(
            os.path.basename(self.model_ckpt_path),
            self.llm_engine_path,
            skip_encoder=self.model_type in ['nougat', 'pix2struct', 'StructEqTable'],
            debug_mode=False,
            stream=self.stream)

        self.model_config = self.model.decoder_model_config
        self.runtime_mapping = self.model.decoder_runtime_mapping

    def __call__(self, image, **kwargs):
        # process image to tokens
        image_tokens = self.data_processor.image_processor(
            images=image,
            return_tensors='pt',
        )

        for k, v in image_tokens.items():
            image_tokens[k] = v.cuda()

        model_output = self.run(
            flattened_patches=image_tokens['flattened_patches'],
            attention_mask=image_tokens['attention_mask'], 
            max_new_tokens=self.max_new_tokens
        )

        # postprocess
        latex_codes = []
        for i, code in enumerate(model_output):
            latex_codes.append(self.postprocess_latex_code(code[0]))

        return latex_codes

    def preprocess(self, warmup, pre_prompt, post_prompt, image,
                   attention_mask):
        if not warmup:
            profiler.start("Vision")

        visual_features, visual_atts = self.get_visual_features(
            torch.stack(image['image_patches'], dim=0)
            if self.model_type == 'fuyu' else image, attention_mask)

        if not warmup:
            profiler.stop("Vision")
       
        pre_input_ids = self.tokenizer(pre_prompt,
                                        return_tensors="pt",
                                        padding=True).input_ids
        if post_prompt[0] is not None:
            post_input_ids = self.tokenizer(post_prompt,
                                            return_tensors="pt",
                                            padding=True).input_ids
            length = pre_input_ids.shape[1] + post_input_ids.shape[
                1] + visual_atts.shape[1]
        else:
            post_input_ids = None
            length = pre_input_ids.shape[1] + visual_atts.shape[1]

        input_lengths = torch.IntTensor([length] * 1).to(
            torch.int32)

        input_ids, ptuning_args = self.setup_fake_prompts(
            visual_features, pre_input_ids, post_input_ids, input_lengths)

        return input_ids, input_lengths, ptuning_args, visual_features

    def generate(self, pre_prompt, post_prompt, image, decoder_input_ids,
                 max_new_tokens, attention_mask, warmup):
        if not warmup:
            profiler.start("Generate")

        input_ids, input_lengths, ptuning_args, visual_features = self.preprocess(
            warmup, pre_prompt, post_prompt, image, attention_mask)

        if warmup: return None

        profiler.start("LLM")

        # Trim encoder input_ids to match visual features shape
        ids_shape = (self.batch_size, visual_features.shape[1])

        input_ids = torch.ones(ids_shape, dtype=torch.int32)

        output_ids = self.model.generate(
            input_ids,
            decoder_input_ids,
            max_new_tokens,
            num_beams=1,
            bos_token_id=self.tokenizer.bos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            debug_mode=False,
            prompt_embedding_table=ptuning_args[0],
            prompt_tasks=ptuning_args[1],
            prompt_vocab_size=ptuning_args[2],
            attention_mask=attention_mask)

        # Reset input_lengths to match decoder_input_ids
        input_lengths = torch.ones(input_lengths.shape,
                                    dtype=input_lengths.dtype)
        profiler.stop("LLM")

        if tensorrt_llm.mpi_rank() == 0:
            # Extract a list of tensors of shape beam_width x output_ids.
            output_beams_list = [
                self.tokenizer.batch_decode(
                    output_ids[batch_idx, :, input_lengths[batch_idx]:],
                    skip_special_tokens=True)
                for batch_idx in range(self.batch_size)
            ]

            stripped_text = [[
                output_beams_list[batch_idx][beam_idx].strip()
                for beam_idx in range(1)
            ] for batch_idx in range(self.batch_size)]
            profiler.stop("Generate")
            return stripped_text
        else:
            profiler.stop("Generate")
            return None
        
    def get_visual_features(self, image, attention_mask):
        visual_features = {
            'input':
            image.to(
                tensorrt_llm._utils.str_dtype_to_torch(self.vision_precision))
        }
        if attention_mask is not None:
            visual_features['attention_mask'] = attention_mask
        tensor_info = [
            TensorInfo('input', str_dtype_to_trt(self.vision_precision),
                       image.shape)
        ]
        if attention_mask is not None:
            tensor_info.append(
                TensorInfo('attention_mask', trt.DataType.INT32,
                           attention_mask.shape))
        visual_output_info = self.visual_encoder_session.infer_shapes(
            tensor_info)
        visual_outputs = {
            t.name: torch.empty(tuple(t.shape),
                                dtype=trt_dtype_to_torch(t.dtype),
                                device=image.device)
            for t in visual_output_info
        }

        ok = self.visual_encoder_session.run(visual_features, visual_outputs,
                                             self.stream.cuda_stream)
        assert ok, "Runtime execution failed for vision encoder session"
        self.stream.synchronize()

        image_embeds = visual_outputs['output']
        image_atts = torch.ones(image_embeds.size()[:-1],
                                dtype=torch.long).to(image.device)

        return image_embeds, image_atts
    
    def setup_fake_prompts(self, visual_features, pre_input_ids, post_input_ids,
                           input_lengths):
        # Assemble fake prompts which points to image embedding actually
        fake_prompt_id = torch.arange(
            self.model_config.vocab_size, self.model_config.vocab_size +
            visual_features.shape[0] * visual_features.shape[1])
        fake_prompt_id = fake_prompt_id.reshape(visual_features.shape[0],
                                                visual_features.shape[1])

        if post_input_ids is not None:
            input_ids = [pre_input_ids, fake_prompt_id, post_input_ids]
        else:
            input_ids = [fake_prompt_id, pre_input_ids]
        
        input_ids = torch.cat(input_ids, dim=1).contiguous().to(torch.int32)

        if self.decoder_llm or self.runtime_mapping.is_first_pp_rank():
            ptuning_args = self.ptuning_setup(visual_features, input_ids,
                                              input_lengths)
        else:
            ptuning_args = [None, None, None]

        return input_ids, ptuning_args

    def ptuning_setup(self, prompt_table, input_ids, input_lengths):
        hidden_size = self.model_config.hidden_size * self.runtime_mapping.tp_size
        if prompt_table is not None:
            task_vocab_size = torch.tensor(
                [prompt_table.shape[1]],
                dtype=torch.int32,
            ).cuda()
            prompt_table = prompt_table.view(
                (prompt_table.shape[0] * prompt_table.shape[1],
                 prompt_table.shape[2]))
            assert prompt_table.shape[
                1] == hidden_size, "Prompt table dimensions do not match hidden size"

            prompt_table = prompt_table.cuda().to(
                dtype=tensorrt_llm._utils.str_dtype_to_torch(
                    self.model_config.dtype))
        else:
            prompt_table = torch.empty([1, hidden_size]).cuda()
            task_vocab_size = torch.zeros([1]).cuda()

        if self.model_config.remove_input_padding:
            tasks = torch.zeros([torch.sum(input_lengths)],
                                dtype=torch.int32).cuda()
            if self.decoder_llm: tasks = tasks.unsqueeze(0)
        else:
            tasks = torch.zeros(input_ids.shape, dtype=torch.int32).cuda()

        return [prompt_table, tasks, task_vocab_size]

    def setup_inputs(self, input_text, raw_image):
        attention_mask = None
       
        image_processor = AutoProcessor.from_pretrained(self.model_ckpt_path)
        if input_text is None:
            input_text = ""
        inputs = image_processor(
            images=raw_image,
            text=input_text,
            return_tensors="pt",
        )
        image = inputs['flattened_patches']
        image = image.expand(self.batch_size, -1, -1).contiguous()
        attention_mask = inputs['attention_mask'].to(self.device).to(
            torch.int)
        attention_mask = attention_mask.expand(self.batch_size,
                                                -1).contiguous()
        pre_prompt = ""
        post_prompt = None

        # Repeat inputs to match batch size
        pre_prompt = [pre_prompt] * self.batch_size
        post_prompt = [post_prompt] * self.batch_size
        image = image.to(self.device)

        # Generate decoder_input_ids for enc-dec models
        # Custom prompts can be added as:
        # decoder_input_ids = model.tokenizer(decoder_prompt).input_ids
        if self.decoder_llm:
            decoder_input_ids = None
        else:
            config = AutoConfig.from_pretrained(self.model_ckpt_path)
            decoder_start_id = config.decoder_start_token_id  # T5
            if decoder_start_id is None:
                decoder_start_id = config.decoder.bos_token_id  # Nougat

            decoder_input_ids = torch.IntTensor([[decoder_start_id]])
            decoder_input_ids = decoder_input_ids.repeat((self.batch_size, 1))

        return input_text, pre_prompt, post_prompt, image, decoder_input_ids, attention_mask

    def run(self, flattened_patches, attention_mask, max_new_tokens):
        # input_text, pre_prompt, post_prompt, processed_image, decoder_input_ids, attention_mask = self.setup_inputs(
        #     None, raw_image)
        pre_prompt = [""] * self.batch_size
        post_prompt = [None] * self.batch_size
        config = AutoConfig.from_pretrained(self.model_ckpt_path)
        decoder_start_id = config.decoder_start_token_id  # T5 
        decoder_input_ids = torch.IntTensor([[decoder_start_id]])
        decoder_input_ids = decoder_input_ids.repeat((self.batch_size, 1))

        processed_image = flattened_patches.expand(self.batch_size, -1, -1).contiguous()
        attention_mask = attention_mask.to(self.device).to(torch.int)
        attention_mask = attention_mask.expand(self.batch_size,-1).contiguous()

        self.generate(pre_prompt,
                       post_prompt,
                       processed_image,
                       decoder_input_ids,
                       max_new_tokens,
                       attention_mask=attention_mask,
                       warmup=True)
        # num_iters = self.profiling_iterations if self.args.run_profiling else 1
        num_iters = 1
        # print(num_iters)
        for _ in range(num_iters):
            output_text = self.generate(pre_prompt,
                                         post_prompt,
                                         processed_image,
                                         decoder_input_ids,
                                         max_new_tokens,
                                         attention_mask=attention_mask,
                                         warmup=False)
        # if self.runtime_rank == 0:
        #     self.print_result(input_text, output_text)
        return output_text


def read_config(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)

    builder_config = config['build_config']
    plugin_config = builder_config['plugin_config']
    pretrained_config = config['pretrained_config']
    lora_config = builder_config['lora_config']
    auto_parallel_config = builder_config['auto_parallel_config']
    use_gpt_attention_plugin = plugin_config["gpt_attention_plugin"]
    remove_input_padding = plugin_config["remove_input_padding"]
    use_lora_plugin = plugin_config["lora_plugin"]
    tp_size = pretrained_config['mapping']['tp_size']
    pp_size = pretrained_config['mapping']['pp_size']
    gpus_per_node = auto_parallel_config['gpus_per_node']
    world_size = tp_size * pp_size
    assert world_size == tensorrt_llm.mpi_world_size(), \
        f'Engine world size ({world_size}) != Runtime world size ({tensorrt_llm.mpi_world_size()})'
    num_heads = pretrained_config["num_attention_heads"]
    hidden_size = pretrained_config["hidden_size"]
    head_size = pretrained_config["head_size"]
    vocab_size = pretrained_config["vocab_size"]
    max_batch_size = builder_config["max_batch_size"]
    max_beam_width = builder_config["max_beam_width"]
    num_layers = pretrained_config["num_hidden_layers"]
    num_kv_heads = pretrained_config.get('num_kv_heads', num_heads)

    assert (num_heads % tp_size) == 0
    num_heads = num_heads // tp_size
    hidden_size = hidden_size // tp_size
    num_kv_heads = (num_kv_heads + tp_size - 1) // tp_size

    cross_attention = pretrained_config["architecture"] == "DecoderModel"
    skip_cross_qkv = pretrained_config.get('skip_cross_qkv', False)
    has_position_embedding = pretrained_config["has_position_embedding"]
    has_token_type_embedding = hasattr(pretrained_config, "type_vocab_size")
    use_custom_all_reduce = plugin_config.get('use_custom_all_reduce', False)
    dtype = pretrained_config["dtype"]

    paged_kv_cache = plugin_config['paged_kv_cache']
    tokens_per_block = plugin_config['tokens_per_block']

    gather_context_logits = builder_config.get('gather_context_logits', False)
    gather_generation_logits = builder_config.get('gather_generation_logits',
                                                  False)
    max_prompt_embedding_table_size = builder_config.get(
        'max_prompt_embedding_table_size', 0)

    model_config = ModelConfig(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        hidden_size=hidden_size,
        head_size=head_size,
        max_batch_size=max_batch_size,
        max_beam_width=max_beam_width,
        vocab_size=vocab_size,
        num_layers=num_layers,
        gpt_attention_plugin=use_gpt_attention_plugin,
        remove_input_padding=remove_input_padding,
        paged_kv_cache=paged_kv_cache,
        tokens_per_block=tokens_per_block,
        cross_attention=cross_attention,
        has_position_embedding=has_position_embedding,
        has_token_type_embedding=has_token_type_embedding,
        use_custom_all_reduce=use_custom_all_reduce,
        dtype=dtype,
        gather_context_logits=gather_context_logits,
        gather_generation_logits=gather_generation_logits,
        max_prompt_embedding_table_size=max_prompt_embedding_table_size,
        lora_plugin=use_lora_plugin,
        lora_target_modules=lora_config.get('lora_target_modules'),
        trtllm_modules_to_hf_modules=lora_config.get(
            'trtllm_modules_to_hf_modules'),
        skip_cross_qkv=skip_cross_qkv,
    )

    return model_config, tp_size, pp_size, gpus_per_node, dtype


class Mapping(object):
    def __init__(
            self,
            world_size=1,
            rank=0,
            gpus_per_node=8,
            tp_size=1,
            pp_size=1,
            moe_tp_size=-1,  # -1 means no moe
            moe_ep_size=-1):  # -1 means no moe
        # set default values for non-moe cases
        if moe_tp_size == -1:
            moe_tp_size = tp_size
            moe_ep_size = 1

        if pp_size * tp_size != world_size:
            raise ValueError(
                f"world_size must equal to pp_size * tp_size, but got {world_size} != {pp_size} * {tp_size}"
            )

        moe_tp_ep_size = moe_tp_size * moe_ep_size
        if moe_tp_ep_size != tp_size:
            raise ValueError(
                f"tp_size must equal to moe_tp_size * moe_ep_size, but got {tp_size} != {moe_tp_size} * {moe_ep_size}"
            )

        self.tp_size = tp_size
        self.pp_size = pp_size
        self.moe_tp_size = moe_tp_size
        self.moe_ep_size = moe_ep_size
        self.world_size = world_size
        self.rank = rank
        self.gpus_per_node = gpus_per_node

        self.pp_groups = []
        self.tp_groups = []
        self.moe_tp_groups = []
        self.moe_ep_groups = []

        # init pp group
        for i in range(tp_size):
            ranks = range(i+ self.rank, world_size+ self.rank, tp_size)
            self.pp_groups.append(list(ranks))

        # init tp group
        for i in range(pp_size):
            ranks = range(i * tp_size + self.rank, (i + 1) * tp_size + self.rank)
            self.tp_groups.append(list(ranks))

        # init moe tp group
        for i in range(pp_size):
            for j in range(moe_ep_size):
                ranks = range(i * moe_tp_ep_size + j, (i + 1) * moe_tp_ep_size,
                              moe_ep_size)
                self.moe_tp_groups.append(list(ranks))

        # init moe ep group
        for i in range(pp_size):
            for j in range(moe_tp_size):
                ranks = range(i * moe_tp_ep_size + j * moe_ep_size,
                              i * moe_tp_ep_size + (j + 1) * moe_ep_size)
                self.moe_ep_groups.append(list(ranks))

        # self.pp_rank = self.rank // self.tp_size
        # self.tp_rank = self.rank % self.tp_size
        self.pp_rank = 0
        self.tp_rank = 0
        self.moe_tp_rank = self.tp_rank // self.moe_ep_size
        self.moe_ep_rank = self.tp_rank % self.moe_ep_size

        # self.tp_group = self.tp_groups[self.pp_rank]
        # self.pp_group = self.pp_groups[self.tp_rank]
        self.moe_tp_group = self.moe_tp_groups[self.pp_rank * moe_ep_size +
                                               self.moe_ep_rank]
        self.moe_ep_group = self.moe_ep_groups[self.pp_rank * moe_tp_size +
                                               self.moe_tp_rank]

        self.node_rank = self.rank // self.gpus_per_node
        self.local_rank = self.rank % self.gpus_per_node

    def get_node_rank(self, rank: int):
        return rank // self.gpus_per_node

    def get_local_rank(self, rank: int):
        return rank % self.gpus_per_node

    def has_tp(self):
        return self.tp_size > 1

    def is_last_pp_rank(self):
        return self.pp_rank == self.pp_size - 1

    def is_first_pp_rank(self):
        return self.pp_rank == 0

    def has_pp(self):
        return self.pp_size > 1

    def prev_pp_rank(self):
        p = self.rank - self.tp_size
        if p < 0:
            p = p + self.world_size
        return p

    def next_pp_rank(self):
        p = self.rank + self.tp_size
        if p >= self.world_size:
            p = p - self.world_size
        return p

    def has_moe_tp(self):
        return self.moe_tp_size > 1

    def has_moe_ep(self):
        return self.moe_ep_size > 1

    def pp_layers(self, num_layers: int) -> List[int]:
        layers_per_pipeline_stage = num_layers // self.pp_size
        layers_range = range(self.pp_rank * layers_per_pipeline_stage,
                             (self.pp_rank + 1) * layers_per_pipeline_stage)
        return list(layers_range)

    def ep_experts(self, num_experts: int) -> List[int]:
        experts_per_rank = num_experts // self.moe_ep_size
        experts_range = range(self.moe_ep_rank * experts_per_rank,
                              (self.moe_ep_rank + 1) * experts_per_rank)
        return list(experts_range)


def get_engine_name(rank):
    return 'rank{}.engine'.format(rank)

class TRTLLMEncDecModel:

    def __init__(
        self,
        engine_name,
        engine_dir,
        lora_dir=None,
        lora_task_uids=None,
        debug_mode=False,
        skip_encoder=False,
        stream: torch.cuda.Stream = None,
    ):
        # in multi-node setup, it's important to set_device at the very beginning so .to('cuda') refers to current device
        # accordingly, all input & output tensors should be moved to current device
        # otherwise, it's default to 'cuda:0'
        
        # self.runtime_rank = tensorrt_llm.mpi_rank()
        self.device_id = torch.cuda.current_device()
        # torch.cuda.set_device(device_id)
        self.device = torch.cuda.current_device()
        self.skip_encoder = skip_encoder
        self.lora_task_uids = lora_task_uids

        # when enc-dec runs by itself, stream can be None and we create new stream here
        # when enc-dec has to run as a component in a bigger workflow (e.g., multimodal), earlier components in the workflow may have results in its stream, which we should pass that stream in to avoid unnecessary stream sync
        self.stream = stream
        if self.stream is None:
            self.stream = torch.cuda.Stream(self.device)
        torch.cuda.set_stream(self.stream)

        def engine_setup(component):
            # model config
            config_path = os.path.join(engine_dir, component, "config.json")
            model_config, tp_size, pp_size, gpus_per_node, dtype = read_config(
                config_path)

            # MGMN config
            world_size = tp_size * pp_size
            # runtime_rank = tensorrt_llm.mpi_rank()
            runtime_rank = torch.cuda.current_device()
            # assert runtime_rank < world_size, "Runtime GPU rank exceeds MPI world size. Did you launch more MPI processes than required?"
            # runtime_mapping = tensorrt_llm.Mapping(world_size,
            #                                        runtime_rank,
            #                                        tp_size=tp_size,
            #                                        pp_size=pp_size,
            #                                        gpus_per_node=gpus_per_node)
            # tensorrt_llm.Mapping
            runtime_mapping = Mapping(world_size,
                                      runtime_rank,
                                      tp_size=tp_size,
                                      pp_size=pp_size,
                                      gpus_per_node=gpus_per_node)
            # load engine
            # engine_fname = get_engine_name(runtime_rank)
            engine_fname = get_engine_name(0)
            with open(os.path.join(engine_dir, component, engine_fname), "rb") as f:
                engine_buffer = f.read()

            return model_config, runtime_mapping, engine_buffer

        # Note: encoder and decoder doesn't necessarily have the same TP & PP config

        if not skip_encoder:
            self.encoder_model_config, self.encoder_runtime_mapping, encoder_engine_buffer = engine_setup(
                component='encoder')

            self.nccl_comm = None
            if self.encoder_runtime_mapping.has_pp():
                # for Pipeline Parallelism in encoder
                self.nccl_comm = torch.classes.trtllm.NcclCommunicatorOp(
                    self.encoder_runtime_mapping.tp_size,
                    self.encoder_runtime_mapping.pp_size,
                    self.encoder_runtime_mapping.rank)

            # session setup
            self.encoder_session = tensorrt_llm.runtime.Session.from_serialized_engine(
                encoder_engine_buffer)

            # encoder lora manager setup
            if self.encoder_model_config.lora_plugin:
                self.encoder_lora_manager = LoraManager()
                # TODO: this is only for bart
                self.encoder_lora_manager.load_from_hf(
                    model_dirs=lora_dir,
                    model_config=self.encoder_model_config,
                    runtime_mapping=self.encoder_runtime_mapping,
                    component='encoder',
                )
            else:
                self.encoder_lora_manager = None
        else:
            self.encoder_model_config, self.encoder_runtime_mapping, encoder_engine_buffer = None, None, None
            self.nccl_comm, self.encoder_session = None, None

        self.decoder_model_config, self.decoder_runtime_mapping, decoder_engine_buffer = engine_setup(
            component='decoder')

        self.decoder_session = tensorrt_llm.runtime.GenerationSession(
            self.decoder_model_config,
            decoder_engine_buffer,
            self.decoder_runtime_mapping,
            debug_mode=debug_mode)

        # decoder lora manager setup
        if self.decoder_model_config.lora_plugin:
            self.decoder_lora_manager = LoraManager()
            # TODO: this is only for bart
            self.decoder_lora_manager.load_from_hf(
                model_dirs=lora_dir,
                model_config=self.decoder_model_config,
                runtime_mapping=self.decoder_runtime_mapping,
                component='decoder',
            )
        else:
            self.decoder_lora_manager = None
    
    @classmethod
    def from_engine(cls,
                    engine_name,
                    engine_dir,
                    lora_dir=None,
                    lora_task_uids=None,
                    debug_mode=False,
                    skip_encoder=False,
                    stream=None):
        return cls(engine_name,
                   engine_dir,
                   lora_dir,
                   lora_task_uids,
                   debug_mode=debug_mode,
                   skip_encoder=skip_encoder,
                   stream=stream)

    def process_input(self,
                      input_ids,
                      remove_input_padding=False,
                      pad_token_id=0,
                      prompt_tasks=None):
        if remove_input_padding:
            # in remove padding mode --> flatten input, calculate actual length and max length
            # Note: 1st token should never be removed, even if it is pad_token_id
            first_ids = input_ids[:, 0]
            input_ids = input_ids[:, 1:]
            input_lengths = 1 + (input_ids != pad_token_id).sum(dim=1).type(
                torch.IntTensor).to(self.device)  # [batch_size]
            new_ids = []
            for i in range(len(input_ids)):
                row = input_ids[i, :]
                row = row[row != pad_token_id]
                new_ids.append(
                    torch.cat(
                        (torch.IntTensor([first_ids[i]]).to(self.device), row)))
            input_ids = torch.cat(new_ids)  # [num_tokens]
            if prompt_tasks is not None:
                prompt_tasks = prompt_tasks[:input_ids.shape[0]]
        else:
            # in padding mode --> keep input, just calculate actual length and max length
            # Note: 1st token should always count, even if it is pad_token_id. e.g., decoder start id in enc-dec models could be a single pad_token_id, we should count
            input_lengths = torch.tensor(
                1 + (input_ids[:, 1:] != pad_token_id).sum(dim=1).type(
                    torch.IntTensor).to(self.device),
                dtype=torch.int32,
                device=self.device)
        max_input_length = torch.max(input_lengths).item()
        return input_ids, input_lengths, max_input_length, prompt_tasks

    def encoder_run(self,
                    input_ids,
                    input_lengths,
                    max_input_length,
                    position_ids=None,
                    token_type_ids=None,
                    debug_mode=False,
                    prompt_embedding_table=None,
                    prompt_tasks=None,
                    prompt_vocab_size=None,
                    attention_mask=None):

        # each engine has hidden_dim/TP, don't forget to multiply TP
        hidden_size = self.encoder_model_config.hidden_size * self.encoder_runtime_mapping.tp_size
        if input_ids.dim() == 1:
            hidden_states_shape = (input_ids.shape[0], hidden_size
                                   )  # [num_tokens,D]
        else:
            hidden_states_shape = (input_ids.shape[0], input_ids.shape[1],
                                   hidden_size)  # [BS,seqlen,D]
        hidden_states_dtype = lambda name: trt_dtype_to_torch(
            self.encoder_session.engine.get_tensor_dtype(name))

        # input tensors. only first PP rank has id input, others are hidden_states input
        inputs = {}
        if self.encoder_runtime_mapping.is_first_pp_rank():
            inputs['input_ids'] = input_ids.contiguous()
            if self.encoder_model_config.has_position_embedding:
                if position_ids is None:
                    if self.encoder_model_config.remove_input_padding:
                        position_ids = [
                            torch.arange(sample_length,
                                         dtype=torch.int32,
                                         device=input_ids.device)
                            for sample_length in torch_to_numpy(input_lengths)
                        ]
                        position_ids = torch.cat(position_ids)
                    else:
                        bsz, seq_len = input_ids.shape[:2]
                        position_ids = torch.arange(
                            seq_len, dtype=torch.int32,
                            device=input_ids.device).expand(bsz, -1)
                inputs['position_ids'] = position_ids.contiguous()
            if self.encoder_model_config.has_token_type_embedding:
                inputs['token_type_ids'] = token_type_ids.contiguous()

            if self.encoder_model_config.max_prompt_embedding_table_size > 0:
                inputs[
                    'prompt_embedding_table'] = prompt_embedding_table.contiguous(
                    )
                inputs['tasks'] = prompt_tasks.contiguous()
                inputs['prompt_vocab_size'] = prompt_vocab_size.contiguous()
        else:
            # just need a placeholder, engine will call NCCL to recv and fill data from previous rank
            inputs['hidden_states_input'] = torch.empty(
                hidden_states_shape,
                dtype=hidden_states_dtype('hidden_states_input'),
                device=self.device).contiguous()
        if attention_mask is not None and not self.encoder_model_config.gpt_attention_plugin:
            inputs['attention_mask'] = attention_mask.contiguous()

        inputs['input_lengths'] = input_lengths
        # use shape info to pass max length info in remove padding mode
        inputs['max_input_length'] = torch.empty(
            (max_input_length, ),
            dtype=hidden_states_dtype('max_input_length'),
            device=self.device).contiguous()
        batch_size = input_lengths.size(0)
        inputs['host_request_types'] = torch.IntTensor([0] *
                                                       batch_size).to('cpu')
        if self.encoder_model_config.remove_input_padding:
            inputs['host_context_lengths'] = input_lengths.to('cpu')

        if self.encoder_model_config.lora_plugin and self.encoder_lora_manager is not None:
            inputs.update(
                self.encoder_lora_manager.input_buffers(
                    self.lora_task_uids,
                    self.encoder_runtime_mapping,
                    self.encoder_model_config.num_layers,
                ))

        # Note: runtime.Session's run() method will set input/output tensor address, here we only need to provide tensor shape
        self.encoder_session.set_shapes(inputs)

        # output tensors. only last PP rank final encoder output, others are intermediate hidden_states output. Need broadcast later
        outputs = {}
        if self.encoder_runtime_mapping.is_last_pp_rank():
            outputs['encoder_output'] = torch.empty(
                hidden_states_shape,
                dtype=hidden_states_dtype('encoder_output'),
                device=self.device).contiguous()
        else:
            outputs['hidden_states_output'] = torch.empty(
                hidden_states_shape,
                dtype=hidden_states_dtype('hidden_states_output'),
                device=self.device).contiguous()

        # -------------------------------------------
        if debug_mode:
            engine = self.encoder_session.engine
            context = self.encoder_session.context
            # setup debugging buffer for the encoder
            for i in range(self.encoder_session.engine.num_io_tensors):
                name = engine.get_tensor_name(i)
                if engine.get_tensor_mode(
                        name
                ) == trt.TensorIOMode.OUTPUT and name not in outputs.keys():
                    dtype = engine.get_tensor_dtype(name)
                    shape = context.get_tensor_shape(name)
                    outputs[name] = torch.zeros(tuple(shape),
                                                dtype=trt_dtype_to_torch(dtype),
                                                device=self.device)
                    context.set_tensor_address(name, outputs[name].data_ptr())
        # -------------------------------------------

        # TRT session run
        # Note: need cuda stream ID, not a torch Stream
        ok = self.encoder_session.run(inputs, outputs, self.stream.cuda_stream)
        assert ok, "Runtime execution failed"
        self.stream.synchronize()

        # Tensor Parallelism is handled by model/engine definition
        # But we need to broadcast among PP group at the end of encoder's Pipeline Parallelism
        # After this, all ranks should recv the encoder output, and world might be re-configured using decoder's TP-PP config
        def pp_communicate_encoder_output(encoder_output):
            if self.encoder_runtime_mapping.is_last_pp_rank():
                for pp_rank in self.encoder_runtime_mapping.pp_group:
                    if pp_rank != self.encoder_runtime_mapping.rank:
                        self.nccl_comm.send(encoder_output, pp_rank)
                return encoder_output
            else:
                self.nccl_comm.recv(encoder_output,
                                    self.encoder_runtime_mapping.pp_group[-1])
                return encoder_output

        if self.encoder_runtime_mapping.has_pp():
            # use hidden_states output buffer to receive output as the shapes are same
            encoder_output_buf = outputs[
                'encoder_output'] if self.encoder_runtime_mapping.is_last_pp_rank(
                ) else outputs['hidden_states_output']
            encoder_output = pp_communicate_encoder_output(encoder_output_buf)
        else:
            encoder_output = outputs['encoder_output']

        return encoder_output

    def generate(self,
                 encoder_input_ids,
                 decoder_input_ids,
                 max_new_tokens,
                 num_beams=1,
                 pad_token_id=None,
                 eos_token_id=None,
                 bos_token_id=None,
                 debug_mode=False,
                 return_dict=False,
                 prompt_embedding_table=None,
                 prompt_tasks=None,
                 prompt_vocab_size=None,
                 attention_mask=None,
                 time_encoder=False,
                 return_encoder_output=False):
        ## ensure all externally provided tensors are on the correct device.
        encoder_input_ids = encoder_input_ids.to(self.device)
        decoder_input_ids = decoder_input_ids.to(self.device)

        if attention_mask is not None:
            attention_mask = torch.tensor(attention_mask,
                                          dtype=torch.int32,
                                          device=self.device)

        ## encoder run
        encoder_remove_input_padding = self.encoder_model_config.remove_input_padding if self.encoder_model_config else self.decoder_model_config.remove_input_padding

        encoder_input_ids, encoder_input_lengths, encoder_max_input_length, prompt_tasks = self.process_input(
            encoder_input_ids, encoder_remove_input_padding, pad_token_id,
            prompt_tasks)

        if not self.skip_encoder:
            #logger.info(f"Rank {self.runtime_rank} Running encoder engine ...")
            if time_encoder:
                tik = time.time()
            encoder_output = self.encoder_run(
                encoder_input_ids,
                encoder_input_lengths,
                encoder_max_input_length,
                debug_mode=debug_mode,
                prompt_embedding_table=prompt_embedding_table,
                prompt_tasks=prompt_tasks,
                prompt_vocab_size=prompt_vocab_size,
                attention_mask=attention_mask)
            if time_encoder:
                tok = time.time()
                print(f"TRT-LLM Encoder time {(tok-tik)*1000}ms")
        else:
            encoder_output = prompt_embedding_table
            if encoder_input_ids.dim() > 1:
                encoder_output = encoder_output.unsqueeze(0)

        ## decoder run
        # logger.info(f"Rank {self.runtime_rank} Running decoder engine ...")
        decoder_input_ids, decoder_input_lengths, decoder_max_input_length, _ = self.process_input(
            decoder_input_ids, self.decoder_model_config.remove_input_padding,
            pad_token_id)

        # `cross_attention_mask` in context phase [batch_size, query_len, encoder_input_len]
        # where query_len happens to be 1 in current cases, but not necessarily always, and
        # `cross_attention_mask` in generation phase [batch_size, 1, encoder_input_len] where
        # the query_len is always 1 since we have kv cache.
        cross_attention_mask = None
        if attention_mask is not None:
            cross_attention_mask = torch.tensor(attention_mask,
                                                dtype=torch.int32,
                                                device=self.device).reshape(
                                                    attention_mask.shape[0], 1,
                                                    attention_mask.shape[1])

        # generation config
        sampling_config = SamplingConfig(end_id=eos_token_id,
                                         pad_id=pad_token_id,
                                         num_beams=num_beams,
                                         min_length=1,
                                         return_dict=return_dict)
        sampling_config.update(output_cum_log_probs=return_dict,
                               output_log_probs=return_dict)

        # decoder autoregressive generation
        self.decoder_session.setup(
            decoder_input_lengths.size(0),
            decoder_max_input_length,
            max_new_tokens,
            num_beams,
            max_attention_window_size=None,
            encoder_max_input_length=encoder_max_input_length,
            lora_manager=self.decoder_lora_manager,
            lora_uids=self.lora_task_uids,
        )

        output = self.decoder_session.decode(
            decoder_input_ids,
            decoder_input_lengths,
            sampling_config,
            encoder_output=encoder_output,
            encoder_input_lengths=encoder_input_lengths,
            return_dict=return_dict,
            cross_attention_mask=cross_attention_mask)

        if return_dict and return_encoder_output:
            output['encoder_output'] = encoder_output

        return output
