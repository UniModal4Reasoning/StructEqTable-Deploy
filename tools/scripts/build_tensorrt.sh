set -x 

HF_CKPT_PATH=${1:-"../ckpts/StructTable-base"}
MODEL_OUTPUT=${2:-"../ckpts/StructTable-base-TensorRT"}
MODEL_TYPE=${3:-"StructEqTable"}

if [ ! -d $MODEL_OUTPUT ]; then
    mkdir -p $MODEL_OUTPUT
fi

# Step1 Convert the model into TensorrtLLM checkpoint format
echo "Step1 Convert the model into TensorrtLLM checkpoint format"

python tensorrt_utils/convert_checkpoint.py --model_type $MODEL_TYPE \
    --model_dir $HF_CKPT_PATH \
    --output_dir $MODEL_OUTPUT/trt_models/float16 \
    --tp_size 1 \
    --pp_size 1 \
    --workers 1 \
    --dtype float16

# Step2 Compile the model
echo "Step2 build LLM Engine"

trtllm-build --checkpoint_dir $MODEL_OUTPUT/trt_models/float16/decoder \
    --output_dir $MODEL_OUTPUT/llm_engines/decoder \
    --paged_kv_cache disable \
    --moe_plugin disable \
    --enable_xqa disable \
    --use_custom_all_reduce disable \
    --gemm_plugin float16 \
    --bert_attention_plugin float16 \
    --gpt_attention_plugin float16 \
    --remove_input_padding enable \
    --context_fmha disable \
    --max_beam_width 1 \
    --max_batch_size 8 \
    --max_seq_len 4096 \
    --max_encoder_input_len 4096 \
    --max_input_len 1

# Step3 build visual engine
echo "Step3 Build Visual Engine"

python tensorrt_utils/build_visual_engine.py --model_type $MODEL_TYPE \
    --model_path $HF_CKPT_PATH \
    --output_dir $MODEL_OUTPUT/visual_engines \
    --max_batch_size 1

if [ -f './model.cache' ]; then
    rm ./model.cache
fi

echo "Build TensorRT model and Visual Engine Successfully"