import time
import torch
import argparse

from PIL import Image
from struct_eqtable import build_model


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--image_path', type=str, default='demo.png', help='data path for table image')
    parser.add_argument('--ckpt_path', type=str, default='U4R/StructTable-base', help='ckpt path for table model, which can be downloaded from huggingface')
    parser.add_argument('--max_new_tokens', type=int, default=2048, help='maximum output tokens of model inference')
    parser.add_argument('-t', '--max_waiting_time', type=int, default=60, help='maximum waiting time of model inference')
    parser.add_argument('-f', '--output_format', type=str, nargs='+', default=['latex'], 
                        help='The model outputs LaTeX format code by default. Simple structured table LaTeX code can be converted to HTML or Markdown format using pypandoc.')
    parser.add_argument('--tensorrt_path', type=str, default=None, help='enable tensorrt for model acceleration')
    parser.add_argument('--lmdeploy', action='store_true', help='use lmdepoly to accelerate model inference')
    args = parser.parse_args()
    return args

def main():
    args = parse_config()

    # build model
    model = build_model(
        args.ckpt_path, 
        max_new_tokens=args.max_new_tokens, 
        max_time=args.max_waiting_time,
        tensorrt_path=args.tensorrt_path,
        lmdeploy=args.lmdeploy
    )

    assert torch.cuda.is_available(), "Our model current only support with gpu"
    if not args.tensorrt_path:
        model = model.cuda()

    # process output format
    output_formats = list(set(args.output_format) & set(model.supported_output_format))
    print(f"Supported output format: {' '.join(output_formats)}")

    # model inference
    raw_image = Image.open(args.image_path)

    output_list = []
    start_time = time.time()

    with torch.no_grad():
        for tgt_fmt in output_formats:
            output = model(raw_image, output_format=tgt_fmt)
            output_list.append(output)

    # show output latex code of table
    cost_time = time.time() - start_time
    print(f"total cost time: {cost_time:.2f}s")

    if cost_time >= args.max_waiting_time:
        warn_log = f"\033[93mThe model inference time exceeds the maximum waiting time {args.max_waiting_time} seconds, the result may be incomplete.\n" \
        "Please increase the maximum waiting time with argument --max_waiting_time or Model may not support the type of input table image \033[0m"
        print(warn_log)

    for i, tgt_fmt in enumerate(output_formats):
        for j, output in enumerate(output_list[i]):
            print(f"Table {j} {tgt_fmt.upper()} format output:\n{output}")


if __name__ == '__main__':
    main()
