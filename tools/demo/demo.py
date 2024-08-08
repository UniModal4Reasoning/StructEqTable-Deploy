import time
import torch
import argparse

from PIL import Image
from struct_eqtable import build_model


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--image_path', type=str, default='demo.png', help='data path for table image')
    parser.add_argument('--ckpt_path', type=str, default='U4R/StructTable-base', help='ckpt path for table model, which can be downloaded from huggingface')
    parser.add_argument('-t', '--max_waiting_time', type=int, default=60, help='maximum waiting time of model inference')
    parser.add_argument('--cpu', action='store_true', default=False, help='using cpu for inference')
    parser.add_argument('-f', '--output_format', type=str, nargs='+', default=['latex'], 
                        help='The model outputs LaTeX format code by default. Simple structured table LaTeX code can be converted to HTML or Markdown format using pypandoc.')
    parser.add_argument('--tensorrt_path', type=str, default=None, help='enable tensorrt for model acceleration')
    args = parser.parse_args()
    return args

def main():
    args = parse_config()
    if 'html' in args.output_format or 'markdown' in args.output_format:
        from pypandoc import convert_text

    # build model
    model = build_model(
        args.ckpt_path, 
        max_new_tokens=4096, 
        max_time=args.max_waiting_time,
        tensorrt_path=args.tensorrt_path
    )
    if not args.cpu and args.tensorrt_path is None:
        model = model.cuda()

    # model inference
    raw_image = Image.open(args.image_path)
    
    start_time = time.time()
    with torch.no_grad():
        output = model(raw_image)

    # show output latex code of table
    cost_time = time.time() - start_time
    print(f"total cost time: {cost_time:.2f}s")

    if cost_time >= args.max_waiting_time:
        warn_log = f"\033[93mThe model inference time exceeds the maximum waiting time {args.max_waiting_time} seconds, the result may be incomplete.\n" \
        "Please increase the maximum waiting time with argument --max_waiting_time or Model may not support the type of input table image \033[0m"
        print(warn_log)

    
    for i, latex_code in enumerate(output):
        for tgt_fmt in args.output_format:
            tgt_code = convert_text(latex_code, tgt_fmt, format='latex') if tgt_fmt != 'latex' else latex_code
            print(f"Table {i} {tgt_fmt.upper()} format output:\n{tgt_code}")


if __name__ == '__main__':
    main()
