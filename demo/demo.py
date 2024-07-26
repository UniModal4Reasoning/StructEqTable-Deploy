import time
import torch
import argparse

from PIL import Image
from struct_eqtable import build_model


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--image_path', type=str, default='demo.png', help='data path for table image')
    parser.add_argument('--ckpt_path', type=str, default='U4R/StructTable-base', help='ckpt path for table model, which can be downloaded from huggingface')
    parser.add_argument('--cpu', action='store_true', default=False, help='using cpu for inference')
    parser.add_argument('--html', action='store_true', default=False, help='output html format table code')
    args = parser.parse_args()
    return args

def main():
    args = parse_config()
    if args.html:
        from pypandoc import convert_text

    # build model
    model = build_model(args.ckpt_path, max_new_tokens=4096, max_time=60)
    if not args.cpu:
        model = model.cuda()

    # model inference
    raw_image = Image.open(args.image_path)
    
    start_time = time.time()
    with torch.no_grad():
        output = model(raw_image)

    # show output latex code of table
    cost_time = time.time() - start_time
    print(f"total cost time: {cost_time:.2f}s")
    for i, latex_code in enumerate(output):
        if args.html:
            html_code = convert_text(latex_code, 'html', format='latex')
            print(f"Table {i} HTML code:\n{html_code}")
        else:
            print(f"Table {i} LaTex code:\n{latex_code}")


if __name__ == '__main__':
    main()
