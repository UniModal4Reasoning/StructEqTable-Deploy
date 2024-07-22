import torch
import argparse

from PIL import Image
from struct_eqtable import build_model


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--image_path', type=str, default='demo.png', help='data path for table image')
    parser.add_argument('--ckpt_path', type=str, default='', help='ckpt path for table model')
    args = parser.parse_args()
    return args

def main():
    args = parse_config()

    # build model
    model = build_model(args.ckpt_path, max_new_tokens=4096, max_time=120)

    # model inference
    raw_image = Image.open(args.image_path)
    with torch.no_grad():
        output = model(raw_image)

    # show output latex code of table
    for i, latex_code in enumerate(output):
        print(f"Table {i}:\n{latex_code}")

if __name__ == '__main__':
    main()
