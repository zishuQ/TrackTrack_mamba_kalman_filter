import argparse
import os

import torch

# Import AFLink utilities (same as used in run_mamba.py)
from AFLink.AppFreeLink import *
from AFLink.model import PostLinker
from AFLink.dataset import LinkData


def main():
    parser = argparse.ArgumentParser(description="Run AFLink over a post-processing directory.")
    parser.add_argument("--input", required=True, help="Directory with per-sequence post .txt files.")
    parser.add_argument("--output", default=None,
                        help="Output directory; default is '<input>_aflink_post'.")
    parser.add_argument("--device", default="cuda", help="Device used when loading the AFLink weights.")
    args = parser.parse_args()

    result_folder = args.input
    output_folder = args.output or (result_folder + '_aflink_post')
    os.makedirs(output_folder, exist_ok=True)

    # Load model once and reuse if possible
    # Note: run_mamba created model per-file; here load once to be efficient.
    state = torch.load('./AFLink/AFLink_epoch20.pth', map_location=args.device)
    model = PostLinker()
    model.load_state_dict(state)

    aflink_dataset = LinkData('', '')

    files = sorted(name for name in os.listdir(result_folder) if name.endswith('.txt'))
    for result_file in files:
        path_in = os.path.join(result_folder, result_file)
        path_out = os.path.join(output_folder, result_file)

        linker = AFLink(path_in=path_in, path_out=path_out, model=model, dataset=aflink_dataset,
                        thrT=(0, 20), thrS=100, thrP=0.05)
        linker.link()
        print(f"Processed {result_file}")

    print(f"All {len(files)} files processed.")


if __name__ == "__main__":
    main()
