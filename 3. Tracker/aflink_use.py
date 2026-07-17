import os
import torch

# Import AFLink utilities (same as used in run_mamba.py)
from AFLink.AppFreeLink import *
from AFLink.model import PostLinker
from AFLink.dataset import LinkData


result_folder = '/home/shang/workspace/TrackTrack/outputs/3. track/sportsmot_test_0.80_sports_s4_e200_iwg_attn_final_post'
output_folder = result_folder + '_aflink_post'
os.makedirs(output_folder, exist_ok=True)

# Load model once and reuse if possible
# Note: run_mamba created model per-file; here load once to be efficient.
state = torch.load('./AFLink/AFLink_epoch20.pth', map_location='cuda')
model = PostLinker()
model.load_state_dict(state)

aflink_dataset = LinkData('', '')

for result_file in sorted(os.listdir(result_folder)):
    path_in = os.path.join(result_folder, result_file)
    path_out = os.path.join(output_folder, result_file)

    linker = AFLink(path_in=path_in, path_out=path_out, model=model, dataset=aflink_dataset,
                    thrT=(0, 20), thrS=100, thrP=0.05)
    linker.link()
    print(f"Processed {result_file}")

print("All files processed.")
