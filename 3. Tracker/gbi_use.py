import os
from utils.gbi import gb_interpolation

# 输入和输出文件夹
result_folder = "/home/shang/workspace/TrackTrack/outputs/3. track/mot17_all_0.80_iwg_v3_oldconfig_trainall_trainonly_seed42_agentguard_iwg"
output_folder = result_folder + "_post"
os.makedirs(output_folder, exist_ok=True)

for result_file in os.listdir(result_folder):
    path_in = os.path.join(result_folder, result_file)
    path_out = os.path.join(output_folder, result_file)
    gb_interpolation(path_in, path_out, interval=30, tau=12)
    print(f"Processed {result_file}")

print("All files processed.")
