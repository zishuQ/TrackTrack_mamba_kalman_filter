import os
from utils.gbi import linear_interpolation_only

# 输入和输出文件夹
result_folder = "/home/shang/workspace/TrackTrack/outputs/3. track/sportsmot_test_0.80_mamba"
output_folder = result_folder + "_post"
os.makedirs(output_folder, exist_ok=True)

for result_file in os.listdir(result_folder):
    path_in = os.path.join(result_folder, result_file)
    path_out = os.path.join(output_folder, result_file)
    linear_interpolation_only(path_in, path_out, n_min=5, n_dti=20)
    print(f"Processed {result_file}")

print("All files processed.")
