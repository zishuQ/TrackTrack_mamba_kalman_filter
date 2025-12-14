import os
import shutil

trackers = ['mot17_test_0.80_mamba']


for tracker in trackers:
    path = '../outputs/3. track/' + tracker + '/'
    files = os.listdir(path)

    if 'mot17' in tracker:
        for file in files:
            # 只处理FRCNN文件，避免重复复制
            if 'FRCNN' in file:
                file_path = path + file
                shutil.copy(file_path, file_path.replace('FRCNN', 'SDP'))
                shutil.copy(file_path, file_path.replace('FRCNN', 'DPM'))
