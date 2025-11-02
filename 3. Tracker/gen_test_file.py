import os
import shutil

trackers = ['mot17_test_0.80_post', 'mot20_test_0.80_post']


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

        dummy_path = './utils/mot17_dummy/'
        if os.path.exists(dummy_path):
            dummy_files = os.listdir(dummy_path)
            for dummy_file in dummy_files:
                shutil.copy(dummy_path + dummy_file, path + dummy_file)

    if 'mot20' in tracker:
        dummy_path = './utils/mot20_dummy/'
        if os.path.exists(dummy_path):
            dummy_files = os.listdir(dummy_path)
            for dummy_file in dummy_files:
                shutil.copy(dummy_path + dummy_file, path.replace('mot17', 'mot20') + dummy_file)
