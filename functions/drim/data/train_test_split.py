"""
File to split certain dataset.
Made by Fabian van Stijn
"""

import os
import shutil
from sklearn.model_selection import train_test_split


source_dirs = ['//home/rjreitsma/Documents/DRIM_CardiacMRI/To reconstruct/mouse/15/', '//home/rjreitsma/Documents/DRIM_CardiacMRI/To reconstruct/mouse/25/',
               '//home/rjreitsma/Documents/DRIM_CardiacMRI/To reconstruct/mouse/30/', '//home/rjreitsma/Documents/DRIM_CardiacMRI/To reconstruct/mouse/50/',
               '//home/rjreitsma/Documents/DRIM_CardiacMRI/To reconstruct/mouse/100/']

train_dirs = ['//home/rjreitsma/Documents/DRIM_CardiacMRI/To reconstruct/mouse/15/train', '//home/rjreitsma/Documents/DRIM_CardiacMRI/To reconstruct/mouse/25/train',
               '//home/rjreitsma/Documents/DRIM_CardiacMRI/To reconstruct/mouse/30/train', '//home/rjreitsma/Documents/DRIM_CardiacMRI/To reconstruct/mouse/50/train',
               '//home/rjreitsma/Documents/DRIM_CardiacMRI/To reconstruct/mouse/100/train']

test_dirs = ['//home/rjreitsma/Documents/DRIM_CardiacMRI/To reconstruct/mouse/15/test', '//home/rjreitsma/Documents/DRIM_CardiacMRI/To reconstruct/mouse/25/test',
               '//home/rjreitsma/Documents/DRIM_CardiacMRI/To reconstruct/mouse/30/test', '//home/rjreitsma/Documents/DRIM_CardiacMRI/To reconstruct/mouse/50/test',
               '//home/rjreitsma/Documents/DRIM_CardiacMRI/To reconstruct/mouse/100/test']

for train_dir in train_dirs:
    os.makedirs(train_dir, exist_ok=True)
for test_dir in test_dirs:
    os.makedirs(test_dir, exist_ok=True)

all_files = os.listdir(source_dirs[0]) + os.listdir(source_dirs[1]) + os.listdir(source_dirs[2]) + os.listdir(source_dirs[3]) + os.listdir(source_dirs[4])

files_dict = {}
for filename in all_files:
    base_name = filename.replace('.mat', '')[:-2]
    if base_name not in files_dict:
        if 'retro' in filename:
            files_dict[base_name] = []
    if 'retro' in filename:
        files_dict[base_name].append(filename)

combined_base_names = list(files_dict.keys())
train_base_names, test_base_names = train_test_split(combined_base_names, test_size=0.2, random_state=42)

def move_files(base_names, source_dirs, dest_dirs):
    for base_name in base_names: 
        print(base_name)
        for source_dir, dest_dir in zip(source_dirs, dest_dirs):
            print(source_dir, dest_dir)
            for filename in os.listdir(source_dir):
                print(filename, base_name)
                if filename.startswith(base_name):
                    shutil.move(os.path.join(source_dir, filename), os.path.join(dest_dir, filename))
                    print('move')

move_files(train_base_names, source_dirs, train_dirs)
move_files(test_base_names, source_dirs, test_dirs)