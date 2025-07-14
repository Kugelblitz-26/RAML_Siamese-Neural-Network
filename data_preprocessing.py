
import os
import shutil
import random
from tqdm import tqdm

src_dir = "/home/charanz/Desktop/MASTERS/RAML/ref/archive/lfw-funneled/lfw_funneled"
dst_dir = "/home/charanz/Desktop/MASTERS/RAML/ref/archive/lfw-funneled/extracted"
os.makedirs(dst_dir, exist_ok=True)

all_classes = sorted(os.listdir(src_dir))
random.seed(42)
random.shuffle(all_classes)

n = len(all_classes)
splits = {
    'train': all_classes[:int(0.8*n)],
    'valid': all_classes[int(0.8*n):int(0.9*n)],
    'test':  all_classes[int(0.9*n):]
}

for split, class_list in splits.items():
    for cls in tqdm(class_list, desc=f"Copying {split}"):
        src = os.path.join(src_dir, cls)
        dst = os.path.join(dst_dir, split, cls)
        os.makedirs(dst, exist_ok=True)
        for file in os.listdir(src):
            shutil.copy(os.path.join(src, file), os.path.join(dst, file))
