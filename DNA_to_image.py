# generate_dna_images.py
import pandas as pd
import numpy as np
import os

# ---------------------------
# 参数设置
# ---------------------------
DATA_FILES = {
    'train': 'user_dna_train.csv',
    'val': 'user_dna_val.csv',
    'test': 'user_dna_test.csv'
}
OUTPUT_DIR = 'dna_images'
IMG_HEIGHT = 32
IMG_WIDTH  = 32
CHAR_MAP = {'T': 0, 'R': 1, 'C': 2}

os.makedirs(OUTPUT_DIR, exist_ok=True)

def dna_to_image(dna_sequence, height=IMG_HEIGHT, width=IMG_WIDTH):
    dna_numeric = [CHAR_MAP.get(c,0) for c in dna_sequence]
    if len(dna_numeric) < height*width:
        dna_numeric += [0]*(height*width - len(dna_numeric))
    else:
        dna_numeric = dna_numeric[:height*width]
    img = np.array(dna_numeric).reshape((height, width))
    return img

for split, file in DATA_FILES.items():
    df = pd.read_csv(file)
    images = np.array([dna_to_image(seq) for seq in df['dna_sequence']])
    labels = df['label'].values
    # 保存 numpy 文件
    np.save(os.path.join(OUTPUT_DIR, f'{split}_images.npy'), images)
    np.save(os.path.join(OUTPUT_DIR, f'{split}_labels.npy'), labels)
    print(f'{split}: {images.shape[0]} images saved')
