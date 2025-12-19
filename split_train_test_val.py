# split_train_val_test.py
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------
# 参数设置
# ---------------------------
INPUT_FILE = "user_dna_all.csv"  # 之前生成的全量 DNA
TRAIN_FILE = "user_dna_train.csv"
VAL_FILE   = "user_dna_val.csv"
TEST_FILE  = "user_dna_test.csv"

TEST_SIZE = 0.2   # 20% 测试集
VAL_SIZE  = 0.1   # 10% 验证集（相对原始数据）
RANDOM_STATE = 42 # 保证可复现

# ---------------------------
# 读取数据
# ---------------------------
df = pd.read_csv(INPUT_FILE)
print(f"Total users: {len(df)}")
print(df['label'].value_counts())

# ---------------------------
# 先划分训练集 + 测试集
# ---------------------------
train_val_df, test_df = train_test_split(
    df,
    test_size=TEST_SIZE,
    stratify=df['label'],
    random_state=RANDOM_STATE
)

# ---------------------------
# 再从训练集中划分验证集
# ---------------------------
val_relative_size = VAL_SIZE / (1 - TEST_SIZE)  # 相对于 train_val_df 的比例
train_df, val_df = train_test_split(
    train_val_df,
    test_size=val_relative_size,
    stratify=train_val_df['label'],
    random_state=RANDOM_STATE
)

# ---------------------------
# 保存结果
# ---------------------------
train_df.to_csv(TRAIN_FILE, index=False)
val_df.to_csv(VAL_FILE, index=False)
test_df.to_csv(TEST_FILE, index=False)

print(f"Train set: {len(train_df)} users -> saved to {TRAIN_FILE}")
print(f"Validation set: {len(val_df)} users -> saved to {VAL_FILE}")
print(f"Test set: {len(test_df)} users -> saved to {TEST_FILE}")
