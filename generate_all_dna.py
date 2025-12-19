# generate_all_dna.py
import pandas as pd
import zipfile
from pathlib import Path

# ---------------------------
# 编码单条 tweet
# ---------------------------
def encode_tweet(row):
    """
    将每条 tweet 编码为一个字符
    T: 普通 tweet
    R: retweet
    C: reply
    """
    if row.get('retweeted_status_id', 0) != 0:
        return 'R'
    elif row.get('in_reply_to_status_id', 0) != 0:
        return 'C'
    else:
        return 'T'

# ---------------------------
# 生成用户 DNA
# ---------------------------
def generate_dna(df):
    dna_dict = {}
    for user_id, group in df.groupby('user_id'):
        group_sorted = group.sort_values('created_at')
        dna_sequence = ''.join(group_sorted.apply(encode_tweet, axis=1))
        dna_dict[user_id] = dna_sequence
    return pd.DataFrame({"user_id": list(dna_dict.keys()), "dna_sequence": list(dna_dict.values())})

# ---------------------------
# 处理单个用户类别
# ---------------------------
def process_user_type(data_dir, user_type):
    zip_path = Path(data_dir) / f"{user_type}.csv.zip"
    tweets_path = f"{user_type}.csv/tweets.csv"

    print(f"\nLoading tweets from {zip_path} ...")
    with zipfile.ZipFile(zip_path) as z:
        if tweets_path not in z.namelist():
            raise FileNotFoundError(f"{tweets_path} not found in {zip_path}")
        df_tweets = pd.read_csv(z.open(tweets_path), encoding="latin1", low_memory=False)

    print(f"Tweets loaded: {len(df_tweets)} rows")
    df_dna = generate_dna(df_tweets)

    # 设置 label
    label = 0 if "genuine" in user_type else 1
    df_dna['label'] = label

    print(f"Digital DNA saved: {len(df_dna)} users")
    return df_dna


if __name__ == "__main__":
    DATA_DIR = "data/datasets_full.csv"  # 修改为你的数据目录
    USER_TYPES = [
        "genuine_accounts",
        "social_spambots_1",
        "social_spambots_2",
        "social_spambots_3"
    ]

    all_dna = []

    for user_type in USER_TYPES:
        df = process_user_type(DATA_DIR, user_type)
        all_dna.append(df)

    # 合并所有 DNA
    df_all = pd.concat(all_dna, ignore_index=True)
    df_all.to_csv("user_dna_all.csv", index=False)
    print(f"\nAll DNA merged, total {len(df_all)} users saved to user_dna_all.csv")
