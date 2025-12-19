# evaluate_model.py
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# ---------------------------
# 参数设置
# ---------------------------
MODEL_FILE = "dna_cnn_model.h5"
DATA_DIR   = "dna_images"
OUTPUT_CSV = "test_metrics.csv"
ROC_PNG    = "roc_curve.png"

# ---------------------------
# 加载模型
# ---------------------------
model = load_model(MODEL_FILE)

# ---------------------------
# 加载测试集
# ---------------------------
X_test = np.load(f'{DATA_DIR}/test_images.npy')
y_test = np.load(f'{DATA_DIR}/test_labels.npy')

# 如果是灰度图需要 reshape
if len(X_test.shape) == 3:
    X_test = X_test.reshape((-1, X_test.shape[1], X_test.shape[2], 1))

# ---------------------------
# 预测
# ---------------------------
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)

# ---------------------------
# 计算指标
# ---------------------------
acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)
roc  = roc_auc_score(y_test, y_pred_prob[:,1])

metrics_df = pd.DataFrame([{
    'accuracy': acc,
    'precision': prec,
    'recall': rec,
    'f1_score': f1,
    'roc_auc': roc
}])

metrics_df.to_csv(OUTPUT_CSV, index=False)
print(f"Test metrics saved to {OUTPUT_CSV}")
print(metrics_df)

# ---------------------------
# 绘制 ROC 曲线
# ---------------------------
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob[:,1])
plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f'ROC AUC = {roc:.3f}')
plt.plot([0,1],[0,1],'--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.savefig(ROC_PNG, dpi=300)
print(f"ROC curve saved as {ROC_PNG}")
plt.show()
