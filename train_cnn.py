import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

# ---------------------------
# 参数
# ---------------------------
IMG_HEIGHT = 32
IMG_WIDTH  = 32
CHANNELS   = 1
MODEL_FILE = "dna_cnn_model.h5"
DATA_DIR   = "dna_images"
METRICS_CSV = "training_metrics.csv"

# ---------------------------
# 加载数据
# ---------------------------
X_train = np.load(f'{DATA_DIR}/train_images.npy').reshape(-1, IMG_HEIGHT, IMG_WIDTH, CHANNELS)
y_train = to_categorical(np.load(f'{DATA_DIR}/train_labels.npy'), num_classes=2)

X_val   = np.load(f'{DATA_DIR}/val_images.npy').reshape(-1, IMG_HEIGHT, IMG_WIDTH, CHANNELS)
y_val   = to_categorical(np.load(f'{DATA_DIR}/val_labels.npy'), num_classes=2)

# ---------------------------
# 构建CNN模型
# ---------------------------
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ---------------------------
# 回调函数
# ---------------------------
callbacks = [
    ModelCheckpoint(MODEL_FILE, save_best_only=True, monitor='val_accuracy', mode='max'),
    EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
]

# ---------------------------
# 训练
# ---------------------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=64,
    callbacks=callbacks
)

# ---------------------------
# 保存训练曲线数值到 CSV
# ---------------------------
metrics_df = pd.DataFrame({
    'epoch': range(1, len(history.history['accuracy'])+1),
    'train_loss': history.history['loss'],
    'train_accuracy': history.history['accuracy'],
    'val_loss': history.history['val_loss'],
    'val_accuracy': history.history['val_accuracy']
})
metrics_df.to_csv(METRICS_CSV, index=False)
print(f"Training metrics saved to {METRICS_CSV}")

# ---------------------------
# 绘制训练曲线并标注每个点
# ---------------------------
plt.figure(figsize=(10,5))
plt.plot(metrics_df['epoch'], metrics_df['train_accuracy'], label='Train Acc', marker='o')
plt.plot(metrics_df['epoch'], metrics_df['val_accuracy'], label='Val Acc', marker='s')

# 标注每个点的数值
for i, (t_acc, v_acc) in enumerate(zip(metrics_df['train_accuracy'], metrics_df['val_accuracy'])):
    plt.text(i+1, t_acc+0.01, f"{t_acc:.3f}", ha='center', fontsize=8)
    plt.text(i+1, v_acc-0.02, f"{v_acc:.3f}", ha='center', fontsize=8)

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid(True)

# 保存训练曲线图
plt.savefig("training_curve.png", dpi=300)
print("Training curve saved as training_curve.png")

plt.show()
