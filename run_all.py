# run_all.py
"""
主控脚本：按顺序运行整个僵尸用户检测项目流程
流程：
1. 生成数字 DNA 图像
2. 划分训练集、验证集和测试集
3. 使用 CNN 训练模型
4. 使用测试集评估模型并生成效果表格和 ROC 曲线
"""

import subprocess
import sys

def run_script(script_name, description=""):
    """
    调用指定 Python 脚本并输出状态
    
    参数:
        script_name (str): 脚本文件名
        description (str): 脚本功能说明，用于打印
    """
    print(f"\n===== Running: {script_name} =====")
    if description:
        print(f"Description: {description}\n")
    
    # 调用 Python 脚本
    result = subprocess.run([sys.executable, script_name], capture_output=True, text=True)
    
    # 输出标准输出
    print(result.stdout)
    
    # 如果执行失败，打印错误并退出
    if result.returncode != 0:
        print(result.stderr)
        print(f"Error running {script_name}. Exiting.")
        sys.exit(1)
    
    print(f"===== Finished: {script_name} =====\n")

if __name__ == "__main__":
    # 按顺序定义所有脚本
    scripts = [
        {
            "file": "generate_all_dna.py",
            "desc": "读取数据，生成所有数字 DNA"
        },
        {
            "file": "DNA_to_image.py",
            "desc": "将 DNA 序列转换为图像并保存"
        },
        {
            "file": "split_train_val_test.py",
            "desc": "划分训练集、验证集和测试集，并保存对应 .npy 文件"
        },
        {
            "file": "train_cnn.py",
            "desc": "使用训练集训练 CNN 模型，同时用验证集 EarlyStopping"
        },
        {
            "file": "evaluate_model.py",
            "desc": "使用测试集评估模型，生成指标表格 CSV 和 ROC 曲线 PNG"
        }
    ]
    
    # 依次运行每个脚本
    for script in scripts:
        run_script(script["file"], script["desc"])
    
    print("All scripts ran successfully! ✅")
