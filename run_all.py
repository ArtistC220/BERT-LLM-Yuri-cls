import subprocess
import sys
import os

# 项目根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(BASE_DIR, "script")

# 全流程脚本（不含 update.py）
steps = [
    "clean_txt.py",
    "model_BERT_infer.py",
    "dialogue_cut.py",
    "drop_dialogue.py",
    "verb_cut.py",
    "LLM_dialogue.py",
    "LLM_verb.py",
    "verb_normalizer.New.py",
    "dialogue_normalizer.New.py",
    "weighted_fun.py",
    "merge.py",
    "book_result_match.py",
    "sort_book.py",
    "volume_result_match.py",
    "sort_volume.py",
    "result_volume_merge.py",
    "result_book_merge.py",
]

def run_step(step):
    """运行单个脚本"""
    cmd = f"{sys.executable} {os.path.join(SCRIPTS_DIR, step)}"
    print(f"\n Running: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"Error，停止在 {step}")
        sys.exit(1)

if __name__ == "__main__":
    mode = input("请输入模式（run 或 update）：").strip().lower()

    if mode == "run":
        for step in steps:
            run_step(step)
        print("\n 全部步骤完成！（不含 update.py）")

    elif mode == "update":
        run_step("update.py")
        print("\n update.py 执行完成！")

    else:
        print("无效输入，请输入 run 或 update(清理缓存)")
