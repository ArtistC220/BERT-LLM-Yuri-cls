import subprocess
import sys
import os
import json

# 项目根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(BASE_DIR, "script")
CHECKPOINT_FILE = os.path.join(BASE_DIR, ".run_checkpoint.json")

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


def load_checkpoint():
    """加载断点记录"""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {"completed": [], "last_failed": None}


def save_checkpoint(completed, last_failed=None):
    """保存断点记录"""
    data = {"completed": completed, "last_failed": last_failed}
    with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def clear_checkpoint():
    """清除断点记录"""
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        print("已清除断点记录")


def run_step(step):
    """运行单个脚本"""
    cmd = f"{sys.executable} {os.path.join(SCRIPTS_DIR, step)}"
    print(f"\n Running: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"Error，停止在 {step}")
        return False
    return True


def print_progress(completed, total, current=None):
    """打印进度"""
    print(f"\n{'='*50}")
    print(f"进度: {len(completed)}/{total}")
    if completed:
        print(f"已完成: {', '.join(completed[-3:])}{'...' if len(completed) > 3 else ''}")
    if current:
        print(f"当前: {current}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    print("请选择模式：")
    print("  run    - 从头开始运行（会清除之前的断点）")
    print("  resume - 从断点继续运行")
    print("  update - 清理缓存")
    
    mode = input("\n请输入模式: ").strip().lower()

    if mode == "run":
        # 从头开始，清除断点
        clear_checkpoint()
        completed = []
        
        for i, step in enumerate(steps):
            print_progress(completed, len(steps), step)
            
            if not run_step(step):
                save_checkpoint(completed, step)
                sys.exit(1)
            
            completed.append(step)
            save_checkpoint(completed)
        
        clear_checkpoint()
        print("\n 全部步骤完成！")

    elif mode == "resume":
        # 从断点继续
        checkpoint = load_checkpoint()
        completed = checkpoint.get("completed", [])
        last_failed = checkpoint.get("last_failed")
        
        if not completed and not last_failed:
            print("没有找到断点记录，将从头开始运行...")
            completed = []
        else:
            print(f"找到断点记录，已完成 {len(completed)} 个步骤")
            if last_failed:
                print(f"上次失败在: {last_failed}")
            print(f"将从 {steps[len(completed)] if len(completed) < len(steps) else '无剩余步骤'} 继续\n")
        
        # 从断点处开始执行
        start_idx = len(completed)
        for i in range(start_idx, len(steps)):
            step = steps[i]
            print_progress(completed, len(steps), step)
            
            if not run_step(step):
                save_checkpoint(completed, step)
                sys.exit(1)
            
            completed.append(step)
            save_checkpoint(completed)
        
        clear_checkpoint()
        print("\n 全部步骤完成！")

    elif mode == "update":
        run_step("update.py")
        # update 后清除断点，因为缓存被清理了
        clear_checkpoint()
        print("\n update.py 执行完成！断点记录已清除")

    else:
        print("无效输入，请输入 run、resume 或 update")
