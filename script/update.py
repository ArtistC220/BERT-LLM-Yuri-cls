import os
import shutil
from utils import load_config
config = load_config()

def clean_folder_recursive(root_path, keep_files=None, keep_dirs=None):
    """
    递归清理指定文件夹及其子文件夹中的文件（保留文件夹结构）
    - keep_files: 不删除的文件名列表
    - keep_dirs: 不清理的子文件夹名列表（完全匹配文件夹名）
    """
    if keep_files is None:
        keep_files = []
    if keep_dirs is None:
        keep_dirs = []

    if not os.path.exists(root_path):
        print(f"文件夹不存在: {root_path}")
        return

    for current_path, dirs, files in os.walk(root_path):
        # 修改 dirs 列表，去掉需要跳过的子文件夹
        dirs[:] = [d for d in dirs if d not in keep_dirs]

        for file in files:
            if file in keep_files:  # 保留的文件
                continue
            file_path = os.path.join(current_path, file)
            try:
                os.remove(file_path)
                print(f"删除文件: {file_path}")
            except Exception as e:
                print(f"删除失败: {file_path}, 错误: {e}")

    print(f"已清理完成: {root_path}")


def move_folder_files(src_folder, dst_folder):
    """
    将 src_folder 下的所有文件搬运到 dst_folder （不处理子文件夹）
    """
    if not os.path.exists(src_folder):
        print(f"源文件夹不存在: {src_folder}")
        return

    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)  # 目标不存在就创建

    for file in os.listdir(src_folder):
        src_path = os.path.join(src_folder, file)
        dst_path = os.path.join(dst_folder, file)

        if os.path.isfile(src_path):  # 只处理文件
            try:
                shutil.move(src_path, dst_path)
                print(f"搬运: {src_path} -> {dst_path}")
            except Exception as e:
                print(f"搬运失败: {src_path}, 错误: {e}")

    print(f"搬运完成: {src_folder} -> {dst_folder}")


# 
if __name__ == "__main__":
    path_1 = os.path.join(os.path.dirname(os.path.dirname(__file__)), "csv")
    path_2 = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")

    target_folders = [path_1, path_2]

    keep = ["ph.md","stopwords.txt", "verbword.txt", "baihe_aid_title_cover_local.csv", "filename_realvolume.csv" , "u-requirements.txt"]
    keep_subfolders = ["history_rank_volume", "history_rank_book","result01","result02","txt_cleaned","txt_val_cleaned1","txt_train_cleaned1","normalizerCSV"]  

    for folder in target_folders:
        clean_folder_recursive(folder, keep_files=keep, keep_dirs=keep_subfolders)

    # 搬运示例（可以改成想搬的文件夹路径）
    src = os.path.join(os.path.dirname(os.path.dirname(__file__)), "txt_test")
    dst = os.path.join(os.path.dirname(os.path.dirname(__file__)), "txt_cache")

    move_folder_files(src, dst)
