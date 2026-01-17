import os
import shutil


def clear_directory_contents(directory_path):
    """
    删除指定目录下的所有文件和子目录，但保留目录本身
    """
    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)

        if os.path.isfile(item_path) or os.path.islink(item_path):
            # 删除文件或符号链接
            os.remove(item_path)
        elif os.path.isdir(item_path):
            # 删除整个子目录
            shutil.rmtree(item_path)
