import logging
import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from common import config
from common.config import DEFAULT_TABLE
from indexer import index
from model import get_feature_extractor

# 全局锁，确保 model.predict 串行执行，避免 GPU OOM
predict_lock = threading.Lock()
# 全局map
cache_map = {}


def do_train(table_name, data_path, embedding_index_type):
    # 创建线程池
    executor = ThreadPoolExecutor(max_workers=config.MAX_THREADS)
    """
    Train the model by indexing images to Milvus
    """
    cache_map.clear()
    try:
        # Create table if not exists
        if index.has_collection(table_name or DEFAULT_TABLE) is False:
            # 创建表
            with predict_lock:
                index.create_table(table_name or config.DEFAULT_TABLE,
                                   False,
                                   embedding_index_type or config.EMBEDDING_INDEX_TYPE)

        # 先把图片copy到DATA_PATH下的一个当前时间戳的目录下
        timestamp = str(int(round(time.time() * 1000)))
        DATA_PATH_SUBDIR = os.path.join(config.DATA_PATH, timestamp)
        if not os.path.exists(DATA_PATH_SUBDIR):
            # 创建目录，如果不存在
            os.makedirs(DATA_PATH_SUBDIR, exist_ok=True)

        # Extract features for all images
        image_paths = []
        futures = []
        total_indexed = 0
        # Get all image files from data path
        for root, dirs, files in os.walk(data_path):
            for file in files:
                if not file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                    # 跳过非图片文件
                    continue
                # 复制图片
                img_path = str(os.path.join(root, file))
                shutil.copy(img_path, DATA_PATH_SUBDIR)
                # 构建图片的绝对路径
                img_path = os.path.join(DATA_PATH_SUBDIR, file)
                # 构建图片的绝对路径
                image_paths.append(img_path)

                # 批量提取特征
                if len(image_paths) >= config.BATCH_SIZE:
                    # 复制数组
                    temp_image_paths = image_paths.copy()
                    # 清空数组
                    image_paths.clear()
                    future = executor.submit(process_predict_and_insert, temp_image_paths, table_name or DEFAULT_TABLE)
                    total_indexed += len(temp_image_paths)
                    cache_map.setdefault("total", total_indexed)
                    futures.append(future)
                    logging.info(f"Batch submitted {len(temp_image_paths)} images.")

        if len(image_paths) > 0:
            # 剩余的图片提取特征
            future = executor.submit(process_predict_and_insert, image_paths, table_name or DEFAULT_TABLE)
            total_indexed += len(image_paths)
            futures.append(future)
        # Wait for all tasks to complete and check for errors
        cache_map.setdefault("total", total_indexed)
        current = 0
        for future in futures:
            current += future.result()
            cache_map.setdefault("current", current)
        executor.shutdown(wait=True)
        logging.info(f"Total submitted so far: {len(futures)},Total indexed: {total_indexed},")
        return str(total_indexed)
    except Exception as e:
        if 'executor' in locals():
            executor.shutdown(wait=False)
        logging.error(f"Error in do_train: {e}")
        return str(e)


def train_status_cache():
    return cache_map


# 提取特征并插入 Milvus
def process_predict_and_insert(image_paths, table_name):
    # 提取特征
    try:
        # 获取特征提取器实例（延迟初始化）
        feature_extractor = get_feature_extractor()

        # 使用锁确保同一时刻只有一个线程在使用 GPU 进行预测
        with predict_lock:
            features = feature_extractor.extract_batch_features(image_paths)

        # Batch insert
        index.insert_vectors(table_name or DEFAULT_TABLE, features, image_paths)
        logging.info(f"Batch insert_vectors {len(image_paths)} images.")
    except Exception as e:
        logging.error(f"Error processing image : {e}")
    return len(features)
