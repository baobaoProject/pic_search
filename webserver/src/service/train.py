import hashlib
import logging
import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import common
from common import config
from indexer import index
from model import get_feature_extractor

# 全局锁，确保 model.predict 串行执行，避免 GPU OOM
predict_lock = threading.Lock()
# 全局map
cache_map = {}


def do_train(table_name, data_path: str, embedding_index_type):
    """
    Train the model by indexing images to Milvus
    """
    cache_map.clear()
    table_name = table_name or common.get_model_default_table()

    # 创建线程池
    executor = ThreadPoolExecutor(max_workers=config.MAX_THREADS)
    try:
        # Create table if not exists
        if index.has_collection(table_name) is False:
            # 创建表
            with predict_lock:
                # 获取特征提取器实例（延迟初始化）
                feature_extractor = get_feature_extractor()
                index.create_table(table_name,
                                   False,
                                   embedding_index_type or config.EMBEDDING_INDEX_TYPE,
                                   dimension=feature_extractor.get_vector_dimension())

        # data_path去空格
        data_path = os.path.normpath(data_path.strip())
        # 计算data_path的md5值并取前10位
        data_path_md5 = hashlib.md5(data_path.encode('utf-8')).hexdigest()[:10]
        DATA_PATH_SUBDIR = os.path.join(config.DATA_PATH, data_path_md5)
        if not os.path.exists(DATA_PATH_SUBDIR):
            # 创建目录，如果不存在
            os.makedirs(DATA_PATH_SUBDIR, exist_ok=True)

        # 流式处理：逐个处理图片，按批次提交任务
        batch_paths = []
        futures = []
        total_indexed = 0

        # 遍历图片文件并立即处理
        for root, dirs, files in os.walk(data_path):
            for file in files:
                if not file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                    # 跳过非图片文件
                    continue

                img_path = str(os.path.join(root, file))
                if not os.path.exists(img_path):
                    logging.error(f"Image {img_path} does not exist.")
                    continue
                # 添加到当前批次
                batch_paths.append(img_path)
                # 当批次达到指定大小时提交任务
                if len(batch_paths) >= config.BATCH_SIZE:
                    # 等待线程池有空闲容量
                    while len([f for f in futures if not f.done()]) >= config.MAX_THREADS * 2:
                        time.sleep(0.1)  # 短暂休眠，避免忙等待

                    # 提交当前批次的任务
                    future = executor.submit(process_predict_and_insert, batch_paths.copy(), table_name, DATA_PATH_SUBDIR)
                    futures.append(future)
                    total_indexed += len(batch_paths)
                    cache_map.setdefault("total", total_indexed)

                    logging.info(f"Batch submitted total {total_indexed} images, current batch: {len(batch_paths)} images.")

                    # 清空当前批次
                    batch_paths.clear()

        # 处理最后一个不满批次的任务
        if len(batch_paths) > 0:
            # 等待线程池有空闲容量
            while len([f for f in futures if not f.done()]) >= config.MAX_THREADS:
                time.sleep(0.5)

            future = executor.submit(process_predict_and_insert, batch_paths, table_name, DATA_PATH_SUBDIR)
            futures.append(future)
            total_indexed += len(batch_paths)
            logging.info(f"Final batch submitted total {total_indexed} images, final batch: {len(batch_paths)} images.")

        # 等待所有任务完成
        cache_map.setdefault("total", total_indexed)
        current = 0
        for i, future in enumerate(futures):
            try:
                result = future.result()  # 这里会阻塞直到任务完成
                current += result
                cache_map.setdefault("current", current)
                logging.info(f"Completed batch {i + 1}/{len(futures)}, total processed: {current} images.")
            except Exception as e:
                logging.error(f"Error processing future: {e}")

        logging.info(f"Total submitted: {len(futures)} batches, Total indexed: {total_indexed} images.")
        return str(total_indexed)
    except Exception as e:
        logging.error(f"Error in do_train: {e}")
        return str(e)
    finally:
        # 确保线程池被关闭
        executor.shutdown(wait=True)


def train_status_cache():
    return cache_map


# 提取特征并插入 Milvus
def process_predict_and_insert(image_paths, table_name, data_path_sub):
    # 提取特征
    batch_paths = []
    try:
        # 遍历图片，把图片复制到DATA_PATH_SUBDIR目录下
        for img_path in image_paths:
            # 构建图片的目标路径
            file = os.path.basename(img_path)
            new_img_path = os.path.join(data_path_sub, file)
            # 判断文件是否存在
            if os.path.exists(new_img_path):
                logging.info(f"Image {new_img_path} already exists, skipping.")
                continue
            shutil.copy2(img_path, data_path_sub)
            batch_paths.append(new_img_path)

        if len(batch_paths) == 0:
            logging.info("No images to process.")
            return 0
        # 获取特征提取器实例（延迟初始化）
        features = get_feature_extractor().batch_extract_image_features(batch_paths)
        # Batch insert
        index.insert_vectors(table_name or common.get_model_default_table(), features, batch_paths)
        logging.info(f"Inserted {len(features)} vectors into Milvus.")
        return len(features)
    except Exception as e:
        logging.error(f"Error processing image : {e}")
        return 0
    finally:
        batch_paths.clear()
