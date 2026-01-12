import logging
import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_input_vgg  # type: ignore
from tensorflow.keras.preprocessing import image  # type: ignore

from common import config
from common.config import DEFAULT_TABLE
from common.const import image_size
from  indexer import  index

# 全局锁，确保 model.predict 串行执行，避免 GPU OOM
predict_lock = threading.Lock()

def do_train(table_name, data_path, model,vector_dimension, embedding_index_type):
    # 创建线程池
    executor = ThreadPoolExecutor(max_workers=config.MAX_THREADS)
    """
    Train the model by indexing images to Milvus
    """
    try:
        # Create table if not exists
        if index.has_collection(table_name or DEFAULT_TABLE) is False:
            # 创建表
            with predict_lock:
                index.create_table(table_name or config.DEFAULT_TABLE,
                             vector_dimension or config.VECTOR_DIMENSION,
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
                    future = executor.submit(process_predict_and_insert,temp_image_paths,model, table_name or DEFAULT_TABLE)
                    futures.append(future)
                    total_indexed += len(temp_image_paths)
                    logging.info(f"Batch submitted {len(temp_image_paths)} images.")

        if len(image_paths) > 0:
            # 剩余的图片提取特征
            future = executor.submit(process_predict_and_insert, image_paths, model, table_name or DEFAULT_TABLE)
            futures.append(future)
            total_indexed += len(image_paths)
        # Wait for all tasks to complete and check for errors
        for future in futures:
            future.result()
        executor.shutdown(wait=True)
        logging.info(f"Total submitted so far: {len(futures)},Total indexed: {total_indexed},")
        return str(total_indexed)
    except Exception as e:
        if 'executor' in locals():
            executor.shutdown(wait=False)
        logging.error(f"Error in do_train: {e}")
        return str(e)


# 提取特征并插入 Milvus
def process_predict_and_insert(image_paths,model,table_name):
    # 提取特征
    features = []
    try:
        batch_tensors = []
        for img_path in image_paths:
            img_tensor = preprocess_img_vgg(img_path)
            batch_tensors.append(img_tensor)

        # 将所有图像合并为一个批次
        batch = np.vstack(batch_tensors)

        # 一次性预测整个批次
        # 使用锁确保同一时刻只有一个线程在使用 GPU 进行预测
        with predict_lock:
            batch_features = model.predict(batch)
            
        # 分别处理每个特征向量
        for i, feat in enumerate(batch_features):
            feat = feat.flatten()
            feat = feat / np.linalg.norm(feat)
            features.append(feat.tolist())

        # Batch insert
        index.insert_vectors(table_name or DEFAULT_TABLE, features, image_paths)
        logging.info(f"Batch insert_vectors {len(image_paths)} images.")
    except Exception as e:
        logging.error(f"Error processing image : {e}")
    return len(features)


# 预处理图片
def preprocess_img_vgg(img_path):
    img = image.load_img(img_path, target_size=image_size)
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor = preprocess_input_vgg(img_tensor)
    return img_tensor