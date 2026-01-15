import logging
import threading

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, ChineseCLIPModel, ChineseCLIPProcessor
from common import  config
# 全局锁，确保 model只加载一次 串行执行，避免 GPU OOM
predict_lock = threading.Lock()
model = None
processor = None
# 支持英文的模型id
en_model_id = "openai/clip-vit-base-patch32"
# 中文模型id
cn_model_id = "OFA-Sys/chinese-clip-vit-base-patch16"
cache_dir="/root/.keras/models/huggingface/hub"
device = "cuda" if torch.cuda.is_available() else "cpu"
# 延迟创建全局实例，避免在主进程中初始化CUDA
# 这样可以防止Gunicorn fork子进程时出现CUDA重新初始化错误
feature_extractor = None


def load_model():
    """Lazy load the CLIP model."""
    global model
    if model is not None:
        return model
    with predict_lock:
        if model is None:
            logging.info(f"Loading CLIP model on {device}...")
            # 使用 OpenAI 的 CLIP 模型 (ViT-B/32)
            try:
                if config.MODEL_LANGUAGE == "CN":
                    model = ChineseCLIPModel.from_pretrained(
                        cn_model_id,
                        cache_dir=cache_dir
                    ).to(device)
                else:
                    model = CLIPModel.from_pretrained(
                        en_model_id,
                        cache_dir=cache_dir
                    ).to(device)
                logging.info("CLIP model loaded successfully.")
            except Exception as e:
                logging.error(f"Failed to load CLIP model: {e}")
                raise e
    return model

def load_processor():
    """Lazy load the processor."""
    global processor
    if processor is not None:
        return processor
    with predict_lock:
        if processor is None:
            logging.info("Loading CLIP processor...")
            try:
                if config.MODEL_LANGUAGE == "CN":
                    processor = ChineseCLIPProcessor.from_pretrained(
                        cn_model_id,
                        cache_dir=cache_dir,
                        use_fast=True
                    )
                else:
                    processor = CLIPProcessor.from_pretrained(
                        en_model_id,
                        cache_dir=cache_dir,
                        use_fast=True
                    )
                logging.info("CLIP processor loaded successfully.")
            except Exception as e:
                logging.error(f"Failed to load CLIP processor: {e}")
                raise e
    return processor

class FeatureExtractor:
    def __init__(self):
        self.model = load_model()
        self.processor = load_processor()
        self.device = device


    def extract_image_features(self, img_path):
        """Extract features for a single image."""
        try:
            image = Image.open(img_path)
            # 预处理图片
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)

            # 推理
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)

            # 归一化 (CLIP 的特征通常需要归一化)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # 转为列表
            return image_features.cpu().numpy()[0].tolist()
        except Exception as e:
            logging.error(f"Error extracting image features: {e}")
            raise e

    def extract_text_features(self, text):
        """Extract features for text."""
        try:
            # 预处理文本 (截断长度设为 77，这是 CLIP 的默认最大长度)
            inputs = self.processor(text=[text], padding=True, truncation=True, return_tensors="pt").to(self.device)

            # 推理
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)

            # 归一化
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            return text_features.cpu().numpy()[0].tolist()
        except Exception as e:
            logging.error(f"Error extracting text features: {e}")
            raise e

    def extract_batch_features(self, image_paths):
        """Extract features for a batch of images."""
        try:
            images = [Image.open(path) for path in image_paths]

            # 批量预处理
            inputs = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)

            with torch.no_grad():
                batch_features = self.model.get_image_features(**inputs)

            # 批量归一化
            batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True)
            return batch_features.cpu().numpy().tolist()
        except Exception as e:
            logging.error(f"Error extracting batch features: {e}")
            # 如果批量失败，尝试逐个提取
            logging.info("Falling back to sequential extraction...")
            features = []
            for path in image_paths:
                try:
                    features.append(self.extract_image_features(path))
                except Exception as inner_e:
                    logging.error(f"Failed to process {path}: {inner_e}")
                    # 使用全0向量占位或跳过，这里选择抛出异常让上层处理
                    raise inner_e
            return features

def get_feature_extractor():
    """获取特征提取器单例，延迟初始化"""
    global feature_extractor
    if feature_extractor is not None:
        return feature_extractor
    if feature_extractor is None:
        logging.info("Initializing feature extractor...")
        feature_extractor = FeatureExtractor()
        logging.info("Feature extractor initialized.")
    return feature_extractor