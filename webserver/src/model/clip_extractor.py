import logging
import threading

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, ChineseCLIPModel, ChineseCLIPProcessor
import common

# 全局锁，确保 model只加载一次 串行执行，避免 GPU OOM
predict_lock = threading.Lock()
cache_dir = "/root/.keras/models/huggingface/hub"
# 延迟创建全局实例，避免在主进程中初始化CUDA
feature_extractor = None


def load_model_processor(device="cpu", language=common.get_model_language(), model_id=common.get_model_id()):
    """Lazy load the CLIP model."""
    # 在这里确定 device，避免在模块导入时初始化 CUDA
    # 使用 OpenAI 的 CLIP 模型 (ViT-B/32)
    model = None
    processor = None
    try:
        if language == "CN":
            logging.info(f"Loading ChineseCLIPModel on {device}...")
            model = ChineseCLIPModel.from_pretrained(
                model_id,
                cache_dir=cache_dir
            ).to(device)
            logging.info("Chinese CLIP model loaded successfully.")
            logging.info("Loading ChineseCLIP processor...")
            processor = ChineseCLIPProcessor.from_pretrained(
                model_id,
                cache_dir=cache_dir,
                use_fast=True
            )
            logging.info("ChineseCLIP processor loaded successfully.")
        else:
            logging.info(f"Loading CLIPModel on {device}...")
            model = CLIPModel.from_pretrained(
                model_id,
                cache_dir=cache_dir
            ).to(device)
            logging.info("CLIPModel loaded successfully.")
            logging.info("Loading CLIP processor processor...")
            processor = CLIPProcessor.from_pretrained(
                model_id,
                cache_dir=cache_dir,
                use_fast=True
            )
            logging.info("CLIP processor loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load CLIP model: {e}")
        raise e
    return model,processor

class FeatureExtractor:
    def __init__(self, model_name):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        new_model, new_processor = load_model_processor(device=self.device,language=common.get_model_language(),  model_id=common.get_model_id())
        self.model = new_model
        self.processor = new_processor
        # 根据模型类型动态获取维度
        self.dimension = common.get_model_dimension()

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
            # 确保文本不为空
            if not text or not text.strip():
                logging.warning("Empty text provided for feature extraction")
                # 返回一个零向量，使用模型实际的维度
                import numpy as np
                return [0.0] * self.dimension

            # 处理文本，添加最大长度限制
            processed_text = text.strip()[:75]  # 限制文本长度，避免超出模型限制

            # 使用处理器处理文本
            inputs = self.processor(
                text=[processed_text],
                padding=True,
                truncation=True,
                max_length=77,  # CLIP 模型的最大序列长度
                return_tensors="pt"
            )

            # 检查 inputs 是否有效
            if inputs is None:
                logging.error("Processor returned None")
                raise ValueError("Processor failed to process text")

            inputs = inputs.to(self.device)

            # 确保所有张量都在正确的设备上
            for key, value in inputs.items():
                if torch.is_tensor(value):
                    inputs[key] = value.to(self.device)

            # 仅保留 Tensor 类型的输入
            model_inputs = {k: v for k, v in inputs.items() if v is not None and torch.is_tensor(v)}

            # 确保 model_inputs 不为空
            if not model_inputs:
                logging.error("Model inputs are empty after filtering")
                raise ValueError("No valid tensors in model inputs")

            # 推理
            with torch.no_grad():
                text_features = self.model.get_text_features(**model_inputs)

            # 检查特征是否有效
            if text_features is None:
                logging.error("Model returned None for text features")
                raise ValueError("Model failed to generate text features")

            # 归一化 (使用 p=2, dim=-1，与官方保持一致)
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

            return text_features.cpu().numpy()[0].tolist()
        except Exception as e:
            logging.error(f"Error extracting text features: {e}")
            # 尝试使用备用方案
            import numpy as np
            return [0.0] * self.dimension

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

# 特征提取器单例
def clip_feature_extractor():
    """获取特征提取器单例，延迟初始化"""
    global feature_extractor
    if feature_extractor is not None:
        return feature_extractor
    with predict_lock:
        if feature_extractor is not None:
            return feature_extractor
        logging.info("Initializing feature extractor...")
        feature_extractor = FeatureExtractor(model_name=common.get_model_name())
        logging.info("Feature extractor initialized.")
    return feature_extractor
