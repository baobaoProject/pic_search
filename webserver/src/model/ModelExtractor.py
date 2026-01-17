import logging
import os
import threading
from abc import abstractmethod
from typing import Optional, Union

import torch
from PIL import Image

import common

# 全局锁，确保 model只加载一次 串行执行，避免 GPU OOM
predict_lock = threading.Lock()
feature_extractor_map = {}


# 抽象特征提取器
class Extractor:
    """Abstract base class for feature extractors."""

    @abstractmethod
    def extract_image_features(self, img_path):
        """Extract features for a single image."""
        raise NotImplementedError("extract_features method not implemented")

    @abstractmethod
    def batch_extract_image_features(self, image_paths):
        """Extract features for a batch of images."""
        raise NotImplementedError("extract_batch_features method not implemented")

    @abstractmethod
    def extract_text_features(self, text):
        """Extract features for a single text."""
        raise NotImplementedError("extract_features method not implemented")


# 定义一个get_feature_extractor的接口
class AbstractFeatureExtractor(Extractor):
    """Abstract base class for feature extractors."""
    cache_dir = "/root/.keras/models/huggingface/hub"
    model = None
    processor = None
    tokenizer = None
    device = None

    def __init__(self,
                 model_name: str = None,
                 model_id: Optional[Union[str, os.PathLike]] = None,
                 dimension=None,
                 language: str = None,
                 device: str = None):
        self.model_name = model_name
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        # 模型设备
        if self.device == "cuda":
            self.device_map = 0
        else:
            self.device_map = "cpu"
        self.model_id = model_id
        self.language = language
        self.dimension = dimension
        # 图片尺寸,定义图像的尺寸
        self.input_shape_size = common.get_image_shape()

        logging.info(f"Loading {model_name} feature extractor...")
        self.load_model()
        logging.info(f"{model_name} feature extractor loaded successfully.")

        logging.info(f"Loading {model_name} feature extractor processor...")
        self.load_processor()
        logging.info(f"{model_name} feature extractor processor loaded successfully.")

        logging.info(f"Loading {model_name} feature extractor tokenizer...")
        self.load_tokenizer()
        logging.info(f"{model_name} feature extractor tokenizer loaded successfully.")

    @abstractmethod
    def load_model(self):
        """Load the feature extractor model."""
        raise NotImplementedError("load_model method not implemented")

    @abstractmethod
    def load_processor(self):
        """Load the feature extractor processor."""
        raise NotImplementedError("load_processor method not implemented")

    @abstractmethod
    def load_tokenizer(self):
        """Load the feature extractor tokenizer."""
        raise NotImplementedError("load_tokenizer method not implemented")

    def _extract_image_features_(self, img_path):
        """Extract features for a single image."""
        logging.info(f"Extracting features for image: {img_path}")
        image = Image.open(img_path)
        try:
            # 预处理图片
            inputs = self.processor(images=image, max_num_patches=determine_max_value(image), return_tensors="pt").to(
                self.device)
            # 推理
            with torch.no_grad():
                # 根据模型类型选择对应的方法
                image_features = self.model.get_image_features(**inputs)
            # 归一化 (CLIP 的特征通常需要归一化)
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            # 转为列表
            return image_features.cpu().numpy()[0].tolist()
        except Exception as e:
            import traceback
            logging.error(f"Error extracting image features: {e}")
            logging.error(traceback.format_exc())
            raise e
        finally:
            image.close()

    def batch_extract_image_features(self, image_paths):
        """Extract features for a batch of images."""
        logging.info(f"Extracting features for {len(image_paths)} images.")
        features = []
        for path in image_paths:
            try:
                features.append(self.extract_image_features(path))
            except Exception as inner_e:
                logging.error(f"Failed to process {path}: {inner_e}")
                # 使用全0向量占位或跳过，这里选择抛出异常让上层处理
                raise inner_e
        return features


class ProxyFeatureExtractor(Extractor):
    """A proxy class for feature extractors."""

    def __init__(self, model_name: str = None):
        self.model_name = model_name
        self.instance: AbstractFeatureExtractor = ProxyFeatureExtractor.get_instance(model_name)

    def extract_image_features(self, img_path):
        logging.info(f"Extracting features for image: {img_path}")
        return self.instance.extract_image_features(img_path)

    def batch_extract_image_features(self, image_paths):
        return self.instance.batch_extract_image_features(image_paths)

    def extract_text_features(self, text):
        logging.info(f"Extracting features for text: {text}")
        return self.instance.extract_text_features(text)

    @classmethod
    def get_instance(cls, model_name=common.get_model_name()) -> "AbstractFeatureExtractor":
        global feature_extractor_map
        """Get a feature extractor instance by model name."""
        if feature_extractor_map.get(model_name) is not None:
            return feature_extractor_map.get(model_name)
        with predict_lock:
            if feature_extractor_map.get(model_name) is not None:
                return feature_extractor_map.get(model_name)
            else:
                logging.info(f"Initializing {model_name} feature extractor...")
                if model_name == "OPENAI-CLIP":
                    from model.extractor.clip_extractor import OpenAIClipFeatureExtractor
                    feature_extractor_map[model_name] = OpenAIClipFeatureExtractor(model_name)
                    logging.info("Feature extractor initialized. use OpenAIClipFeatureExtractor...")
                elif model_name == "OFA-ChineseCLIP":
                    from model.extractor.clip_extractor import OFAChineseClipFeatureExtractor
                    feature_extractor_map[model_name] = OFAChineseClipFeatureExtractor(model_name)
                    logging.info("Feature extractor initialized. use OFAChineseClipFeatureExtractor...")
                elif model_name == "EfficientNetV2S":
                    from model.extractor.efficientnet_extractor import EfficientNetFeatureExtractor
                    feature_extractor_map[model_name] = EfficientNetFeatureExtractor(model_name)
                    logging.info("Feature extractor initialized. use EfficientNetFeatureExtractor...")
                elif model_name == "Jinaai-CLIP":
                    from model.extractor.jinaai_extractor import JinaaiFeatureExtractor
                    feature_extractor_map[model_name] = JinaaiFeatureExtractor(model_name)
                    logging.info("Feature extractor initialized. use EfficientNetFeatureExtractor...")
                elif model_name == "Qwen3-VL":
                    from model.extractor.qwen_extractor import QwenFeatureExtractor
                    feature_extractor_map[model_name] = QwenFeatureExtractor(model_name)
                    logging.info("Feature extractor initialized. use QwenFeatureExtractor...")
                elif model_name == "Qihoo-CLIP2":
                    from model.extractor.qihuoo_extractor import QihooFeatureExtractor
                    feature_extractor_map[model_name] = QihooFeatureExtractor(model_name)
                    logging.info("Feature extractor initialized. use QihooFeatureExtractor...")
                else:
                    raise ValueError(f"Invalid model name : {model_name}")
        return feature_extractor_map.get(model_name)


def determine_max_value(image):
    w, h = image.size
    max_val = (w // 16) * (h // 16)
    if max_val > 784:
        return 1024
    elif max_val > 576:
        return 784
    elif max_val > 256:
        return 576
    elif max_val > 128:
        return 256
    else:
        return 128
