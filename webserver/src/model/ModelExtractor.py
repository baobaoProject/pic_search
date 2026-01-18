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
    device = common.get_device()

    def __init__(self,
                 model_name: str = None,
                 model_id: Optional[Union[str, os.PathLike]] = None,
                 dimension=None,
                 language: str = None,
                 device: str = None,
                 torch_dtype=None):
        self.model_name = model_name
        # 设备不是cpu，自动计算
        if self.device != "cpu":
            self.device = "cuda"
            self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        # 模型设备
        if self.device == "cuda":
            self.device_map = 0
        else:
            self.device_map = "cpu"
        self.model_id = model_id
        self.language = language
        self.dimension = dimension
        self.torch_dtype = torch_dtype
        # 图片尺寸,定义图像的尺寸
        self.input_shape_size = common.get_image_shape()
        # 打印上面所有参数
        logging.info(
            f"params::: model_name: {self.model_name}, device:{self.device}, model_id:{self.model_id}, dimension:{self.dimension}, language:{self.language}, torch_dtype:{self.torch_dtype}")

        logging.info(f"Loading {model_name} feature extractor...")
        self.load_model()
        try:
            self.model = self.model.to(self.device)
        except Exception as e:
            logging.error(f"Error model to {self.device}: {e}")
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

    def extract_image_features(self, img_path):
        """Extract features for a single image."""
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
        logging.debug(f"Extracting features for {len(image_paths)} images.")
        features = []
        for path in image_paths:
            try:
                features.append(self.extract_image_features(path))
            except Exception as inner_e:
                logging.error(f"Failed to process {path}: {inner_e}")
                # 使用全0向量占位或跳过，这里选择抛出异常让上层处理
                raise inner_e
        return features

    def extract_text_features(self, text):
        """Extract features for text."""
        try:
            # 处理文本，添加最大长度限制
            model_inputs = self.processor(text=[text], padding=True, return_tensors="pt").to(self.device)
            # model_inputs = self.tokenizer([text], padding=True, return_tensors="pt").to(self.device)
            logging.info(f"inputs: {model_inputs}")
            # 推理
            with torch.no_grad():
                # text_features = self.model.get_text_features(**model_inputs)
                # 由于 transformers 版本兼容性问题，get_text_features 可能因 pooler_output 为 None 而崩溃
                # 这里手动实现 get_text_features 的逻辑：获取 last_hidden_state -> 取 [CLS] -> text_projection
                text_outputs = self.model.text_model(**model_inputs)

                # 获取 last_hidden_state (BatchEncoding返回值通常是对象，也可以像元组一样索引)
                if isinstance(text_outputs, tuple):
                    last_hidden_state = text_outputs[0]
                else:
                    last_hidden_state = text_outputs.last_hidden_state

                # 取 [CLS] token 对应的特征 (batch_size, hidden_size)
                pooled_output = last_hidden_state[:, 0, :]

                # 投影到联合嵌入空间
                text_features = self.model.text_projection(pooled_output)
                logging.info(f"text_features shape: {text_features.shape}")

            # 归一化 (使用 p=2, dim=-1，与官方保持一致)
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            return text_features.cpu().numpy()[0].tolist()
        except Exception as e:
            import traceback
            logging.error(f"Error extracting text features: {e}")
            logging.error(traceback.format_exc())
            raise ValueError("Processor failed to process text")

    def extract_text_features_tokenizer(self, text):
        try:
            # 处理文本，添加最大长度限制
            model_inputs = self.tokenizer([text], padding="max_length", truncation=True,
                                          return_tensors="pt").to(self.device)
            # 推理
            with torch.no_grad():
                text_features = self.model.get_text_features(**model_inputs)
                logging.info(f"text_features shape: {text_features.shape}")

            # 归一化 (使用 p=2, dim=-1，与官方保持一致)
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            # 判断text_features是否是torch.float32类型
            if self.torch_dtype is not None and text_features.dtype != self.torch_dtype:
                text_features = text_features.to(self.torch_dtype)
            return text_features.cpu().numpy()[0].tolist()
        except Exception as e:
            import traceback
            logging.error(f"Error extracting text features: {e}")
            logging.error(traceback.format_exc())
            raise ValueError("Processor failed to process text")


class ProxyFeatureExtractor(Extractor):
    """A proxy class for feature extractors."""

    def __init__(self, model_name: str = None):
        self.model_name = model_name
        self.instance: AbstractFeatureExtractor = ProxyFeatureExtractor.get_instance(model_name)

    def extract_image_features(self, img_path):
        logging.debug(f"Extracting features for image: {img_path}")
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
                model_type = common.get_model_type(model_name).lower()
                logging.info(f"Initializing {model_name} feature extractor...")
                instanceObj = None
                if model_type == "CLIPModel".lower():
                    from model.extractor.clip_extractor import ClipFeatureExtractor
                    instanceObj = ClipFeatureExtractor(model_name)
                elif model_type == "ChineseCLIPModel".lower():
                    from model.extractor.clip_extractor import ChineseClipFeatureExtractor
                    instanceObj = ChineseClipFeatureExtractor(model_name)
                elif model_type == "EfficientNet".lower():
                    from model.extractor.efficientnet_extractor import EfficientNetFeatureExtractor
                    instanceObj = EfficientNetFeatureExtractor(model_name)
                elif model_type == "Qwen3VLForConditionalGeneration".lower():
                    from model.extractor.qwen_extractor import QwenFeatureExtractor
                    instanceObj = QwenFeatureExtractor(model_name)
                elif model_type == "AutoModelForCausalLM".lower():
                    from model.extractor.qihuoo_extractor import QihooFeatureExtractor
                    instanceObj = QihooFeatureExtractor(model_name)
                elif model_type == "AutoModel".lower():
                    from model.extractor.google_extractor import GoogleFeatureExtractor
                    instanceObj = GoogleFeatureExtractor(model_name)
                elif model_type == "jinaai".lower():
                    from model.extractor.jinaai_extractor import JinaaiFeatureExtractor
                    instanceObj = JinaaiFeatureExtractor(model_name)
                else:
                    raise ValueError(f"Invalid model name : {model_name}")
                feature_extractor_map[model_name] = instanceObj
                logging.info(f"Feature extractor initialized. use {instanceObj.__class__}...")
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
