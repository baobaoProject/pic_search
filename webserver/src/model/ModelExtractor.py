import logging
import threading
from abc import abstractmethod

import torch

import common

# 全局锁，确保 model只加载一次 串行执行，避免 GPU OOM
predict_lock = threading.Lock()
feature_extractor_map={}
# 定义一个get_feature_extractor的接口
class ProxyFeatureExtractor:

    """Abstract base class for feature extractors."""
    def __init__(self, model_name=None,model_id=None,dimension=None,language=None,device=None):
        self.model_name = model_name
        self.device = device if device is not None  else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_id = model_id
        self.language = language
        self.dimension = dimension
        self.model = None
        self.load_model()

        self.processor =None
        self.load_processor()

        self.tokenizer = None
        self.load_tokenizer()

    @classmethod
    def get_instance(cls, model_name=common.get_model_name()) -> "ProxyFeatureExtractor":
        global feature_extractor_map
        """Get a feature extractor instance by model name."""
        if feature_extractor_map.get(model_name) is not None:
            return feature_extractor_map.get(model_name)
        with predict_lock:
            if feature_extractor_map.get(model_name) is not None:
                return feature_extractor_map.get(model_name)
            else:
                logging.info("Initializing {} feature extractor...",model_name)
                if model_name == "CLIP":
                    from .clip_extractor import ClipFeatureExtractor
                    feature_extractor_map[model_name] = ClipFeatureExtractor(model_name)
                    logging.info("Feature extractor initialized. use ClipFeatureExtractor...")
                elif model_name == "EfficientNet":
                    from .efficientnet_extractor import EfficientNetFeatureExtractor
                    feature_extractor_map[model_name] = EfficientNetFeatureExtractor(model_name)
                    logging.info("Feature extractor initialized. use EfficientNetFeatureExtractor...")
                else:
                     raise ValueError("Invalid model name")
        return feature_extractor_map.get(model_name)

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