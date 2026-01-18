import logging

from transformers import AutoTokenizer, CLIPModel, CLIPProcessor, ChineseCLIPModel, ChineseCLIPProcessor

import common
from model.ModelExtractor import AbstractFeatureExtractor


# CLIP 特征提取器
class ClipFeatureExtractor(AbstractFeatureExtractor):

    # 构造函数
    def __init__(self, model_name="OPENAI-CLIP"):
        super().__init__(model_name=model_name, model_id=common.get_model_id(), dimension=common.get_model_dimension(),
                         language=common.get_model_language())

    # 加载模型
    def load_model(self):
        try:
            logging.info(f"Loading CLIPModel on {self.device}...")
            self.model = CLIPModel.from_pretrained(self.model_id, cache_dir=self.cache_dir)
            logging.info("CLIPModel loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load CLIP model: {e}")
            raise e
        return self.model

    # 加载处理器
    def load_processor(self):
        try:
            logging.info("Loading CLIP processor processor...")
            self.processor = CLIPProcessor.from_pretrained(self.model_id, cache_dir=self.cache_dir, use_fast=True)
            logging.info("CLIP processor loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load CLIP model: {e}")
            raise e
        return self.processor

    # 加载tokenizer
    def load_tokenizer(self):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        except Exception as e:
            logging.error(f"Failed to load CLIP model: {e}")
            raise e
        return self.tokenizer


# 中文CLIP 特征提取器
class ChineseClipFeatureExtractor(ClipFeatureExtractor):
    """CLIP 特征提取器"""

    def __init__(self, model_name="OFA-ChineseCLIP"):
        super().__init__(model_name=model_name)

    # 加载模型
    def load_model(self):
        try:
            logging.info(f"Loading ChineseCLIPModel on {self.device}...")
            self.model = ChineseCLIPModel.from_pretrained(self.model_id, cache_dir=self.cache_dir).to(self.device)
            logging.info("Chinese CLIP model loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load CLIP model: {e}")
            raise e
        return self.model

    # 加载处理器
    def load_processor(self):
        try:
            logging.info("Loading ChineseCLIP processor...")
            self.processor = ChineseCLIPProcessor.from_pretrained(self.model_id, cache_dir=self.cache_dir,
                                                                  use_fast=True)
            logging.info("ChineseCLIP processor loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load CLIP model: {e}")
            raise e
        return self.processor
