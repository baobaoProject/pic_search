from transformers import AutoModel, AutoProcessor, AutoTokenizer

import common
from model import AbstractFeatureExtractor


class GoogleFeatureExtractor(AbstractFeatureExtractor):

    def __init__(self, model_name="GoogleSiglip2"):
        super().__init__(model_name=model_name, model_id=common.get_model_id(), dimension=common.get_model_dimension(),
                         language=common.get_model_language())

    def load_model(self):
        self.model = AutoModel.from_pretrained(self.model_id,
                                               device_map=self.device_map,
                                               trust_remote_code=True,
                                               cache_dir=self.cache_dir)
        return self.model

    def load_processor(self):
        self.processor = AutoProcessor.from_pretrained(self.model_id, use_fast=True)
        return self.processor

    def load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        return self.tokenizer

    # 不支持文本特征提取
    def extract_text_features(self, text):
        pass
