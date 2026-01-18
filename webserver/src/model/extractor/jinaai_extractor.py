import torch
from transformers import AutoImageProcessor, AutoTokenizer

import common
from model import AbstractFeatureExtractor
from model.extractor.jinaai.modeling_clip import JinaCLIPModel


class JinaaiFeatureExtractor(AbstractFeatureExtractor):

    def __init__(self, model_name="Jinaai-CLIP"):
        super().__init__(model_name=model_name, model_id=common.get_model_id(), dimension=common.get_model_dimension(),
                         language=common.get_model_language(), torch_dtype=torch.float16)

    def load_model(self):
        self.model = JinaCLIPModel.from_pretrained(self.model_id,
                                                   cache_dir=self.cache_dir,
                                                   trust_remote_code=True,
                                                   dtype=self.torch_dtype)  # 明确指定数据类型
        return self.model

    def load_processor(self):
        self.processor = AutoImageProcessor.from_pretrained(self.model_id, trust_remote_code=True,
                                                            cache_dir=self.cache_dir)
        return self.processor

    def load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True, cache_dir=self.cache_dir)
        return self.tokenizer

    def extract_text_features(self, text):
        return super().extract_text_features_tokenizer(text)
