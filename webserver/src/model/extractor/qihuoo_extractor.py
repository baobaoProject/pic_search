from transformers import AutoModelForCausalLM

import common
from model import AbstractFeatureExtractor


class QihooFeatureExtractor(AbstractFeatureExtractor):

    def __init__(self, model_name="Qihoo-CLIP2"):
        super().__init__(model_name=model_name, model_id=common.get_model_id(), dimension=common.get_model_dimension(),
                         language=common.get_model_language())

    def load_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, trust_remote_code=True,
                                                          cache_dir=self.cache_dir)
        return self.model

    def load_processor(self):
        self.processor = self.load_image_processor()
        return self.processor

    def extract_text_features(self, text):
        return super().extract_text_features_tokenizer(text)
