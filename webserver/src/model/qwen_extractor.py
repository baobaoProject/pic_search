from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

import common
from model import AbstractFeatureExtractor


class QwenFeatureExtractor(AbstractFeatureExtractor):

    def __init__(self, model_name="Qwen3-VL"):
        super().__init__(model_name=model_name, model_id=common.get_model_id(), dimension=common.get_model_dimension(),
                         language=common.get_model_language())

    def load_model(self):
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(self.model_id, dtype="auto", device_map="auto",
                                                                     cache_dir=self.cache_dir)
        return self.model

    def load_processor(self):
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        return self.processor

    def load_tokenizer(self):
        pass

    def extract_image_features(self, img_path):
        return super()._extract_image_features_(img_path)

    def batch_extract_image_features(self, image_paths):
        pass

    def extract_text_features(self, text):
        pass
