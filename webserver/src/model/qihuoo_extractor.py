import logging

import torch
from transformers import AutoImageProcessor, AutoModelForCausalLM, AutoTokenizer

import common
from model import AbstractFeatureExtractor


class QihooFeatureExtractor(AbstractFeatureExtractor):

    def __init__(self, model_name="Qihoo-CLIP2"):
        super().__init__(model_name=model_name, model_id=common.get_model_id(), dimension=common.get_model_dimension(),
                         language=common.get_model_language())

    def load_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, trust_remote_code=True,
                                                          cache_dir=self.cache_dir, ).cuda()
        return self.model

    def load_processor(self):
        self.processor = AutoImageProcessor.from_pretrained(self.model_id)
        return self.processor

    def load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        return self.tokenizer

    def extract_image_features(self, img_path):
        return super()._extract_image_features_(img_path)

    def batch_extract_image_features(self, image_paths):
        return super().batch_extract_image_features(image_paths)

    def extract_text_features(self, text):
        """Extract features for text."""
        try:
            # 处理文本，添加最大长度限制
            model_inputs = self.tokenizer([text], padding="max_length", max_length=196, truncation=True,
                                          return_tensors="pt").to(self.device)
            # 推理
            with torch.no_grad():
                text_features = self.model.get_text_features(**model_inputs, walk_type="long")
                logging.info(f"text_features shape: {text_features.shape}")

            # 归一化 (使用 p=2, dim=-1，与官方保持一致)
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            return text_features.cpu().numpy()[0].tolist()
        except Exception as e:
            import traceback
            logging.error(f"Error extracting text features: {e}")
            logging.error(traceback.format_exc())
            raise ValueError("Processor failed to process text")


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
