import logging

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, ChineseCLIPModel, ChineseCLIPProcessor, AutoTokenizer

import common
from model.ModelExtractor import ProxyFeatureExtractor

cache_dir = "/root/.keras/models/huggingface/hub"

# CLIP 特征提取器
class ClipFeatureExtractor(ProxyFeatureExtractor):

    # 构造函数
    def __init__(self, model_name="CLIP"):
        super().__init__(model_name=model_name, model_id=common.get_model_id(), dimension=common.get_model_dimension(), language=common.get_model_language())

    # 加载模型
    def load_model(self):
        try:
            if self.language == "CN":
                logging.info(f"Loading ChineseCLIPModel on {self.device}...")
                self.model = ChineseCLIPModel.from_pretrained(self.model_id,cache_dir=cache_dir).to(self.device)
                logging.info("Chinese CLIP model loaded successfully.")
            else:
                logging.info(f"Loading CLIPModel on {self.device}...")
                self.model = CLIPModel.from_pretrained(self.model_id,cache_dir=cache_dir).to(self.device)
                logging.info("CLIPModel loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load CLIP model: {e}")
            raise e
        return self.model

    # 加载处理器
    def load_processor(self):
        try:
            if self.language == "CN":
                logging.info("Loading ChineseCLIP processor...")
                self.processor = ChineseCLIPProcessor.from_pretrained(self.model_id, cache_dir=cache_dir, use_fast=True)
                logging.info("ChineseCLIP processor loaded successfully.")
            else:
                logging.info("Loading CLIP processor processor...")
                self.processor = CLIPProcessor.from_pretrained(self.model_id, cache_dir=cache_dir, use_fast=True)
                logging.info("CLIP processor loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load CLIP model: {e}")
            raise e
        return  self.processor

    # 加载tokenizer
    def load_tokenizer(self):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        except Exception as e:
            logging.error(f"Failed to load CLIP model: {e}")
            raise e
        return  self.tokenizer

    def extract_image_features(self, img_path):
        """Extract features for a single image."""
        logging.info(f"Extracting features for image: {img_path}")
        try:
            image = Image.open(img_path)
            # 预处理图片
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            # 推理
            with torch.no_grad():
                # 根据模型类型选择对应的方法
                image_features = self.model.get_image_features(**inputs)
            # 归一化 (CLIP 的特征通常需要归一化)
            image_features = image_features / image_features.norm(p=2,dim=-1, keepdim=True)
            # 转为列表
            return image_features.cpu().numpy()[0].tolist()
        except Exception as e:
            import traceback
            logging.error(f"Error extracting image features: {e}")
            logging.error(traceback.format_exc())
            raise e

    def extract_text_features(self, text):
        """Extract features for text."""
        try:
            # 处理文本，添加最大长度限制
            model_inputs = self.processor(text=[text],padding=True,return_tensors="pt").to(self.device)
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