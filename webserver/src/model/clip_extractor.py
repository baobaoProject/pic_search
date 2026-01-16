import logging
import threading

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, ChineseCLIPModel, ChineseCLIPProcessor, PreTrainedModel, \
    AutoTokenizer

import common

# 全局锁，确保 model只加载一次 串行执行，避免 GPU OOM
predict_lock = threading.Lock()
cache_dir = "/root/.keras/models/huggingface/hub"
# 延迟创建全局实例，避免在主进程中初始化CUDA
feature_extractor = None


def load_model_processor(device="cpu", language=common.get_model_language(), model_id=common.get_model_id()):
    """Lazy load the CLIP model."""
    # 在这里确定 device，避免在模块导入时初始化 CUDA
    # 使用 OpenAI 的 CLIP 模型 (ViT-B/32)
    model = None
    processor = None
    tokenizer = None
    try:
        if language == "CN":
            logging.info(f"Loading ChineseCLIPModel on {device}...")
            model = ChineseCLIPModel.from_pretrained(
                model_id,
                cache_dir=cache_dir
            ).to(device)
            logging.info("Chinese CLIP model loaded successfully.")
            logging.info("Loading ChineseCLIP processor...")
            processor = ChineseCLIPProcessor.from_pretrained(
                model_id,
                cache_dir=cache_dir,
                use_fast=True
            )
            logging.info("ChineseCLIP processor loaded successfully.")
            tokenizer = AutoTokenizer.from_pretrained(model_id)
        else:
            logging.info(f"Loading CLIPModel on {device}...")
            model = CLIPModel.from_pretrained(
                model_id,
                cache_dir=cache_dir
            ).to(device)
            logging.info("CLIPModel loaded successfully.")
            logging.info("Loading CLIP processor processor...")
            processor = CLIPProcessor.from_pretrained(
                model_id,
                cache_dir=cache_dir,
                use_fast=True
            )
            logging.info("CLIP processor loaded successfully.")
            tokenizer = AutoTokenizer.from_pretrained(model_id)
    except Exception as e:
        logging.error(f"Failed to load CLIP model: {e}")
        raise e
    return model,processor,tokenizer

class FeatureExtractor:
    def __init__(self, model_name):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        new_model, new_processor, new_tokenizer = load_model_processor(device=self.device,language=common.get_model_language(),  model_id=common.get_model_id())
        self.model = new_model
        self.processor = new_processor
        self.tokenizer = new_tokenizer
        # 根据模型类型动态获取维度
        self.dimension = common.get_model_dimension()

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

    def extract_batch_features(self, image_paths):
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


# 特征提取器单例
def clip_feature_extractor():
    """获取特征提取器单例，延迟初始化"""
    global feature_extractor
    if feature_extractor is not None:
        return feature_extractor
    with predict_lock:
        if feature_extractor is not None:
            return feature_extractor
        logging.info("Initializing feature extractor...")
        feature_extractor = FeatureExtractor(model_name=common.get_model_name())
        logging.info("Feature extractor initialized.")
    return feature_extractor
