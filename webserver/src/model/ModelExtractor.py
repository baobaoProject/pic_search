import logging
import os
import threading
from abc import abstractmethod
from typing import Optional, Union

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, AutoProcessor, AutoTokenizer

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

    @abstractmethod
    def get_vector_dimension(self):
        """
        """
        raise NotImplementedError("get_vector_dimension method not implemented")


# 定义一个get_feature_extractor的接口
class AbstractFeatureExtractor(Extractor):
    """Abstract base class for feature extractors."""
    cache_dir = "/root/.keras/models/huggingface/hub"
    cache_checkpoint_dir = "/root/.keras/models/checkpoints"
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
        self.model_config: common.ModelConfig = common.get_model_config(model_name)
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
        self.torch_dtype = torch_dtype
        # 图片尺寸,定义图像的尺寸
        self.input_shape_size = common.get_image_shape()
        logging.info(f"Loading {model_name} feature extractor...")
        self.load_model()
        # 优先从模型内部获取向量维度
        self.dimension = self.get_vector_dimension() or dimension

        # 打印上面所有参数
        logging.info(
            f"params::: model_name: {self.model_name}, device:{self.device}, model_id:{self.model_id}, dimension:{self.dimension}, language:{self.language}, torch_dtype:{self.torch_dtype}")

        try:
            self.model = self.model.to(self.device)
        except Exception as e:
            logging.error(f"Error model to {self.device}: {e}")

        # 加载checkpoints
        self.load_model_checkpoints()

        logging.info(f"{model_name} feature extractor loaded successfully.")

        logging.info(f"Loading {model_name} feature extractor processor...")
        self.load_processor()
        logging.info(f"{model_name} feature extractor processor loaded successfully.")

        logging.info(f"Loading {model_name} feature extractor tokenizer...")
        self.load_tokenizer()
        logging.info(f"{model_name} feature extractor tokenizer loaded successfully.")

    # 加载模型检查点
    def load_model_checkpoints(self):
        if self.model_config is None or self.model_config.model_checkpoints is None:
            return
        # 是数组，则遍历
        for model_checkpoint in self.model_config.model_checkpoints:
            checkpoint_path = os.path.join(self.cache_checkpoint_dir, model_checkpoint)
            if os.path.exists(checkpoint_path):
                logging.info(f"Loading {self.model_name} feature extractor checkpoints:{checkpoint_path}...")
                try:
                    # 由于PyTorch 2.6默认weights_only=True，而检查点文件包含不支持的对象，
                    # 所以设置weights_only=False来加载模型检查点
                    checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
                    sd = checkpoint.get("state_dict", checkpoint)  # 支持直接是state_dict的检查点
                    if next(iter(sd.items()))[0].startswith('module'):
                        sd = {k[len('module.'):]: v for k, v in sd.items() if "bert.pooler" not in k}
                    # 加载状态字典，允许部分匹配
                    missing_keys, unexpected_keys = self.model.load_state_dict(sd, strict=False)
                    if missing_keys:
                        logging.warning(f"Missing keys when loading checkpoints: {missing_keys}")
                    if unexpected_keys:
                        logging.warning(f"Unexpected keys when loading checkpoints: {unexpected_keys}")
                    logging.info(f"Successfully loaded checkpoints: {checkpoint_path}")
                except Exception as e:
                    logging.error(f"Failed to load checkpoints {checkpoint_path}: {e}")

    def load_model(self):
        self.model = AutoModel.from_pretrained(self.model_id, device_map=self.device_map, trust_remote_code=True,
                                               cache_dir=self.cache_dir)
        return self.model

    def load_processor(self):
        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True, cache_dir=self.cache_dir,
                                                       use_fast=True)
        return self.processor

    # 加载image_processor
    def load_image_processor(self):
        try:
            self.processor = AutoImageProcessor.from_pretrained(self.model_id, trust_remote_code=True,
                                                                cache_dir=self.cache_dir,
                                                                use_fast=True)
        except Exception as e:
            logging.error(f"Failed to load CLIP model: {e}")
            raise e
        return self.processor

    # 加载tokenizer
    def load_tokenizer(self):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True,
                                                           cache_dir=self.cache_dir, use_fast=True)
        except Exception as e:
            logging.error(f"Failed to load CLIP model: {e}")
            raise e
        return self.tokenizer

    def get_vector_dimension(self):
        # 首先尝试从模型配置中获取向量维度
        if self.model is not None:
            try:
                # 对于CLIP类模型，优先获取projection_dim作为输出向量维度
                if hasattr(self.model.config, 'projection_dim'):
                    # 顶层配置中的projection_dim，适用于CLIP等多模态模型
                    return self.model.config.projection_dim

                # 对于CLIP类模型，图像和文本编码器的输出都会投影到projection_dim维度
                if hasattr(self.model.config, 'vision_config'):
                    vision_config = self.model.config.vision_config
                    # 优先获取projection_dim，它是最终的输出维度
                    if hasattr(vision_config, 'projection_dim'):
                        return vision_config.projection_dim
                    elif hasattr(vision_config, 'hidden_size'):
                        return vision_config.hidden_size

                if hasattr(self.model.config, 'text_config'):
                    # 检查是否有文本配置，并从中获取维度
                    text_config = self.model.config.text_config
                    # 优先获取projection_dim，它是最终的输出维度
                    if hasattr(text_config, 'projection_dim'):
                        return text_config.projection_dim
                    elif hasattr(text_config, 'hidden_size'):
                        return text_config.hidden_size

                # 对于一般的transformer模型
                if hasattr(self.model.config, 'hidden_size'):
                    return self.model.config.hidden_size

                # 对于某些特定模型架构
                if hasattr(self.model, 'vision_model') and hasattr(self.model.vision_model, 'config'):
                    vision_model_config = self.model.vision_model.config
                    if hasattr(vision_model_config, 'projection_dim'):
                        return vision_model_config.projection_dim
                    elif hasattr(vision_model_config, 'hidden_size'):
                        return vision_model_config.hidden_size

                if hasattr(self.model, 'text_model') and hasattr(self.model.text_model, 'config'):
                    # 检查文本模型配置
                    text_model_config = self.model.text_model.config
                    if hasattr(text_model_config, 'projection_dim'):
                        return text_model_config.projection_dim
                    elif hasattr(text_model_config, 'hidden_size'):
                        return text_model_config.hidden_size

                # 如果模型有embed_dim属性
                if hasattr(self.model, 'embed_dim'):
                    return self.model.embed_dim

            except AttributeError:
                pass

        # 如果无法从模型配置中获取，则返回初始化时传入的dimension
        return self.dimension

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

    def batch_extract_image_features(self, image_paths: list[str]):
        """Extract features for a batch of images."""
        logging.debug(f"Extracting features for {len(image_paths)} images.")
        try:
            # 预处理图片
            inputs = self.processor(images=image_paths, return_tensors="pt").to(self.device)
            # 推理
            with torch.no_grad():
                # 根据模型类型选择对应的方法
                image_features = self.model.get_image_features(**inputs)
            # 归一化 (CLIP 的特征通常需要归一化)
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            # 转为列表
            return image_features.cpu().numpy().tolist()
        except Exception as e:
            import traceback
            logging.error(f"Error extracting image features: {e}")
            logging.error(traceback.format_exc())
            raise e

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
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
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

    def get_vector_dimension(self):
        return self.instance.get_vector_dimension()

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
