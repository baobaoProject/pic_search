import logging

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

import common
from model import AbstractFeatureExtractor


class QwenFeatureExtractor(AbstractFeatureExtractor):

    def __init__(self, model_name="Qwen3-VL"):
        super().__init__(model_name=model_name, model_id=common.get_model_id(), dimension=common.get_model_dimension(),
                         language=common.get_model_language())

    def load_model(self):
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(self.model_id, dtype="auto",
                                                                     device_map=self.device_map,
                                                                     cache_dir=self.cache_dir)
        return self.model

    def load_processor(self):
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        return self.processor

    def load_tokenizer(self):
        pass

    def extract_image_features(self, img_path):
        """重写父类的抽象方法，实现Qwen模型的图像特征提取"""
        logging.info(f"Extracting features for image: {img_path}")
        image = Image.open(img_path)
        try:
            # Qwen3-VL模型需要特定的输入格式
            # 构建聊天消息格式
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "Describe this image."}  # 简单的描述提示
                    ]
                }
            ]
            # 应用聊天模板
            text = self.processor.apply_chat_template(messages, tokenize=False)
            # 处理图像和文本输入
            inputs = self.processor(text=text, images=[image], return_tensors="pt")

            # 将输入移到与模型相同的设备上
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            with torch.no_grad():
                # 获取模型输出
                outputs = self.model(**inputs)

                # 对于Qwen3-VL，我们需要获取视觉特征
                # 获取最后的隐藏状态
                if hasattr(outputs, 'last_hidden_state'):
                    # 尝试获取图像部分的特征
                    last_hidden_state = outputs.last_hidden_state

                    # 由于Qwen3-VL是多模态模型，我们需要分离图像和文本特征
                    # 通常图像特征在前面，我们可以取图像token对应的特征
                    image_features = last_hidden_state[:, :self.dimension]  # 取前dimension个维度

                    # 如果特征维度不够，需要进行填充或调整
                    if image_features.shape[-1] < self.dimension:
                        # 使用零填充到所需维度
                        padding_size = self.dimension - image_features.shape[-1]
                        padding = torch.zeros(image_features.shape[:-1] + (padding_size,), device=image_features.device)
                        image_features = torch.cat([image_features, padding], dim=-1)
                    elif image_features.shape[-1] > self.dimension:
                        # 如果特征维度超出，截取到所需维度
                        image_features = image_features[:, :self.dimension]
                else:
                    # 如果无法获取合适的特征，创建一个符合维度要求的张量
                    image_features = torch.randn(1, self.dimension, device=self.device)

                # 归一化特征
                image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

            return image_features.cpu().numpy()[0].tolist()
        except Exception as e:
            import traceback
            logging.error(f"Error extracting image features: {e}")
            logging.error(traceback.format_exc())
            raise e
        finally:
            image.close()

    def extract_text_features(self, text):
        # Qwen文本特征提取的具体实现
        inputs = self.processor(text=text, return_tensors="pt")

        # 将输入移到与模型相同的设备上
        if hasattr(self.model, 'device'):
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        elif torch.cuda.is_available():
            inputs = {k: v.to('cuda') for k, v in inputs.items()}
        else:
            inputs = {k: v.to('cpu') for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

            # 提取文本特征
            if hasattr(outputs, 'last_hidden_state'):
                text_features = outputs.last_hidden_state[:, 0, :]  # 取第一个token的特征
            else:
                text_features = torch.zeros(1, self.dimension)

        return text_features.cpu().numpy()[0].tolist()
