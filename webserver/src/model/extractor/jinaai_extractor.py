import logging

from PIL import Image
from transformers import AutoModel

import common
from model import AbstractFeatureExtractor


class JinaaiFeatureExtractor(AbstractFeatureExtractor):

    def __init__(self, model_name="Jinaai-CLIP"):
        super().__init__(model_name=model_name, model_id=common.get_model_id(), dimension=common.get_model_dimension(),
                         language=common.get_model_language())

    def load_model(self):
        self.model = AutoModel.from_pretrained(self.model_id,
                                               cache_dir=self.cache_dir,
                                               trust_remote_code=True)
        return self.model

    def load_processor(self):
        # Jina CLIP v2 doesn't require a separate processor
        # We'll use the model's encode methods directly
        return None

    def load_tokenizer(self):
        pass

    def extract_image_features(self, img_path):
        """重写父类的抽象方法，实现Jina CLIP模型的图像特征提取"""
        logging.info(f"Extracting features for image: {img_path}")
        image = Image.open(img_path)
        try:
            # 使用Jina CLIP v2的encode_image方法
            # 设置truncate_dim为配置的维度
            image_embeddings = self.model.encode_image([image], truncate_dim=self.dimension)

            # 转换为numpy数组并返回第一个（也是唯一的）图像的特征
            features = image_embeddings[0].detach().cpu().numpy().tolist()

            # 确保特征维度正确
            if len(features) != self.dimension:
                # 如果维度不匹配，调整到目标维度
                if len(features) > self.dimension:
                    features = features[:self.dimension]
                else:
                    # 如果维度不足，用0填充
                    features.extend([0.0] * (self.dimension - len(features)))

            return features
        except Exception as e:
            import traceback
            logging.error(f"Error extracting image features: {e}")
            logging.error(traceback.format_exc())
            raise e
        finally:
            image.close()

    def batch_extract_image_features(self, image_paths):
        features_list = []
        images = []

        # 打开所有图像
        for img_path in image_paths:
            image = Image.open(img_path)
            images.append(image)

        try:
            # 批量处理图像
            image_embeddings = self.model.encode_image(images, truncate_dim=self.dimension)

            # 转换每个图像的特征
            for i in range(len(images)):
                features = image_embeddings[i].detach().cpu().numpy().tolist()

                # 确保特征维度正确
                if len(features) != self.dimension:
                    if len(features) > self.dimension:
                        features = features[:self.dimension]
                    else:
                        features.extend([0.0] * (self.dimension - len(features)))

                features_list.append(features)
        except Exception as e:
            import traceback
            logging.error(f"Error batch extracting image features: {e}")
            logging.error(traceback.format_exc())
            raise e
        finally:
            # 关闭所有图像
            for image in images:
                image.close()

        return features_list

    def extract_text_features(self, text):
        """Jina CLIP文本特征提取的具体实现"""
        logging.info(f"Extracting features for text: {text}")

        # 使用Jina CLIP v2的encode_text方法
        # 使用retrieval.document任务类型作为默认选项
        text_embeddings = self.model.encode_text([text], task='retrieval.document', truncate_dim=self.dimension)

        # 转换为numpy数组并返回第一个（也是唯一的）文本的特征
        features = text_embeddings[0].detach().cpu().numpy().tolist()

        # 确保特征维度正确
        if len(features) != self.dimension:
            if len(features) > self.dimension:
                features = features[:self.dimension]
            else:
                features.extend([0.0] * (self.dimension - len(features)))

        return features
