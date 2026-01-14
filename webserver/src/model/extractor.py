import logging
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2S, preprocess_input
from common.const import image_size,model_info
from common.config import MODEL_NAME

model = None

def load_model():
    """Lazy load the EfficientNetV2S model."""
    global model
    if model is None:
        logging.info("Loading EfficientNetV2S model...")
        # EfficientNetV2S: Small version, higher accuracy and faster than VGG16
        # include_top=False: Exclude the classification layer
        # pooling='avg': Global Average Pooling, results in a 1D vector (1280 dimensions)
        model = EfficientNetV2S(
                weights='imagenet',
                include_top=False,
                pooling='avg'
        )
        logging.info("EfficientNetV2S model loaded.")
    return  model

class FeatureExtractor:
    def __init__(self):
        self.model = None

    def preprocess_image(self, img_path):
        """Preprocess an image for EfficientNetV2."""
        img = image.load_img(img_path, target_size=image_size)
        img_tensor = image.img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        # EfficientNetV2 uses its own preprocess_input (typically scaling to [-1, 1] or [0, 1])
        img_tensor = preprocess_input(img_tensor)
        return img_tensor

    def extract_features(self, img_path):
        """Extract features for a single image."""
        img_tensor = self.preprocess_image(img_path)
        features = load_model().predict(img_tensor)
        
        # Flatten and normalize
        feat = features.flatten()
        norm_feat = feat / np.linalg.norm(feat)
        return norm_feat.tolist()

    def extract_batch_features(self, batch_tensors):
        """Extract features for a batch of images."""
        batch_features = load_model().predict(batch_tensors)
        
        normalized_features = []
        for feat in batch_features:
            feat = feat.flatten()
            norm_feat = feat / np.linalg.norm(feat)
            normalized_features.append(norm_feat.tolist())
            
        return normalized_features

    # 根据模型名称获取对应模型的向量维度
    def get_vector_dimension(self) -> int:
        return model_info[MODEL_NAME]["vector_dimension"]


# Create a global instance
feature_extractor = FeatureExtractor()