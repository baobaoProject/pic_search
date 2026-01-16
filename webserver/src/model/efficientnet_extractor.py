import logging

import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2S, preprocess_input

import common
from common.const import image_size
from model.ModelExtractor import ProxyFeatureExtractor


def preprocess_image(img_path):
    """Preprocess an image for EfficientNetV2."""
    img = image.load_img(img_path, target_size=image_size)
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    # EfficientNetV2 uses its own preprocess_input (typically scaling to [-1, 1] or [0, 1])
    img_tensor = preprocess_input(img_tensor)
    return img_tensor


class EfficientNetFeatureExtractor(ProxyFeatureExtractor):

    def __init__(self, model_name="EfficientNetV2S"):
        super().__init__(model_name=model_name, model_id=common.get_model_id(), dimension=common.get_model_dimension(), language=common.get_model_language())

    def load_model(self):
        """Lazy load the EfficientNetV2S model."""
        if self.model is None:
            logging.info("Loading EfficientNetV2S model...")
            # EfficientNetV2S: Small version, higher accuracy and faster than VGG16
            # include_top=False: Exclude the classification layer
            # pooling='avg': Global Average Pooling, results in a 1D vector (1280 dimensions)
            self.model = EfficientNetV2S(weights='imagenet',include_top=False,pooling='avg')
            logging.info("EfficientNetV2S model loaded.")
        return self.model

    def load_processor(self):
        pass

    def load_tokenizer(self):
        pass

    def extract_image_features(self, img_path):
        """Extract features for a single image."""
        img_tensor = preprocess_image(img_path)
        features = self.model.predict(img_tensor)

        # Flatten and normalize
        feat = features.flatten()
        norm_feat = feat / np.linalg.norm(feat)
        return norm_feat.tolist()

    def extract_text_features(self, text):
        pass

    def batch_extract_image_features(self, batch_tensors):
        """Extract features for a batch of images."""
        batch_features = self.model.predict(batch_tensors)

        normalized_features = []
        for feat in batch_features:
            feat = feat.flatten()
            norm_feat = feat / np.linalg.norm(feat)
            normalized_features.append(norm_feat.tolist())

        return normalized_features