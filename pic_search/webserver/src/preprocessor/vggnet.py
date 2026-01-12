import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_input_vgg
from tensorflow.keras.preprocessing import image
from numpy import linalg as LA
from common.const import image_size,input_shape

def vgg_extract_feat(img_path):
    """
    Extract image features using VGG16 model
    """
    try:
        # Load pre-trained VGG16 model without top classification layer
        # 确保在eager模式下运行
        model = VGG16(weights='imagenet',
                      input_shape=input_shape,
                      pooling='max',
                      include_top=False)
        
        # Load and preprocess image
        img = image.load_img(img_path, target_size=image_size)
        img_tensor = image.img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        img_tensor = preprocess_input_vgg(img_tensor)
        
        # Extract features
        # 使用eager模式进行预测
        features = model.predict(img_tensor)
        features = features.flatten()
        
        # Normalize features
        features = features / LA.norm(features)
        
        return features
    except Exception as e:
        print(f"Error extracting features from {img_path}: {e}")
        raise e