import logging

import numpy as np
from tensorflow.keras.preprocessing import image

from common.config import DEFAULT_TABLE,EMBEDDING_INDEX_TYPE
from common.const import image_size
from  indexer import  index
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_input_vgg

def do_search(table_name, file_path, top_k, model):
    """
    Perform image similarity search
    """
    try:
        # Load and preprocess image

        img = image.load_img(file_path, target_size=image_size)
        img_tensor = image.img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        
        # Extract features using the model

        img_tensor = preprocess_input_vgg(img_tensor)
        
        feat = model.predict(img_tensor)
        
        # Normalize the feature vector
        feat = feat.flatten() / np.linalg.norm(feat)
        feat = feat.tolist()
        output_fields = ["image_path"]
        # Search in Milvus
        ids, distances, paths = index.search_vectors(table_name or DEFAULT_TABLE, [feat], top_k,output_fields,EMBEDDING_INDEX_TYPE)
        
        return paths, distances
    except Exception as e:
        logging.error(f"Error in do_search: {e}")
        return str(e), None


def query_name_from_ids(table_name, ids):
    """
    Query image names from IDs
    """
    try:
        # This would typically retrieve image paths from the database
        # For now, returning the IDs as is
        return ids
    except Exception as e:
        logging.error(f"Error in query_name_from_ids: {e}")
        return str(e)