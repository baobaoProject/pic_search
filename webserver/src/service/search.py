import logging

from common.config import DEFAULT_TABLE,EMBEDDING_INDEX_TYPE
from  indexer import  index
from model.extractor import feature_extractor

def do_search(table_name, file_path, top_k):
    """
    Perform image similarity search
    """
    try:
        # Extract features using the model
        feat = feature_extractor.extract_features(file_path)
        
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