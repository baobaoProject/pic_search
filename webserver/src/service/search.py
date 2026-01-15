import logging

from common.config import DEFAULT_TABLE, EMBEDDING_INDEX_TYPE
from indexer import index
from model import get_feature_extractor


def do_search(table_name, file_path, top_k):
    """
    Perform image similarity search
    """
    try:
        # Extract features using the model
        logging.info(f"do_search features from {file_path}")
        feat = get_feature_extractor().extract_image_features(file_path)

        output_fields = ["image_path"]
        # Search in Milvus
        ids, distances, paths = index.search_vectors(table_name or DEFAULT_TABLE, [feat], top_k, output_fields,
                                                     EMBEDDING_INDEX_TYPE)

        return paths, distances
    except Exception as e:
        logging.error(f"Error in do_search: {e}")
        return str(e), None


def do_text_search(table_name, text, top_k):
    """
    Perform text-to-image similarity search
    """
    try:
        # Extract features from text using CLIP
        logging.info(f"do_text_search features from text: {text}")
        feat = get_feature_extractor().extract_text_features(text)

        output_fields = ["image_path"]
        # Search in Milvus
        ids, distances, paths = index.search_vectors(table_name or DEFAULT_TABLE, [feat], top_k, output_fields,
                                                     EMBEDDING_INDEX_TYPE)

        return paths, distances
    except Exception as e:
        logging.error(f"Error in do_text_search: {e}")
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
