import logging
from common.config import DEFAULT_TABLE
from  indexer import  index

def do_count(table_name):
    """
    Count the number of vectors in the collection
    """
    try:
        table_name = table_name or DEFAULT_TABLE
        # Check if collection exists
        if not index.has_collection(table_name):
            logging.info(f"Collection {table_name} does not exist")
            return 0
        
        # Get collection and count entities
        count = index.count_rows(table_name)
        
        logging.info(f"Collection {table_name} has {count} entities")
        return count
    except Exception as e:
        logging.error(f"Error counting entities: {e}")
        return str(e)