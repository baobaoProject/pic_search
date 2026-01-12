import logging
from common.config import DEFAULT_TABLE
from  indexer import  index

def do_delete(table_name):
    """
    Delete the collection
    """
    try:
        table_name = table_name or DEFAULT_TABLE
        index.delete_table(table_name)
        logging.info(f"Collection {table_name} deleted successfully")
        return f"Collection {table_name} deleted successfully"
    except Exception as e:
        logging.error(f"Error deleting collection: {e}")
        return str(e)