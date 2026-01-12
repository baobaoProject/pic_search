import threading
from concurrent.futures import ThreadPoolExecutor
import logging


def thread_runner(max_workers, func, *args):
    """
    Run function in thread pool
    """
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future = executor.submit(func, *args)
            result = future.result()
            return result
    except Exception as e:
        logging.error(f"Error in thread_runner: {e}")
        raise e