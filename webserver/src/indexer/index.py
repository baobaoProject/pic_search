import logging
import threading

from pymilvus import DataType, MilvusClient

import common
from common.config import DEFAULT_DATABASE, DEFAULT_TABLE, MILVUS_HOST, MILVUS_PORT
from common.const import index_params_map, vector_field_name

# 创建一个线程锁
client_lock = threading.Lock()
# 全局 MilvusClient 实例
_client = None


def milvus_client():
    """
    Connect to Milvus server
    """
    global _client
    try:
        # 第一层检查：如果连接已存在，直接返回，避免获取锁
        if _client is not None:
            return _client

        with client_lock:
            # 第二层检查：双重确认，防止并发穿透
            if _client is None:
                uri = f"http://{MILVUS_HOST}:{MILVUS_PORT}"
                _client = MilvusClient(uri=uri)
                databases = _client.list_databases()
                if DEFAULT_DATABASE not in databases:
                    _client.create_database(DEFAULT_DATABASE)
                    logging.info(f"Database {DEFAULT_DATABASE} created")
                    _client.close()
                # 使用指定的数据库
                _client = MilvusClient(uri=uri, db_name=DEFAULT_DATABASE)
        return _client
    except Exception as e:
        logging.error(f"Failed to connect to Milvus: {e}")
        raise e


# 定一个方法，判断表是否存在
def has_collection(table_name=DEFAULT_TABLE):
    """
    :param table_name:
    :return:
    """
    client = milvus_client()
    return client.has_collection(table_name)


# 获取集合中的元素数量
def count_rows(table_name=DEFAULT_TABLE):
    """
    :param table_name:
    :return:
    """
    client = milvus_client()
    result = client.query(table_name, "id > 0", output_fields=["count(*)"])
    logging.info(f"Count: {result}")
    # Count: data: ["{'count(*)': 128}"], extra_info: {}
    return result[0]["count(*)"]


# FLAT: 精确搜索，速度慢但精度最高
# IVF_FLAT: 倒排文件索引，速度快于FLAT
# IVF_SQ8: 压缩版本的IVF，节省内存
# HNSW: 基于图的索引，精度高、速度快
# ANNOY: 近似最近邻搜索
# 精度优先: FLAT > HNSW > IVF_FLAT
# 速度优先: GPU_IVF_PQ > GPU_IVF_FLAT > IVF_PQ > IVF_FLAT
# 内存优化: IVF_PQ > IVF_SQ8 > IVF_FLAT > IVF_FLAT > FLAT
# 大数据集: DISKANN > HNSW > IVF_PQ
# GPU可用: GPU_IVF_FLAT 或 GPU_IVF_PQ
def create_table(table_name=common.get_model_default_table(), delete_if_exists=False, embedding_index_type="IVF_FLAT"):
    """
    Create a new collection with the specified name
    """
    try:
        client = milvus_client()
        if delete_if_exists and client.has_collection(table_name):
            logging.info(f"Collection {table_name} already exists, dropping it...")
            client.drop_collection(table_name)

        # 2.6 版本 MilvusClient.create_collection 更加简化
        # 如果不提供 schema，会创建一个带有 auto_id=True 的 id 主键和指定维度的 vector 字段
        # 但为了保持与原有字段一致（包含 image_path），我们最好还是定义 schema 或者使用快速创建参数

        # 使用新 API 创建 Schema
        schema = client.create_schema(
            auto_id=True,
            enable_dynamic_field=True,
        )

        # 字段可以启动mmap_enabled=true属性，以节约内存
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        dimension = common.get_model_dimension()
        schema.add_field(field_name=vector_field_name, datatype=DataType.FLOAT_VECTOR, dim=dimension)
        schema.add_field(field_name="image_path", datatype=DataType.VARCHAR, max_length=512)

        # 准备索引参数
        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name=vector_field_name,
            index_type=embedding_index_type,
            metric_type="L2",
            params=get_params(embedding_index_type, is_search=False)
        )

        # 创建集合
        client.create_collection(
            collection_name=table_name,
            vector_field_name=vector_field_name,
            schema=schema,
            index_params=index_params
        )

        logging.info(f"Collection {table_name} created successfully")
        return True
    except Exception as e:
        logging.error(f"Error creating collection: {e}")
        raise e


def insert_vectors(table_name, vectors, image_paths):
    """
    Insert vectors to milvus collection
    """
    try:
        client = milvus_client()

        # MilvusClient insert 接收 list of dicts
        data = []
        for i in range(len(vectors)):
            data.append({
                vector_field_name: vectors[i],
                "image_path": image_paths[i]
            })

        insert_result = client.insert(collection_name=table_name, data=data)
        logging.debug(f"Successfully inserted {len(vectors)} vectors to collection: {table_name}")
        return insert_result["ids"]
    except Exception as e:
        logging.error(f"Error inserting vectors: {e}")
        raise e


def drop_collection(table_name):
    """
    Delete a collection
    """
    try:
        client = milvus_client()
        if client.has_collection(table_name):
            client.drop_collection(table_name)
            logging.info(f"Collection {table_name} dropped")
    except Exception as e:
        logging.error(f"Error dropping collection: {e}")
        raise e


def clear_collection(table_name):
    """
    Clear a collection
    """
    try:
        client = milvus_client()
        if client.has_collection(table_name):
            client.delete(table_name, filter="id > 0")
            logging.info(f"Collection {table_name} clear")
    except Exception as e:
        logging.error(f"Error clearing collection: {e}")
        raise e


def search_vectors(table_name, vectors, top_k, output_fields, embedding_index_type="IVF_FLAT"):
    """
    Search similar vectors in milvus collection
    """
    try:
        client = milvus_client()
        # 确保集合已加载
        # client.load_collection(table_name)
        results = client.search(
            collection_name=table_name,
            anns_field=vector_field_name,
            data=vectors,
            limit=top_k,
            search_params=get_params(embedding_index_type, is_search=True),
            output_fields=output_fields
        )

        # Extract image paths and distances from results
        ids = []
        distances = []
        paths = []

        for result in results:
            for hit in result:
                ids.append(hit["id"])
                distances.append(hit["distance"])
                paths.append(hit["entity"].get('image_path'))

        logging.info(f"Successfully searched in collection: {table_name}")
        return ids, distances, paths
    except Exception as e:
        logging.error(f"Error searching vectors: {e}")
        raise e


# 索引参数
def get_params(embedding_index_type, is_search=False):
    if is_search:
        return index_params_map[embedding_index_type]["search_params"]
    else:
        return index_params_map[embedding_index_type]["index_params"]
