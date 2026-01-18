import os

# 当前默认使用的中文模型
MODEL_NAME = os.getenv("MODEL_NAME", "JinaCLIP")
# 设备类型cpu/gpu
DEVICE = os.getenv("DEVICE", "cuda")
MILVUS_HOST = os.getenv("MILVUS_HOST", "127.0.0.1")
MILVUS_PORT = os.getenv("MILVUS_PORT", 19530)
# 支持的语言，默认CN
MODEL_LANGUAGE = os.getenv("MODEL_LANGUAGE", "CN")
# 默认的数据库
DEFAULT_DATABASE = os.getenv("DEFAULT_DATABASE", "pic_search")
DATA_PATH = os.getenv("DATA_PATH", "/data/images")
DEFAULT_TABLE = os.getenv("DEFAULT_TABLE", "milvus")
UPLOAD_PATH = "/tmp/search-images"
# 从环境变量获取输入的图片尺寸，默认为 224
INPUT_SHAPE_SIZE = os.getenv("INPUT_SHAPE_SIZE", 224)
# 最大线程数
MAX_THREADS = int(os.getenv("MAX_THREADS", 5))
# 批量预处理的图片数量
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 50))
# 索引类型 embedding
EMBEDDING_INDEX_TYPE = os.getenv("EMBEDDING_INDEX_TYPE", "IVF_FLAT")
# 聚族中心个数
N_LIST = int(os.getenv("N_LIST", 128))
# 搜索时查询的聚类数量
SEARCH_N_LIST = int(os.getenv("SEARCH_N_LIST", 32))
