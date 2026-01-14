from .config import INPUT_SHAPE_SIZE,N_LIST,SEARCH_N_LIST,MODEL_NAME

# 输入图片的尺寸
input_shape=(int(INPUT_SHAPE_SIZE), int(INPUT_SHAPE_SIZE), 3)
image_size=(int(INPUT_SHAPE_SIZE),int(INPUT_SHAPE_SIZE))

# 向量字段名，表示向量的字段名为 embedding
vector_field_name = "embedding"

# 定义模型的基础信息
model_info = {
    "VGG16":{
        "vector_dimension": 512,
        "input_shape_size": 224,
    },
    "EfficientNetV2S":{
        "vector_dimension": 1280,
        "input_shape_size": INPUT_SHAPE_SIZE,
    }
}

# 定义索引类型和对应的参数映射
index_params_map = {
    "IVF_FLAT": { # 索引类型为 IVF_FLAT
        "index_params": {"nlist": N_LIST},  # 在建立索引时使用 k-means 算法创建的簇数，默认值128，值越大，通过创建更精细的簇来提高召回率
        "search_params": {"nprobe": SEARCH_N_LIST}  # 搜索候选集群的集群数，默认值8，值越大，搜索速度越慢，但召回率越高
    },
    "IVF_SQ8": { # 一种基于量化的索引算法，旨在解决大规模相似性搜索难题。与穷举搜索方法相比，这种索引类型的搜索速度更快，占用内存更少
        "index_params": {
            "nlist": N_LIST,  # 在建立索引时使用 k-means 算法创建的簇数，默认值128，值越大，通过创建更精细的簇来提高召回率
        },
        "search_params": {
            "nprobe": SEARCH_N_LIST, # 搜索候选集群的集群数，默认值8，值越大，搜索速度越慢，但召回率越高
        }
    },
    "IVF_PQ": { # 一种基于量化的索引算法，用于高维空间中的近似近邻搜索
        "index_params": {
            "nlist": N_LIST,  # 在建立索引时使用 k-means 算法创建的簇数，默认值128，值越大，通过创建更精细的簇来提高召回率
            "m": 32, # 在量化过程中将每个高维向量分成的子向量数,向量编码的维度，m 必须是向量维数(D) 的除数
            "nbits": 8, # 用于以压缩形式表示每个子向量中心点索引的比特数,向量编码的位数,整数整数[1, 24]，默认值8,
        },
        "search_params": {
            "nprobe": SEARCH_N_LIST, # 搜索候选集群的集群数，默认值8，值越大，搜索速度越慢，但召回率越高
        }
    },
    "HNSW": {  # 索引类型为 HNSW
        "index_params": {
            "M": N_LIST,  # 图中每个节点可拥有的最大连接数
            "efConstruction": 360  # 索引构建过程中考虑连接的候选邻居数量
        },
        "search_params": {
            "params": {
                "ef": SEARCH_N_LIST,  # 控制近邻检索时的搜索范围
            }
        }
    },
    "HNSW_SQ": { # 将层次导航小世界（HNSW）图与标量量化（SQ）相结合，创建了一种先进的向量索引方法，提供了可控的大小与精度权衡
        "index_params": {
            "M": N_LIST,
            "efConstruction": 360,
            "sq_type": "SQ6", # 指定用于压缩向量的标量量化方法:范围[SQ4U,SQ6,SQ8,BF16,FP16 ]
            "refine":True, # 计算查询向量和候选向量之间的精确距离对初始结果进行重新排序
            "refine_type": "SQ8", # 决定用于细化的数据精度
        },
        "search_params": {
            "params": {
                "ef": SEARCH_N_LIST,
                "refine_k":1 # 放大系数，用于控制相对于请求的前 K 个结果
            }
        }
    },
    "DISKANN": { # 一种基于磁盘的方法，可以在数据集大小超过可用 RAM 时保持较高的搜索精度和速度,需要启用参数：queryNode.enableDisk
        # 只能通过 Milvus 配置文件进行配置 (milvus.yaml)
    },
    "SCANN":{ # 即使数据集越来越大、越来越复杂，也能在高维空间中高效地找到最相关的向量
        "index_params": {
            "nlist": N_LIST,  # 在建立索引时使用 k-means 算法创建的簇数，默认值128，值越大，通过创建更精细的簇来提高召回率
            "with_raw_data": True, # 是否在量化表示的同时存储原始向量数据
        },
        "search_params": {
            "params": {
                "n_probes": SEARCH_N_LIST, # 搜索时使用的探针数量
                "reorder_k":10 # 控制在重新排序阶段精炼的候选向量数量
            }
        }
    }
}