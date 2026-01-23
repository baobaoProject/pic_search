from .config import BATCH_SIZE, DEFAULT_DATABASE, DEFAULT_TABLE, DEVICE, INPUT_SHAPE_SIZE, MAX_THREADS, \
    MILVUS_HOST, MILVUS_PORT, MODEL_LANGUAGE, MODEL_NAME, UPLOAD_PATH
from .const import image_size, input_shape, model_info


class ModelConfig:
    def __init__(self,
                 model_name,
                 model_language,
                 model_id,
                 model_type,
                 model_checkpoints,
                 model_input_shape_size,
                 model_vector_dimension):
        self.model_name = model_name
        self.model_language = model_language
        self.model_id = model_id
        self.model_type = model_type
        self.model_checkpoints = model_checkpoints
        self.model_input_shape_size = model_input_shape_size
        self.model_vector_dimension = model_vector_dimension


# 获取模型名称
def get_model_name():
    return MODEL_NAME


# 获取向量维度
def get_model_dimension(model_name=MODEL_NAME):
    """根据模型类型获取向量维度"""
    return model_info[model_name]["vector_dimension"] or 512


# 获取模型id
def get_model_id(model_name=MODEL_NAME):
    """根据模型类型获取模型id"""
    return model_info[model_name]["model_id"] or ""


def get_model_checkpoints(model_name=MODEL_NAME):
    """根据模型类型获取模型id"""
    return model_info[model_name]["checkpoints"] or []


# 获取设备类型
def get_device():
    return DEVICE or "cpu"


# 获取图片尺寸
def get_image_shape(model_name=MODEL_NAME):
    return model_info[model_name]["input_shape_size"] or INPUT_SHAPE_SIZE


# 获取模型的语言
def get_model_language():
    return MODEL_LANGUAGE


# 获取模型默认的表
def get_model_default_table():
    """获取默认的表名"""
    return DEFAULT_TABLE


# 获取模型类型
def get_model_type(model_name=MODEL_NAME):
    return model_info[model_name]["type"] or "ofa-sys"


def get_model_config(model_name=MODEL_NAME):
    """根据模型名称获取模型配置"""
    return ModelConfig(
        model_name=model_name,
        model_language=get_model_language(),
        model_id=get_model_id(model_name),
        model_type=get_model_type(model_name),
        model_checkpoints=get_model_checkpoints(model_name),
        model_input_shape_size=get_image_shape(model_name),
        model_vector_dimension=get_model_dimension(model_name)
    )
