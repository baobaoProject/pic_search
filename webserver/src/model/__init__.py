import common
from .ModelExtractor import ProxyFeatureExtractor
# 使用 Lazy Import 避免在模块级别导入具体的 Extractor 模块
# 这对于防止在 Gunicorn Master 进程中初始化 CUDA 至关重要

def get_feature_extractor() -> ProxyFeatureExtractor:
    """根据配置获取对应的特征提取器"""
    # MODEL_NAME以CLIP开头
    return ProxyFeatureExtractor.get_instance(common.get_model_name())