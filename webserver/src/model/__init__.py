from common.config import MODEL_NAME

# 使用 Lazy Import 避免在模块级别导入具体的 Extractor 模块
# 这对于防止在 Gunicorn Master 进程中初始化 CUDA 至关重要

def get_feature_extractor():
    """根据配置获取对应的特征提取器"""
    if MODEL_NAME == "CLIP":
        from .clip_extractor import get_feature_extractor as get_clip_feature_extractor
        return get_clip_feature_extractor()
    else:
        return None

# 为了保持向后兼容，如果需要FeatureExtractor类，也应该动态获取
# 但由于这通常用于类型检查或实例化，我们尽量避免在模块级别导出它，
# 或者仅在确实需要时才导入。
# 这里我们定义一个占位符或代理，或者干脆移除对 FeatureExtractor 的直接导出，
# 强迫使用者使用 get_feature_extractor()。
# 为了最大程度减少对其他文件的影响（虽然我已经修改了 search.py 和 train.py），
# 我将不再导出 FeatureExtractor 类。