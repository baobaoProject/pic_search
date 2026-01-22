import common
from model import AbstractFeatureExtractor


class GoogleFeatureExtractor(AbstractFeatureExtractor):

    def __init__(self, model_name="GoogleSiglip2"):
        super().__init__(model_name=model_name, model_id=common.get_model_id(), dimension=common.get_model_dimension(),
                         language=common.get_model_language())
