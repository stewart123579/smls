from .tensorflow import TFSentenceEncoderBuilder
from .ml_services import MLServiceProvider


ml_services = MLServiceProvider()
ml_services.register_builder("TFSentenceEncoder", TFSentenceEncoderBuilder())
