__version__ = "1.0.0"
__author__ = "Cache4Diffusion Team"

from .cache_functions import *
from .forwards import *
from .npu_utils import *

__all__ = [
    "cache_init",
    "cal_type", 
    "qwen_taylorseer_forward",
    "qwen_transformer_forward",
    "NPUOptimizer",
    "NPUMemoryManager"
]

