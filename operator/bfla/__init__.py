"""Block-Filtered Long-Context Attention: framework-independent prefill."""
from .config import BFLAConfig
from .geometry import HeadLayout
from .capabilities import SupportResult, supports

__all__ = ['BFLAConfig', 'HeadLayout', 'SupportResult', 'supports',
           'BFLAPrefillPlan', 'IndexPoolState']
__version__ = '0.1.1'


def __getattr__(name):
    # Configuration and capability queries do not load torch, CUDA or a framework.
    if name == 'BFLAPrefillPlan':
        from .prepared import BFLAPrefillPlan
        return BFLAPrefillPlan
    if name == 'IndexPoolState':
        from .state import IndexPoolState
        return IndexPoolState
    raise AttributeError(name)
