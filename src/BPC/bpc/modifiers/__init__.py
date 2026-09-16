def get_modifier(method: str, model_structure: str = None):

    if method == "origin":
        from .modifiers_untils import Origin
        return model_structure, Origin

    elif method == 'sparse':
        from .modifiers_untils import Sparse
        return model_structure, Sparse
    
    elif method == 'sparsetopk':
        from .modifiers_untils import SparseTopk
        return model_structure, SparseTopk
    
    elif method == 'snapkv':
        from .modifiers_untils import SnapKV
        return model_structure, SnapKV
    
    elif method == 'pyramidkv':
        from .modifiers_untils import PyramidKV
        return model_structure, PyramidKV
    
    elif method == 'cakekv':
        from .modifiers_untils import CakeKV
        return model_structure, CakeKV
    
    elif method == 'compresskv':
        from .modifiers_untils import CompressKV
        return model_structure, CompressKV
    
    elif method == 'streamingllm':
        from .modifiers_untils import StreamingLLM
        return model_structure, StreamingLLM
    
    elif method == 'spotlight':
        from .modifiers_untils import Spotlight
        return model_structure, Spotlight
    
    elif method == 'quest':
        from .modifiers_untils import Quest
        return model_structure, Quest

    elif method == 'rocketkv':
        from .modifiers_untils import RocketKV
        return model_structure, RocketKV

    elif method == 'clusterkv':
        from .modifiers_untils import ClusterKV
        return model_structure, ClusterKV

    elif method == 'magicpig':
        from .modifiers_untils import MagicPIG
        return model_structure, MagicPIG

    elif method == 'bpc':
        from .modifiers_untils import BPC
        return model_structure, BPC
    
    elif method == 'topk':
        from .modifiers_untils import TopK
        return model_structure, TopK
    else:
        raise NotImplementedError(method, model_structure)