from typing import Optional

from xc_cache.models import infer_model_type
from xc_cache.inference.abstract_lm_interface import AbstractLMInterface
from xc_cache.inference.reader_interfaces import CrossAttnInterface, FiDInterface

def get_interface(
    *,
    model_path: Optional[str] = None,
    model_ckpt: Optional[str] = None,
    model_type: Optional[str] = None,
    **kwargs,
) -> AbstractLMInterface:
    """Detect the correct interface to instantiate and return it."""
    if model_type is None:
        model_type = infer_model_type(model_path=model_path, model_ckpt=model_ckpt)

    if model_type in {"gptbigcode", "llama", "mistral", "tulu"}:
        return CrossAttnInterface(model_path=model_path, model_ckpt=model_ckpt, **kwargs)
    elif model_type == "fid":
        assert model_ckpt is None
        return FiDInterface(model_path=model_path, **kwargs)

    raise NotImplementedError(f"Unknown model interface for model_type={model_type}")
