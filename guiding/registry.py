"""Registry for guiding distribution methods.

This module provides a simple registration system for guiding distributions.
To add a new guiding method:

1. Create a new file (e.g., my_guiding.py) implementing GuidingDistribution
2. Import and register it in this file:

   from my_guiding import MyGuidingDistribution
   register("My Method", MyGuidingDistribution, default_config={...})

The method will automatically appear in the UI dropdown.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Type

from guiding.config import get_logger

logger = get_logger("registry")


@dataclass
class GuidingMethodInfo:
    """Information about a registered guiding method."""
    name: str
    distribution_class: Type
    default_config: Dict[str, Any]
    description: str = ""
    factory: Optional[Callable] = None  # Optional factory function


# Global registry of guiding methods
_REGISTRY: Dict[str, GuidingMethodInfo] = {}
_DEFAULT_METHOD: Optional[str] = None


def register(
    name: str,
    distribution_class: Type,
    default_config: Optional[Dict[str, Any]] = None,
    description: str = "",
    is_default: bool = False,
    factory: Optional[Callable] = None
) -> None:
    """Register a guiding distribution method.
    
    Args:
        name: Display name for the method (appears in UI dropdown)
        distribution_class: The distribution class to instantiate
        default_config: Default configuration dictionary
        description: Optional description of the method
        is_default: If True, this becomes the default method
        factory: Optional factory function that creates the distribution
    """
    global _DEFAULT_METHOD
    
    if name in _REGISTRY:
        logger.warning(f"Overwriting existing guiding method: {name}")
    
    _REGISTRY[name] = GuidingMethodInfo(
        name=name,
        distribution_class=distribution_class,
        default_config=default_config or {},
        description=description,
        factory=factory
    )
    
    if is_default or _DEFAULT_METHOD is None:
        _DEFAULT_METHOD = name
    
    logger.debug(f"Registered guiding method: {name}")


def get_method_names() -> List[str]:
    """Get list of all registered method names."""
    return list(_REGISTRY.keys())


def get_method_info(name: str) -> Optional[GuidingMethodInfo]:
    """Get info for a specific method."""
    return _REGISTRY.get(name)


def get_default_method() -> Optional[str]:
    """Get the name of the default method."""
    return _DEFAULT_METHOD


def set_default_method(name: str) -> None:
    """Set the default method by name."""
    global _DEFAULT_METHOD
    if name in _REGISTRY:
        _DEFAULT_METHOD = name
    else:
        raise ValueError(f"Unknown guiding method: {name}")


def create_distribution(name: str, device: str = "cuda", **kwargs):
    """Create a distribution instance by name.
    
    Args:
        name: Name of the registered method
        device: Device to create distribution on
        **kwargs: Additional arguments passed to the config
        
    Returns:
        Instantiated distribution object
    """
    if name not in _REGISTRY:
        raise ValueError(f"Unknown guiding method: {name}. Available: {list(_REGISTRY.keys())}")
    
    info = _REGISTRY[name]
    
    # Merge default config with provided kwargs
    config_kwargs = {**info.default_config, **kwargs}
    config_kwargs['device'] = device
    
    logger.info(f"Creating guiding distribution: {name}")
    
    # Use the factory function if provided, otherwise instantiate directly
    if info.factory is not None:
        return info.factory(**config_kwargs)
    else:
        return info.distribution_class(**config_kwargs)


# =============================================================================
# REGISTER ALL GUIDING METHODS HERE
# =============================================================================

# --- VMF (von Mises-Fisher) Mixture ---
from guiding.vmf_guiding import VMFGuidingDistribution, VMFConfig

def _create_vmf(device: str = "cuda", **kwargs) -> VMFGuidingDistribution:
    """Factory function for VMF - uses VMFConfig defaults."""
    config = VMFConfig(device=device, **kwargs)
    return VMFGuidingDistribution(config)

register(
    name="VMF Mixture",
    distribution_class=VMFGuidingDistribution,
    description="Fast von Mises-Fisher mixture model (recommended for real-time)",
    is_default=True,
    factory=_create_vmf
)

# --- NIS (Neural Importance Sampling) ---
from guiding.nis_guiding import NISGuidingDistribution, NISConfig

def _create_nis(device: str = "cuda", **kwargs) -> NISGuidingDistribution:
    """Factory function for NIS - uses NISConfig defaults."""
    config = NISConfig(device=device, **kwargs)
    return NISGuidingDistribution(config)

register(
    name="NIS (Neural Importance Sampling)",
    distribution_class=NISGuidingDistribution,
    description="Normalizing flow-based importance sampling (slower but more expressive)",
    factory=_create_nis
)

# --- DF (Distribution Factorization) ---
from guiding.df_guiding import DFGuidingDistribution, DFConfig

def _create_df(device: str = "cuda", **kwargs) -> DFGuidingDistribution:
    """Factory function for DF - uses DFConfig defaults."""
    config = DFConfig(device=device, **kwargs)
    return DFGuidingDistribution(config)

register(
    name="DF (Distribution Factorization)",
    distribution_class=DFGuidingDistribution,
    description="Distribution factorization with area-preserving mapping",
    factory=_create_df
)

# --- PPG (Practical Path Guiding) ---
from guiding.ppg_guiding import PPGGuidingDistribution, PPGConfig

def _create_ppg(device: str = "cuda", **kwargs) -> PPGGuidingDistribution:
    """Factory function for PPG - uses PPGConfig defaults.
    
    Note: PPG requires scene bounding box to be set via:
        distribution.set_scene_bounds(bbox_min, bbox_max)
    after creation.
    """
    config = PPGConfig(device=device, **kwargs)
    return PPGGuidingDistribution(config)

register(
    name="PPG (Practical Path Guiding)",
    distribution_class=PPGGuidingDistribution,
    description="SD-Tree based guiding (no neural network, requires scene bounds)",
    factory=_create_ppg
)


# =============================================================================
# Add your custom guiding methods below!
# Example:
#
# from my_custom_guiding import MyCustomDistribution, MyCustomConfig
#
# def _create_my_custom(device: str = "cuda", **kwargs):
#     config = MyCustomConfig(device=device, **kwargs)
#     return MyCustomDistribution(config)
#
# register(
#     name="My Custom Method",
#     distribution_class=MyCustomDistribution,
#     description="My awesome custom guiding method",
#     factory=_create_my_custom
# )
# =============================================================================