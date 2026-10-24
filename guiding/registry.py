"""Registry of guiding methods shown in the UI.

A method is a PathGuidingIntegrator subclass that binds a GuidingDistribution and its
config. To add one, define the subclass next to the distribution and register it below.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Type

import mitsuba as mi

from guiding.config import get_logger

logger = get_logger("registry")

_REGISTRY: Dict[str, Type] = {}
_DEFAULT_METHOD: Optional[str] = None


def register(name: str, integrator_class: Type, is_default: bool = False) -> None:
    global _DEFAULT_METHOD
    if name in _REGISTRY:
        logger.warning(f"Overwriting existing guiding method: {name}")
    _REGISTRY[name] = integrator_class
    if is_default or _DEFAULT_METHOD is None:
        _DEFAULT_METHOD = name


def get_method_names() -> List[str]:
    return list(_REGISTRY.keys())


def get_default_method() -> Optional[str]:
    return _DEFAULT_METHOD


def get_integrator_class(name: str) -> Type:
    return _REGISTRY[name]


def create_integrator(name: str, props: Optional[mi.Properties] = None, **kwargs):
    """Instantiate a registered method's integrator; kwargs go to its constructor (e.g. nrc_system)."""
    if name not in _REGISTRY:
        raise ValueError(f"Unknown guiding method: {name}. Available: {list(_REGISTRY.keys())}")
    logger.info(f"Creating guiding integrator: {name}")
    return _REGISTRY[name](props if props is not None else mi.Properties(), **kwargs)


from guiding.vmf_guiding import VMFIntegrator
from guiding.nis_guiding import NISIntegrator
from guiding.df_guiding import DFIntegrator

register("VMF Mixture", VMFIntegrator, is_default=True)
register("NIS (Neural Importance Sampling)", NISIntegrator)
register("DF (Distribution Factorization)", DFIntegrator)
