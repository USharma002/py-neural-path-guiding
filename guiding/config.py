"""Centralized configuration for the neural path guiding system."""
from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional, Callable

import numpy as np
import torch

# Try to import numba for JIT compilation of numpy operations
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Provide a no-op decorator if numba is not available
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range


# =============================================================================
# Performance Optimizations
# =============================================================================

def get_optimal_torch_settings() -> None:
    """Configure PyTorch for optimal performance."""
    # Enable TF32 for faster matmul on Ampere+ GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Enable cudnn benchmark for consistent input sizes
    torch.backends.cudnn.benchmark = True
    
    # Disable debug/profiling overhead
    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(False)
    torch.autograd.profiler.emit_nvtx(False)


# Numba-accelerated image processing functions
@jit(nopython=True, parallel=True, cache=True, fastmath=True)
def fast_tone_map_and_gamma(rgb: np.ndarray, gamma: float = 2.2) -> np.ndarray:
    """Combined tone mapping and gamma correction for efficiency."""
    h, w, c = rgb.shape
    inv_gamma = 1.0 / gamma
    out = np.empty_like(rgb)
    for i in prange(h):
        for j in range(w):
            for k in range(c):
                val = rgb[i, j, k]
                # Reinhard tone map
                val = val / (1.0 + val)
                # Clamp and gamma correct
                val = max(0.0, min(1.0, val))
                out[i, j, k] = val ** inv_gamma
    return out


@jit(nopython=True, parallel=True, cache=True)
def fast_float_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Fast conversion from float [0,1] to uint8 [0,255]."""
    h, w, c = arr.shape
    out = np.empty((h, w, c), dtype=np.uint8)
    for i in prange(h):
        for j in range(w):
            for k in range(c):
                val = arr[i, j, k]
                val = max(0.0, min(1.0, val)) * 255.0
                out[i, j, k] = int(val + 0.5)  # Round
    return out


# =============================================================================
# Logging Configuration
# =============================================================================

def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
    optimize_torch: bool = True
) -> logging.Logger:
    """Set up and return the root logger for the application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path to write logs to
        format_string: Optional custom format string
        optimize_torch: Whether to apply PyTorch performance optimizations
        
    Returns:
        Configured logger instance
    """
    # Apply PyTorch optimizations early
    if optimize_torch:
        get_optimal_torch_settings()
    if format_string is None:
        format_string = "[%(asctime)s] %(levelname)s - %(name)s - %(message)s"
    
    # Create formatter
    formatter = logging.Formatter(format_string, datefmt="%H:%M:%S")
    
    # Get root logger for the package
    logger = logging.getLogger("npg")
    logger.setLevel(level)
    
    # Avoid duplicate handlers
    if not logger.handlers:
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler (optional)
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name under the npg namespace.
    
    Args:
        name: Logger name (will be prefixed with 'npg.')
        
    Returns:
        Logger instance
    """
    return logging.getLogger(f"npg.{name}")


# =============================================================================
# Configuration Dataclasses
# =============================================================================

@dataclass
class NetworkConfig:
    """Configuration for neural network architecture."""
    
    # Number of vMF lobes in the mixture
    num_lobes: int = 8
    
    # Hash grid encoding settings
    hash_grid_levels: int = 16
    hash_grid_features_per_level: int = 2
    hash_grid_log2_size: int = 19
    hash_grid_base_resolution: int = 16
    hash_grid_per_level_scale: float = 1.5
    
    # MLP settings
    mlp_hidden_layers: int = 4
    mlp_neurons: int = 64
    
    # NRC-specific settings
    nrc_hidden_layers: int = 8
    nrc_neurons: int = 128


@dataclass
class TrainingConfig:
    """Configuration for training hyperparameters."""
    
    # Learning rates
    guiding_learning_rate: float = 1e-3
    nrc_learning_rate: float = 1e-3
    
    # Optimizer settings
    lr_scheduler_gamma: float = 0.9
    
    # Loss function settings
    importance_clipping_threshold: float = 10.0
    importance_weight_max: float = 100.0
    kappa_regularization_strength: float = 0.001
    
    # Gradient clipping
    gradient_clip_value: float = 1.0
    gradient_clip_norm: float = 1.0
    
    # Numerical stability
    epsilon: float = 1e-5


@dataclass
class IntegratorConfig:
    """Configuration for the path guiding integrator."""
    
    # Path tracing settings
    max_depth: int = 5
    rr_depth: int = 5
    
    # MIS settings (0.5 = equal weight between BSDF and guiding)
    bsdf_sampling_fraction: float = 0.5
    
    # NRC settings
    nrc_query_probability: float = 0.8
    nrc_min_depth: int = 2
    
    # Store NEE radiance for training
    store_nee_radiance: bool = False


@dataclass
class RenderConfig:
    """Configuration for rendering settings."""
    
    # Resolution
    width: int = 256
    height: int = 256
    
    # Samples per pixel
    spp: int = 1
    
    # Device
    device: str = "cuda"
    
    # Performance flags
    enable_loop_recording: bool = False
    enable_vcall_recording: bool = False


@dataclass
class VisualizerConfig:
    """Configuration for the visualization application."""
    
    # Main render resolution
    main_resolution: tuple[int, int] = (256, 256)
    
    # Hover/inspection resolution
    hover_resolution: tuple[int, int] = (256, 256)
    
    # Ground truth render SPP
    gt_spp: int = 128
    
    # Real-time render settings
    realtime_interval_ms: int = 100
    
    # FPS display settings
    fps_update_interval: float = 0.5  # seconds
    fps_smoothing: float = 0.9  # exponential moving average factor


@dataclass
class Config:
    """Master configuration containing all sub-configurations."""
    
    network: NetworkConfig = field(default_factory=NetworkConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    integrator: IntegratorConfig = field(default_factory=IntegratorConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    visualizer: VisualizerConfig = field(default_factory=VisualizerConfig)
    
    # Logging settings
    log_level: int = logging.INFO
    log_file: Optional[str] = None
    
    def __post_init__(self):
        """Initialize logging after config is created."""
        setup_logging(self.log_level, self.log_file)


# Global config instance (can be overridden)
_global_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance.
    
    Returns:
        The global Config instance, creating a default one if needed.
    """
    global _global_config
    if _global_config is None:
        _global_config = Config()
    return _global_config


def set_config(config: Config) -> None:
    """Set the global configuration instance.
    
    Args:
        config: The Config instance to use globally.
    """
    global _global_config
    _global_config = config


# =============================================================================
# Performance Profiler
# =============================================================================

class FPSProfiler:
    """Simple FPS profiler measuring actual frame render time."""
    
    def __init__(
        self, 
        update_interval: float = 0.5,
        smoothing: float = 0.9
    ):
        """Initialize the FPS profiler."""
        self._frame_start_time = 0.0
        self._current_frame_time_ms = 0.0
        self._fps = 0.0
        
    def begin_frame(self) -> None:
        """Call at the START of each frame."""
        self._frame_start_time = time.perf_counter()
    
    def end_frame(self) -> None:
        """Call at the END of each frame to calculate FPS."""
        elapsed = time.perf_counter() - self._frame_start_time
        if elapsed > 0:
            self._current_frame_time_ms = elapsed * 1000.0
            self._fps = 1.0 / elapsed
    
    def tick(self) -> None:
        """Legacy method - calls end_frame then begin_frame for compatibility."""
        if self._frame_start_time > 0:
            self.end_frame()
        self.begin_frame()
    
    @property
    def fps(self) -> float:
        """Get the FPS based on last frame time."""
        return self._fps
    
    @property
    def frame_time_ms(self) -> float:
        """Get the last frame time in milliseconds."""
        return self._current_frame_time_ms
    
    def get_stats_string(self) -> str:
        """Get a formatted string with FPS and frame time stats."""
        if self._fps == 0:
            return "FPS: --"
        return f"FPS: {self._fps:.1f} ({self._current_frame_time_ms:.1f}ms)"
    
    def reset(self) -> None:
        """Reset the profiler."""
        self._frame_start_time = 0.0
        self._current_frame_time_ms = 0.0
        self._fps = 0.0


# Initialize default config and logging on module import
_default_config = Config()
