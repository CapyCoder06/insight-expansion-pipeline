"""
Configuration loader for the pipeline.
"""

import os
from pathlib import Path
from typing import Dict, Any


class Config:
    """Pipeline configuration loader with defaults."""

    DEFAULTS = {
        'chunking': {
            'chunk_size': 200,
            'overlap': 50,
            'use_nltk': True
        },
        'enrichment': {
            'enabled': False,
            'variations_to_generate': 8,
            'variations_to_keep': 2
        },
        'validation': {
            'semantic_similarity_threshold': 0.65,
            'confidence': {
                'valid_boost': 0.2,
                'invalid_penalty': 0.3,
                'issue_penalty': 0.1,
                'warning_penalty': 0.05
            }
        },
        'retrieval_optimization': {
            'enabled': False
        },
        'evaluation': {
            'top_k': [1, 3, 5, 10]
        },
        'retrieval': {
            'use_metadata': False,
            'weights': {
                'text': 0.4,
                'metadata_query': 0.5,
                'expanded': 0.1
            },
            'metadata_bonus': {
                'enabled': False,
                'threshold': 0.7,
                'boost_factor': 1.5
            }
        }
    }

    def __init__(self, config_path: str = None):
        """
        Initialize configuration.

        Args:
            config_path: Path to config.yaml file. If None, searches in:
                         - ./config.yaml
                         - ../config.yaml (from src/pipeline/)
        """
        self.config = self.DEFAULTS.copy()
        self.config_path = None

        if config_path:
            self.load_from_file(config_path)
        else:
            # Try to find config.yaml automatically
            possible_paths = [
                Path('config.yaml'),
                Path('../config.yaml'),
                Path('../../config.yaml')
            ]
            for path in possible_paths:
                if path.exists():
                    self.load_from_file(str(path))
                    break

    def load_from_file(self, path: str):
        """Load configuration from YAML file."""
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        try:
            import yaml
        except ImportError:
            raise ImportError(
                "PyYAML is required to load configuration. "
                "Install it with: pip install pyyaml"
            )

        with open(path_obj, 'r') as f:
            user_config = yaml.safe_load(f) or {}

        # Deep merge user config into defaults
        self._deep_merge(self.config, user_config)
        self.config_path = path_obj

    def _deep_merge(self, target: Dict, source: Dict):
        """Recursively merge source dict into target dict."""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_merge(target[key], value)
            else:
                target[key] = value

    def get(self, key_path: str, default=None):
        """
        Get config value using dot notation.

        Args:
            key_path: Dot-separated path (e.g., 'chunking.chunk_size')
            default: Default value if key not found

        Returns:
            Config value or default
        """
        keys = key_path.split('.')
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    def as_dict(self) -> Dict[str, Any]:
        """Return full configuration as dictionary."""
        return self.config.copy()

    def __repr__(self):
        return f"Config(path={self.config_path})"


# Global config instance (can be overridden)
_global_config = None


def load_config(config_path: str = None) -> Config:
    """
    Load configuration singleton.

    Args:
        config_path: Optional path to config file

    Returns:
        Config instance
    """
    global _global_config
    if _global_config is None:
        _global_config = Config(config_path)
    return _global_config


def get_config() -> Config:
    """Get the global config instance (loads default if not loaded)."""
    return load_config()
