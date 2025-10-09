import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path


class ConfigManager:
    """Manages system configuration and settings"""
    
    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "settings.yaml"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            return self._get_default_config()
        
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"Warning: Failed to load config from {self.config_path}: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "web_search": {
                "enabled": True,
                "max_results_per_query": 10,
                "timeout_seconds": 30,
                "rate_limit_delay": 1.0,
                "user_agent": "Catalog-Auto-Tagger/1.0",
                "search_engines": ["google", "bing"]
            },
            "scraping": {
                "enabled": True,
                "timeout_seconds": 30,
                "max_page_size_mb": 5,
                "respect_robots_txt": True,
                "max_concurrent_requests": 5,
                "delay_between_requests": 1.0
            },
            "ml_models": {
                "text_classification": {
                    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                    "cache_dir": "./models/cache",
                    "device": "auto"
                },
                "openai": {
                    "enabled": False,
                    "model": "gpt-3.5-turbo",
                    "max_tokens": 1000,
                    "temperature": 0.1
                }
            },
            "processing": {
                "batch_size": 10,
                "max_concurrent_products": 3,
                "retry_attempts": 3,
                "cache_results": True,
                "cache_ttl_hours": 24
            },
            "output": {
                "format": "json",
                "include_reasoning": True,
                "min_confidence_threshold": 0.3,
                "max_tags_per_product": 20
            },
            "logging": {
                "level": "INFO",
                "file": "./logs/catalog_tagger.log",
                "max_file_size_mb": 10,
                "backup_count": 5
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_api_key(self, service: str) -> Optional[str]:
        """Get API key from environment variables"""
        env_var_map = {
            "google": "GOOGLE_SEARCH_API_KEY",
            "openai": "OPENAI_API_KEY",
            "serp": "SERPAPI_KEY",
            "bing": "BING_SEARCH_API_KEY"
        }
        
        env_var = env_var_map.get(service.lower())
        if env_var:
            return os.getenv(env_var)
        
        return None
    
    def save_config(self) -> None:
        """Save current configuration to file"""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
    
    def update(self, key: str, value: Any) -> None:
        """Update configuration value using dot notation"""
        keys = key.split('.')
        config_ref = self.config
        
        for k in keys[:-1]:
            if k not in config_ref:
                config_ref[k] = {}
            config_ref = config_ref[k]
        
        config_ref[keys[-1]] = value