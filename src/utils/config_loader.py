#!/usr/bin/env python3
"""
Configuration loader for the catalog auto-tagger system
Loads tag definitions and settings from YAML files
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Load configuration from YAML files"""

    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)

    def load_tags(self, catalog_type: str) -> Dict[str, Any]:
        """Load tag definitions from YAML configuration"""
        if not catalog_type:
            raise ValueError("catalog_type must be specified")

        try:
            tag_file = self.config_dir / "tags" / f"{catalog_type}_tags.yaml"

            if not tag_file.exists():
                logger.error(f"Tag configuration file not found: {tag_file}")
                logger.error(
                    f"💡 Create this file or use an existing one like 'real_estate'"
                )
                logger.error(f"📋 Available examples in config/tags/:")

                # List available tag files
                tag_dir = self.config_dir / "tags"
                if tag_dir.exists():
                    available_files = list(tag_dir.glob("*_tags.yaml"))
                    for file in available_files:
                        logger.error(f"   - {file.stem.replace('_tags', '')}")

                raise FileNotFoundError(
                    f"No tag configuration found for catalog_type '{catalog_type}'"
                )

            with open(tag_file, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            # Convert YAML structure to the format expected by the taggers
            tags = {}

            for section_name, section_tags in config.items():
                if isinstance(section_tags, dict):
                    for tag_name, tag_config in section_tags.items():
                        # Convert YAML format to internal format
                        internal_config = {
                            "keywords": tag_config.get("keywords", []),
                            "category": tag_config.get("category", "general"),
                            "weight": tag_config.get("weight", 1.0),
                        }

                        # Add matching_mode if it exists
                        if "matching_mode" in tag_config:
                            internal_config["matching_mode"] = tag_config["matching_mode"]

                        # Add patterns if they exist
                        if "patterns" in tag_config:
                            internal_config["patterns"] = tag_config["patterns"]

                        # Add price range if it exists
                        if "price_range" in tag_config:
                            price_range = tag_config["price_range"]
                            min_price = price_range.get("min", 0)
                            max_price = price_range.get("max")
                            if max_price is None:
                                max_price = float("inf")
                            internal_config["price_range"] = (min_price, max_price)

                        tags[tag_name] = internal_config

            if not tags:
                raise ValueError(f"No valid tags found in {tag_file}")

            logger.info(f"✅ Loaded {len(tags)} tags from {tag_file}")
            return tags

        except Exception as e:
            logger.error(f"Failed to load tag configuration for '{catalog_type}': {e}")
            raise

    def load_settings(self) -> Dict[str, Any]:
        """Load system settings from YAML configuration"""
        try:
            settings_file = self.config_dir / "settings.yaml"

            if not settings_file.exists():
                logger.warning(f"Settings file not found: {settings_file}")
                return self._get_default_settings()

            with open(settings_file, "r", encoding="utf-8") as f:
                settings = yaml.safe_load(f)

            logger.info(f"✅ Loaded settings from {settings_file}")
            return settings

        except Exception as e:
            logger.error(f"Failed to load settings: {e}")
            return self._get_default_settings()

    def _get_default_settings(self) -> Dict[str, Any]:
        """Default settings if config file can't be loaded"""
        return {
            "processing": {
                "max_workers": 4,
                "chunk_size": 50,
                "confidence_thresholds": {
                    "semantic_only": 0.12,
                    "with_web_boost": 0.18,
                    "default": 0.15,
                },
            },
            "ml_models": {
                "sentence_transformers": {
                    "model_name": "all-MiniLM-L6-v2",
                    "show_progress_bar": False,
                }
            },
            "file_io": {
                "default_input_file": "input/Book3.xlsx",
                "default_output_files": {
                    "catalog_only": "output/catalog_only_results.csv",
                    "hybrid": "output/hybrid_tagged_results.csv",
                },
            },
        }


    def load_catalog_config(self, catalog_type: str) -> Dict[str, Any]:
        """Load catalog type configuration (field mappings, output fields, etc.)"""
        try:
            catalog_config_file = self.config_dir / "catalog_types" / f"{catalog_type}.yaml"

            if not catalog_config_file.exists():
                logger.error(f"Catalog config file not found: {catalog_config_file}")
                raise FileNotFoundError(
                    f"No catalog configuration found for catalog_type '{catalog_type}'"
                )

            with open(catalog_config_file, "r", encoding="utf-8") as f:
                catalog_config = yaml.safe_load(f) or {}

            logger.info(f"✅ Loaded catalog config from {catalog_config_file}")
            return catalog_config

        except Exception as e:
            logger.error(f"Failed to load catalog configuration for '{catalog_type}': {e}")
            raise


# Global config loader instance
config_loader = ConfigLoader()


def load_catalog_config(catalog_type: str) -> Dict[str, Any]:
    """Convenience function to load catalog configuration"""
    return config_loader.load_catalog_config(catalog_type)


def load_tags(catalog_type: str) -> Dict[str, Any]:
    """Convenience function to load tags"""
    return config_loader.load_tags(catalog_type)


def load_settings() -> Dict[str, Any]:
    """Convenience function to load settings"""
    return config_loader.load_settings()
