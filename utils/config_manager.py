import yaml
import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Any, Optional
from constants import BASE_DIR, CONFIG_PATH, DEFAULT_CONFIG

class ConfigManager:
    def __init__(self):
        load_dotenv()
        self.config_path = CONFIG_PATH
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            return self._create_default_config()
        
        with open(self.config_path, 'r') as f:
            try:
                user_config = yaml.safe_load(f) or {}
                # Deep merge would be better, but this is fine for this app
                merged_config = DEFAULT_CONFIG.copy()
                merged_config.update(user_config)
                return merged_config
            except yaml.YAMLError:
                return DEFAULT_CONFIG

    def _create_default_config(self) -> Dict[str, Any]:
        default_config = {
            "apis": {
                "census": {"key": os.getenv("CENSUS_KEY")},
            },
            **DEFAULT_CONFIG
        }
        
        try:
            with open(self.config_path, 'w') as f:
                yaml.dump(default_config, f, default_flow_style=False)
        except IOError:
            print("Could not write default config file.")
        return default_config

    def get_api_key(self, service: str) -> Optional[str]:
        # Priority: 1. Environment Var, 2. Config File
        env_var = os.getenv(service.upper() + "_KEY")
        if env_var:
            return env_var
        return self.config.get("apis", {}).get(service, {}).get("key")