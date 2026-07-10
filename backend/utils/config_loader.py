from pathlib import Path
from typing import Any

import yaml


class ConfigLoader:
  

    @staticmethod
    def load(config_path: str) -> dict[str, Any]:
     

        path = Path(config_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Configuration file '{config_path}' not found."
            )

        with open(path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)

    @staticmethod
    def save(config: dict[str, Any], output_path: str) -> None:
       
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as file:
            yaml.dump(
                config,
                file,
                default_flow_style=False,
                sort_keys=False
            )