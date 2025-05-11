import os
from typing import Dict
from dotenv import load_dotenv

load_dotenv()

class ModelRegistryService:
    def __init__(self):
        self.model_dir = os.getenv("MODEL_DIR", "../models")
        self.models = self._discover_models()

    def _discover_models(self) -> Dict[str, Dict]:
        available = {}
        for filename in os.listdir(self.model_dir):
            if filename.endswith(".onnx"):
                model_name = os.path.splitext(filename)[0].lower()
                available[model_name] = {
                    "path": os.path.join(self.model_dir, filename),
                    "input_name": "data",
                    "output_dim": 512
                }
        return available

    def get_all_models(self) -> Dict[str, Dict]:
        return self.models

    def get_model_path(self, model_name: str) -> str:
        return self.models.get(model_name, {}).get("path")
