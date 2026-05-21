import os
import yaml
from pathlib import Path
from types import SimpleNamespace as config

def _strip_openrouter_env() -> None:
    """Remove OpenRouter keys so LiteLLM cannot route to openrouter.ai."""
    for var in ("OPENROUTER_API_KEY", "OPENROUTER_API_BASE", "OR_SITE_URL", "OR_APP_NAME"):
        os.environ.pop(var, None)
    for var in ("OPENAI_API_KEY", "CHATGPT_API_KEY"):
        val = os.getenv(var, "")
        if val.startswith("sk-or-"):
            os.environ.pop(var, None)

def setup_pageindex_env() -> str:
    """Ensure PAGEINDEX_API_KEY is available; block OpenRouter entirely."""
    _strip_openrouter_env()
    key = os.getenv("PAGEINDEX_API_KEY", "").strip()
    if not key:
        legacy = os.getenv("OPENAI_API_KEY") or os.getenv("CHATGPT_API_KEY") or ""
        if legacy and not legacy.startswith("sk-or-"):
            os.environ["PAGEINDEX_API_KEY"] = legacy
            key = legacy
    return key

class ConfigLoader:
    def __init__(self, default_path: str = None):
        if default_path is None:
            default_path = Path(__file__).parent / "config.yaml"
        self._default_dict = self._load_yaml(default_path)

    @staticmethod
    def _load_yaml(path):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _validate_keys(self, user_dict):
        unknown_keys = set(user_dict) - set(self._default_dict)
        if unknown_keys:
            raise ValueError(f"Unknown config keys: {unknown_keys}")

    def load(self, user_opt=None) -> config:
        """
        Load the configuration, merging user options with default values.
        """
        if user_opt is None:
            user_dict = {}
        elif isinstance(user_opt, config):
            user_dict = vars(user_opt)
        elif isinstance(user_opt, dict):
            user_dict = user_opt
        else:
            raise TypeError("user_opt must be dict, config(SimpleNamespace) or None")

        self._validate_keys(user_dict)
        merged = {**self._default_dict, **user_dict}
        return config(**merged)
