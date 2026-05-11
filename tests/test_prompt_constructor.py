import importlib.util
import os
from pathlib import Path
import sys
import types
from types import SimpleNamespace

import pytest


def _load_prompt_constructor():
    for env_var in [
        "REDDIT",
        "SHOPPING",
        "SHOPPING_ADMIN",
        "GITLAB",
        "WIKIPEDIA",
        "MAP",
        "HOMEPAGE",
    ]:
        os.environ.setdefault(env_var, f"http://{env_var.lower()}.example")

    module_path = Path("agent/prompts/prompt_constructor.py")
    tokenizers_module = types.ModuleType("llms.tokenizers")
    tokenizers_module.Tokenizer = object
    sys.modules["llms.tokenizers"] = tokenizers_module

    spec = importlib.util.spec_from_file_location("prompt_constructor_under_test", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PromptConstructor


def test_huggingface_prompt_constructor_without_model_tag_raises_value_error() -> None:
    prompt_constructor = _load_prompt_constructor()
    constructor = prompt_constructor.__new__(prompt_constructor)
    constructor.lm_config = SimpleNamespace(
        provider="huggingface",
        model="microsoft/phi-2",
        mode="chat",
        gen_config={},
    )

    with pytest.raises(ValueError, match="microsoft/phi-2"):
        constructor.get_lm_api_input("intro", [("observation", "action")], "current")
