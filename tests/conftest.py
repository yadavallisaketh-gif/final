import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import load_config  # noqa: E402


@pytest.fixture
def cfg():
    c = load_config()
    c["evaluate"]["warmup_s"] = 120
    c["evaluate"]["post_s"] = 20
    return c
