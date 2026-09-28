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
    # synthetic IMUs are body-frame (like a live phone), so gyro-propagated attitude
    # is valid; the IO-VNBD default turns it off (Earth-levelled accelerometer)
    c["preprocess"]["attitude_use_gyro"] = True
    return c
