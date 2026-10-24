import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(autouse=True)
def tmp_state(tmp_path, monkeypatch):
    from fclbot import store
    monkeypatch.setattr(store, "STATE_DIR", str(tmp_path))
    return tmp_path
