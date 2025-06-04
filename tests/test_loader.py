import os
import sys
import types

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "sarvabhasha_tts")))

# Provide a minimal fake torch module so config can be imported without real dependencies
fake_torch = types.ModuleType("torch")
fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
sys.modules.setdefault("torch", fake_torch)

from sarvabhasha_tts.api import server

class Dummy:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

def test_get_caches_instance(monkeypatch):
    # Use Dummy class for pipeline and reset cache
    monkeypatch.setitem(server.config.PIPELINES, "dummy", "tests.test_loader:Dummy")
    server._instances.clear()

    first = server._get("dummy")
    second = server._get("dummy")
    assert first is second

def test_get_accepts_kwargs(monkeypatch):
    monkeypatch.setitem(server.config.PIPELINES, "acoustic", "tests.test_loader:Dummy")
    server._instances.clear()
    inst = server._get("acoustic", device="cpu")
    assert isinstance(inst, Dummy)
