import sys
from types import ModuleType, SimpleNamespace

from airllm_benchmark.metrics import ChildProcessMemorySampler, ProcessMemorySampler, _rss_bytes
from airllm_benchmark.runners import airllm_runtime, baseline_transformers


class FakeIds:
    shape = (1, 3)


class FakeGenerated:
    shape = (2,)


class FakeSequence:
    def __getitem__(self, key):
        return FakeGenerated() if isinstance(key, slice) else self


class FakeTokenizer:
    eos_token_id = 0

    def __call__(self, prompt: str, return_tensors: str):
        return {"input_ids": FakeIds()}

    def decode(self, ids: object, skip_special_tokens: bool) -> str:
        return "decoded"


class FakeModel:
    tokenizer = FakeTokenizer()

    def eval(self) -> None:
        return None

    def generate(self, *args, **kwargs):
        return [FakeSequence()]


class NoGrad:
    def __enter__(self):
        return self

    def __exit__(self, *_exc: object) -> None:
        return None


def install_fake_torch(monkeypatch) -> None:
    torch = ModuleType("torch")
    torch.float32 = "float32"
    torch.no_grad = lambda: NoGrad()
    monkeypatch.setitem(sys.modules, "torch", torch)


def test_baseline_transformers_success_path_with_fake_modules(monkeypatch) -> None:
    install_fake_torch(monkeypatch)
    transformers = ModuleType("transformers")
    transformers.AutoTokenizer = SimpleNamespace(
        from_pretrained=lambda *_args, **_kwargs: FakeTokenizer()
    )
    transformers.AutoModelForCausalLM = SimpleNamespace(
        from_pretrained=lambda *_args, **_kwargs: FakeModel()
    )
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    result = baseline_transformers.run_transformers_success_path(
        run_id="run",
        model_id="model",
        prompt="prompt",
        max_new_tokens=2,
        temperature=0.0,
        local_files_only=True,
        notes_prefix=["prefix"],
    )

    assert result.status == "success"
    assert result.input_tokens == 3
    assert result.output_tokens == 2
    assert result.output_sample == "decoded"
    assert "prefix" in result.notes


def test_airllm_runtime_success_and_cache_config(monkeypatch, tmp_path) -> None:
    install_fake_torch(monkeypatch)
    airllm = ModuleType("airllm")
    airllm.AutoModel = SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: FakeModel())
    monkeypatch.setitem(sys.modules, "airllm", airllm)

    monkeypatch.setattr(airllm_runtime, "configure_huggingface_cache", lambda _path: None)
    result = airllm_runtime.run_airllm_success_path(
        run_id="run",
        model_id="model",
        prompt="prompt",
        max_new_tokens=2,
        temperature=0.2,
        layer_shards_saving_path=tmp_path / "shards",
        huggingface_cache_dir=tmp_path / "hf",
        compression="4bit",
        delete_original=False,
    )

    assert result.status == "success"
    assert result.quantization == "4bit"
    assert result.input_tokens == 3
    assert result.output_tokens == 2


def test_airllm_cache_config_windows_symlink_patch(monkeypatch, tmp_path) -> None:
    hub = ModuleType("huggingface_hub")
    constants = ModuleType("huggingface_hub.constants")
    file_download = ModuleType("huggingface_hub.file_download")
    file_download.are_symlinks_supported = lambda cache_dir=None: True
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "huggingface_hub.constants", constants)
    monkeypatch.setitem(sys.modules, "huggingface_hub.file_download", file_download)
    monkeypatch.setattr(sys, "platform", "win32")

    airllm_runtime.configure_huggingface_cache(tmp_path / "hf")

    assert str(tmp_path / "hf") == constants.HF_HOME
    assert file_download.are_symlinks_supported() is False


def test_memory_samplers_and_rss_error_path(monkeypatch) -> None:
    with ProcessMemorySampler(interval_seconds=0.001) as memory:
        assert memory.peak_ram_mb > 0

    child = ChildProcessMemorySampler(999999, interval_seconds=0.001)
    child._record_current_rss()
    assert child.peak_ram_mb is None

    class BadProcess:
        def memory_info(self):
            import psutil

            raise psutil.Error()

    assert _rss_bytes(BadProcess()) == 0
