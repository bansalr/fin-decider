import pytest
import yaml
from pydantic import ValidationError

from classifier.config import Config


def raw(repo_root):
    return yaml.safe_load((repo_root / "configs/classification.yaml").read_text())


def test_real_config_loads(cfg):
    assert cfg.models["jev"].model == "typesafe-ai/jev"
    assert cfg.models["laya"].template == cfg.models["jev"].template


def test_unknown_key_fails(repo_root):
    d = raw(repo_root)
    d["hierarchy"]["traversal"]["surprise"] = 1
    with pytest.raises(ValidationError):
        Config.model_validate(d)


@pytest.mark.parametrize("alias", ["typesafe-ai/latest", "jev-latest", "typesafe-ai/jev-latest"])
def test_latest_alias_fails(repo_root, alias):
    d = raw(repo_root)
    d["models"]["jev"]["model"] = alias
    with pytest.raises(ValidationError, match="unpinned"):
        Config.model_validate(d)


def test_bad_sha_and_revision_fail(repo_root):
    d = raw(repo_root)
    d["ontology"]["sha256"] = "<SHA256>"
    with pytest.raises(ValidationError):
        Config.model_validate(d)
    d = raw(repo_root)
    d["dataset"]["revision"] = "main"
    with pytest.raises(ValidationError):
        Config.model_validate(d)


def test_threshold_range(repo_root):
    d = raw(repo_root)
    d["hierarchy"]["classification"]["positive_threshold"] = 1.5
    with pytest.raises(ValidationError):
        Config.model_validate(d)


def test_local_model_needs_pinned_revision(repo_root):
    d = raw(repo_root)
    d["models"]["gliclass"]["revision"] = "main"
    with pytest.raises(ValidationError, match="revision"):
        Config.model_validate(d)
    d = raw(repo_root)
    del d["models"]["gliclass"]["dtype"]
    with pytest.raises(ValidationError):
        Config.model_validate(d)


def test_gateway_model_rejects_local_fields(repo_root):
    d = raw(repo_root)
    d["models"]["jev"]["dtype"] = "float16"
    with pytest.raises(ValidationError, match="not valid for gateway"):
        Config.model_validate(d)


def test_overrides(repo_root, tmp_path):
    from classifier.config import load_config
    cfg = load_config(repo_root / "configs/classification.yaml", ["models.gliclass.device=mps", "models.gliclass.dtype=float32"])
    assert cfg.models["gliclass"].device == "mps" and cfg.models["gliclass"].dtype == "float32"
    with pytest.raises(ValueError, match="unknown key"):
        load_config(repo_root / "configs/classification.yaml", ["models.gliclass.devise=mps"])


def test_template_kind_enforced(repo_root):
    d = raw(repo_root)
    del d["templates"]["boolean_v1"]["kind"]
    with pytest.raises(ValidationError):
        Config.model_validate(d)
