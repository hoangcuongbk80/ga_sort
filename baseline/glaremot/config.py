"""Central configuration loader.

Every script and module resolves paths through here, so the repository has no
hard-coded machine paths. Edit ``configs/paths.yaml`` (or point the
``GLAREMOT_CONFIG`` environment variable at your own copy) and everything - datasets, weights, caches, results - follows.
"""
import os

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = os.path.join(REPO_ROOT, "configs", "paths.yaml")


class Cfg(dict):
    """dict with attribute access (cfg.data.tracking)."""

    def __getattr__(self, k):
        try:
            v = self[k]
        except KeyError as e:
            raise AttributeError(k) from e
        return Cfg(v) if isinstance(v, dict) else v


_cached = None


def load_config(path=None):
    """Load configs/paths.yaml (cached). Relative paths resolve to the repo root."""
    global _cached
    if path is None and _cached is not None:
        return _cached
    p = path or os.environ.get("GLAREMOT_CONFIG", DEFAULT_CONFIG)
    with open(p, encoding="utf-8") as f:
        cfg = Cfg(yaml.safe_load(f))
    if path is None:
        _cached = cfg
    return cfg


def resolve(p):
    """Resolve a config path: absolute stays, relative is repo-rooted."""
    p = str(p)
    return p if os.path.isabs(p) else os.path.join(REPO_ROOT, p)


def data_path(cfg, key):
    return resolve(cfg.data[key])


def weight_path(cfg, key):
    return resolve(cfg.weights[key])


def cache_dir(cfg, *sub):
    d = os.path.join(resolve(cfg.cache_dir), *sub)
    os.makedirs(d, exist_ok=True)
    return d


def results_dir(cfg, *sub):
    d = os.path.join(resolve(cfg.results_dir), *sub)
    os.makedirs(d, exist_ok=True)
    return d
