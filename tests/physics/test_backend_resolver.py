"""Tests for the physics backend resolver."""
import pytest

from vsss_sim.physics import get_backend


class TestGetBackend:
    def test_default_is_numpy(self, monkeypatch):
        monkeypatch.delenv("VSSS_PHYSICS_BACKEND", raising=False)
        backend = get_backend()
        assert backend.__name__.endswith("numpy_backend")

    def test_explicit_numpy(self, monkeypatch):
        monkeypatch.delenv("VSSS_PHYSICS_BACKEND", raising=False)
        backend = get_backend("numpy")
        assert backend.__name__.endswith("numpy_backend")

    def test_explicit_jax(self, monkeypatch):
        monkeypatch.delenv("VSSS_PHYSICS_BACKEND", raising=False)
        backend = get_backend("jax")
        assert backend.__name__.endswith("jax_backend")

    def test_env_var_jax(self, monkeypatch):
        monkeypatch.setenv("VSSS_PHYSICS_BACKEND", "jax")
        backend = get_backend()
        assert backend.__name__.endswith("jax_backend")

    def test_kwarg_overrides_env_var(self, monkeypatch):
        monkeypatch.setenv("VSSS_PHYSICS_BACKEND", "jax")
        backend = get_backend("numpy")
        assert backend.__name__.endswith("numpy_backend")

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            get_backend("isaac")
