import sys


def test_supported_cpython_runtime() -> None:
    assert sys.implementation.name == "cpython"
    assert sys.version_info[:2] == (3, 14)
    assert sys.version_info >= (3, 14, 6)
