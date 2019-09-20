import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-integration", action="store_true", default=False, help="integration test"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-integration"):
        # --run-integration given in cli: do not skip slow tests
        return
    skip_int = pytest.mark.skip(reason="need --run-integration option to run")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_int)
