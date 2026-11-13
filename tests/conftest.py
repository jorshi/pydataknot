from pathlib import Path

import pytest

INTEGRATION_DIR = Path(__file__).parent.joinpath("integration")


def pytest_addoption(parser):
    parser.addoption(
        "--unit",
        action="store_true",
        help="Run only unit tests (skip collecting tests/integration)",
    )


def pytest_ignore_collect(collection_path, config):
    # Skipping collection (rather than deselecting) avoids importing the heavy
    # integration test modules, which keeps unit test runs fast
    if config.getoption("--unit") and collection_path == INTEGRATION_DIR:
        return True
    return None


def pytest_collection_modifyitems(config, items):
    # Mark everything under tests/integration so `-m integration` also works
    for item in items:
        if INTEGRATION_DIR in item.path.parents:
            item.add_marker(pytest.mark.integration)
