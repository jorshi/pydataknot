import os

import pytest


@pytest.fixture
def rundir(tmp_path):
    current_dir = os.getcwd()
    yield tmp_path
    os.chdir(current_dir)
