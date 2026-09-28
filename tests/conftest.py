import pytest
from make_samples import create_samples


@pytest.fixture(scope='session')
def insurance_workbook(tmp_path_factory):
    return create_samples(tmp_path_factory.mktemp('insurance_samples'))
