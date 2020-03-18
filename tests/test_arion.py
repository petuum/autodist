import pytest
import textwrap

@pytest.fixture
def tmp_resource_spec(tmp_path):
    p = tmp_path / "resource_spec.yml"
    p.write_text(textwrap.dedent(
        """
        nodes:
            - address: localhost
              gpus: [0,1]
        """
    ))
    return p

def test_single_process(tmp_resource_spec):
    from autodist import AutoDist
    AutoDist(resource_spec_file=tmp_resource_spec)
    with pytest.raises(NotImplementedError):
        AutoDist(resource_spec_file=tmp_resource_spec)