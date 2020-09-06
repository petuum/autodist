from autodist.resource_spec import ResourceSpec

import pytest
import os
import textwrap

resource_specs = [
    os.path.join(os.path.dirname(__file__), 'integration/resource_specs/r0.yml'),  # single node with 2 GPUs
    os.path.join(os.path.dirname(__file__), 'integration/resource_specs/r9.yml')  # single node with 2 CPUs
]


@pytest.mark.parametrize(
    argnames='resource_spec_path',
    argvalues=[
        os.path.join(os.path.dirname(__file__), 'integration/resource_specs/r{}.yml').format(i)
        for i in range(10)
    ]
)
def test_bandwidth_default(resource_spec_path):
    r = ResourceSpec(resource_file=resource_spec_path)
    assert r.num_cpus >= 0
    for k, v in r.node_cpu_devices:
        # by default it is set as 1
        assert r.network_bandwidth[k] == 1


@pytest.fixture
def tmp_resource_spec_with_bandwidth(tmp_path):
    p = tmp_path / "resource_spec.yml"
    p.write_text(textwrap.dedent(
        """
        nodes:
          - address: 1.1.1.1
            cpus: [0]
            chief: true
            network_bandwidth: 100
          - address: 2.2.2.2
            cpus: [0]
            network_bandwidth: 222
        """
    ))
    return p


def test_bandwidth_fix(tmp_resource_spec):
    r = ResourceSpec(resource_file=tmp_resource_spec)
    assert r.network_bandwidth['1.1.1.1'] == 100
    assert r.network_bandwidth['2.2.2.2'] == 222
