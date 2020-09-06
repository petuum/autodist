from autodist.resource_spec import ResourceSpec

import pytest
import os
import textwrap


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
    for k in r.node_cpu_devices:
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
            network_bandwidth: 123
            ssh_config: conf
          - address: 2.2.2.2
            cpus: [0]
            network_bandwidth: 456
            ssh_config: conf
        ssh:
          conf:
            username: 'root'
            key_file: '/root/.ssh/id_rsa'
            port: 12345
        """
    ))
    return p


def test_bandwidth_fix(tmp_resource_spec_with_bandwidth):
    r = ResourceSpec(resource_file=tmp_resource_spec_with_bandwidth)
    assert r.network_bandwidth['1.1.1.1'] == 123
    assert r.network_bandwidth['2.2.2.2'] == 456
