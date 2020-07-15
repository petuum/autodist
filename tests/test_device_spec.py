from autodist.resource_spec import DeviceSpec, DeviceType

import os
from autodist.resource_spec import DeviceSpec, DeviceType, ResourceSpec

resource_specs = [
    os.path.join(os.path.dirname(__file__), 'integration/resource_specs/r0.yml'),  # single node with 2 GPUs
    os.path.join(os.path.dirname(__file__), 'integration/resource_specs/r5.yml')  # single node with 2 CPUs
]

def test_device_spec_string():
    d1 = DeviceSpec(host_address='0.0.0.0', device_index=0)
    s1 = d1.name_string()
    assert s1 == '0.0.0.0:CPU:0'
    assert DeviceSpec.from_string(s1).name_string() == s1

    d2 = DeviceSpec(host_address='localhost', device_type=DeviceType.GPU, device_index=2)
    s2 = d2.name_string()
    assert s2 == 'localhost:GPU:2'
    assert DeviceSpec.from_string(s2).name_string() == s2


def test_gpu_cpu_num():
    d1 = ResourceSpec(resource_file=resource_specs[0])
    s1 = d1.num_gpus
    assert s1 == 2
    d2 = ResourceSpec(resource_file=resource_specs[1])
    s2 = d2.num_cpus
    assert s2 == 2
