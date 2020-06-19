from autodist.resource_spec import DeviceSpec, DeviceType


def test_device_spec_string():
    d1 = DeviceSpec(host_address='0.0.0.0', device_index=0)
    s1 = d1.name_string()
    assert s1 == '0.0.0.0:CPU:0'
    assert DeviceSpec.from_string(s1).name_string() == s1

    d2 = DeviceSpec(host_address='localhost', device_type=DeviceType.GPU, device_index=2)
    s2 = d2.name_string()
    assert s2 == 'localhost:GPU:2'
    assert DeviceSpec.from_string(s2).name_string() == s2
