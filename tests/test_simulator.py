from autodist.simulator.utils import _resolve_device_address
from autodist.resource_spec import ResourceSpec
from autodist.cluster import SSHCluster
from autodist.kernel.device.resolver import DeviceResolver
from autodist.simulator.base import SimulatorBase
from autodist.simulator.utils import _resolve_device_address

# def test_resolve_device_address():
#     resource_spec_file = '/home/hao.zhang/project/pycharm/autodist/examples/resource_spec.yml'
#     rs = ResourceSpec(resource_spec_file)
#     cluster = SSHCluster(rs)
#     resolver = DeviceResolver(cluster)
#     return True

def test_resolve():
    resource_spec_file = '/home/hao.zhang/project/pycharm/autodist/examples/resource_spec.yml'
    rs = ResourceSpec(resource_spec_file)
    cluster = SSHCluster(rs)
    resolver = DeviceResolver(cluster)
    SimulatorBase.network_bandwidth(rs, resolver)
    devices = [device for device, _ in rs.devices]

    resolved_devices_1 = [_resolve_device_address(device, resolver) for device, _ in rs.devices]
    devices = resolver.resolve_to_device_str(devices)

    for d1, d2 in zip(resolved_devices_1, devices):
        assert d1 == d2