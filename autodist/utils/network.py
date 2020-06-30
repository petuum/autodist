# Copyright 2020 Petuum. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Network utility functions."""
from ipaddress import ip_address

import netifaces


def is_loopback_address(address):
    """
    Determine whether an address is a loopback address (e.g. 127.0.0.1).

    Args:
        address (str): Address (can be IP or IP:port)

    Returns:
        Boolean
    """
    ip = _get_ip_from_address(address)
    return ip.is_loopback


def is_local_address(address):
    """
    Determine whether an address is a local (including loopback) IP address.

    Adapted from stackoverflow.com/questions/166506.

    Args:
        address (str): Address (can be IP or IP:port)

    Returns:
        Boolean
    """
    ip = _get_ip_from_address(address)

    # Get all addresses
    addresses = set()
    for iface_name in netifaces.interfaces():
        for i in netifaces.ifaddresses(iface_name).setdefault(netifaces.AF_INET, [{'addr': None}]):
            if i['addr']:
                addresses.add(ip_address(i['addr']))

    return ip in addresses


def _get_ip_from_address(address):
    """
    Extract an IP Address object from an address string.

    Args:
        address (str): Address (can be IP or IP:port)

    Returns:
        An IPv4Address or IPv6Address object.
    """
    ip, _, _ = address.rpartition(':')
    ip = ip or address  # If there was no separation, ip will be empty so use original string
    if ip == 'localhost':
        # These should be equivalent
        # `ip_address` will throw an error if given localhost
        ip = '127.0.0.1'
    return ip_address(ip.strip("[]"))  # IPv6 addresses might contain [] to separate address and port
