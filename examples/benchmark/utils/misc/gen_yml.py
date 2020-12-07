# Copyright 2020 Petuum, Inc. All Rights Reserved.
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

import logging
import requests
import adaptdl.collective as collective
import os
import getpass
import adaptdl.env as env
logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

def generate():
    url = env.supervisor_url()
    if url:
        key = env.job_id()
        group = env.num_restarts()
        while True:
            response = requests.get(url=f"{url}/discover/{key}/{group}", 
                                    params={"gpu" : True})
            if response.status_code != 408:  # Timeout.
                break
        response.raise_for_status()
        master_addr = response.json()[0][0]
    else:
        raise ValueError("supervisor url not found.")
    # write to the share path
    path = os.path.join(env.share_path(), "resource_spec.yml")
    LOG.info(f"writing to {path}")

    f = open(path, "w")
    f.write("nodes: \n")
    num_nodes = len(response.json())
    for i in range(num_nodes):
        f.write(f"  - address: {response.json()[i][0]} \n")
        f.write(f"    gpus: {list(range(response.json()[i][1]))} \n")
        if i == 0:  # chief
            master_addr = response.json()[i][0]
            f.write("    chief: true \n")
        else:
            f.write("    ssh_config: conf \n")
    f.write("ssh: \n")
    f.write("  conf: \n")
    f.write(f"    username: '{getpass.getuser()}' \n")
    f.write("    key_file: '/root/.ssh/id_rsa' \n")
    f.close()
    # Initialize collective module.
    master_port = env.master_port()
    collective.initialize(master_addr, master_port)
