# to manage the autodist repo on each test node

node1=$NODE1 
node2=$NODE2

# TODO: put this into a for loop
echo "setup $node1"
echo "download autodist from the build artifact"
ssh -i /tmp/credentials/id_rsa autodist@$node1 "rm -f /home/autodist/dist/autodist*.whl"
scp -i /tmp/credentials/id_rsa dist/autodist*.whl autodist@$node1:/home/autodist/dist
echo "install autodist"
ssh -i /tmp/credentials/id_rsa autodist@$node1 "/home/autodist/venv/autodist/bin/pip uninstall -y autodist"
ssh -i /tmp/credentials/id_rsa autodist@$node1 "/home/autodist/venv/autodist/bin/pip install /home/autodist/dist/autodist*.whl"

echo "setup $node2"
echo "download autodist from the build artifact"
ssh -i /tmp/credentials/id_rsa autodist@$node2 "rm -f /home/autodist/dist/autodist*.whl"
scp -i /tmp/credentials/id_rsa dist/autodist*.whl autodist@$node2:/home/autodist/dist
echo "install autodist"
ssh -i /tmp/credentials/id_rsa autodist@$node2 "/home/autodist/venv/autodist/bin/pip uninstall -y autodist"
ssh -i /tmp/credentials/id_rsa autodist@$node2 "/home/autodist/venv/autodist/bin/pip install /home/autodist/dist/autodist*.whl"