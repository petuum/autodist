# to manage the autodist repo on each test node

node1=$NODE1 
node2=$NODE2

# TODO: put this into a for loop
echo "setup $node1"
if ssh -i /tmp/credentials/id_rsa autodist@$node1 '[ -d /home/autodist/autodist ]'
then
    echo "autodist exists on $node1. remove repo"
    ssh -i /tmp/credentials/id_rsa autodist@$node1 'rm -fr /home/autodist/autodist'
fi
echo "download autodist"
ssh -i /tmp/credentials/id_rsa autodist@$node1 'git clone git@gitlab.int.petuum.com:internal/scalable-ml/autodist.git'
echo "target commit hash is $CI_COMMIT_REF_NAME"
ssh -i /tmp/credentials/id_rsa autodist@$node1 "cd /home/autodist/autodist; pwd; git checkout $CI_COMMIT_REF_NAME"
echo "install autodist"
ssh -i /tmp/credentials/id_rsa autodist@$node1 "/home/autodist/venv/autodist/bin/pip uninstall -y autodist"
ssh -i /tmp/credentials/id_rsa autodist@$node1 "/home/autodist/venv/autodist/bin/pip install -e /home/autodist/autodist"

echo "setup $node2"
if ssh -i /tmp/credentials/id_rsa autodist@$node2 '[ -d /home/autodist/autodist ]'
then
    echo "autodist exists on $node2. remove repo"
    ssh -i /tmp/credentials/id_rsa autodist@$node2 'rm -fr /home/autodist/autodist'
fi
echo "download autodist"
ssh -i /tmp/credentials/id_rsa autodist@$node2 'git clone git@gitlab.int.petuum.com:internal/scalable-ml/autodist.git'
echo "target commit hash is $CI_COMMIT_REF_NAME"
ssh -i /tmp/credentials/id_rsa autodist@$node2 "cd /home/autodist/autodist; pwd; git checkout $CI_COMMIT_REF_NAME"
echo "install autodist"
ssh -i /tmp/credentials/id_rsa autodist@$node2 "/home/autodist/venv/autodist/bin/pip uninstall -y autodist"
ssh -i /tmp/credentials/id_rsa autodist@$node2 "/home/autodist/venv/autodist/bin/pip install -e /home/autodist/autodist"