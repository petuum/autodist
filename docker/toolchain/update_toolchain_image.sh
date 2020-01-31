#!/bin/bash

#
# Copyright (c) 2019 Petuum, Inc. All rights reserved.
#

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# We have to do two `dirname`s because this script is two directories deep
ROOT_DIR="$(dirname $(dirname "$SCRIPT_DIR"))"
DOCKER_REGISTRY="registry.petuum.com/internal/scalable-ml/autodist/toolchain"
HASH=$(find $ROOT_DIR -name requirements\*.txt | xargs cat | sort | uniq | md5sum | cut -d" " -f1)
MD5TAG="md5-$HASH"

# Enable experimental Docker
jq '. + {"experimental": "enabled"}' /root/.docker/config.json > temp.json
mv temp.json /root/.docker/config.json

echo "Checking if $DOCKER_REGISTRY:$MD5TAG exists"
docker manifest inspect $DOCKER_REGISTRY:$MD5TAG > /dev/null
code=$?

if [ $code -ne 0 ]; then
    # image does not exist. Build and push it
    echo "Image not found in $DOCKER_REGISTRY. Building it locally..."
    # TensorFlow 2
    docker build -t "$DOCKER_REGISTRY:$MD5TAG" -f $ROOT_DIR/docker/toolchain/update_auto_tf2.Dockerfile $ROOT_DIR
    docker push $DOCKER_REGISTRY:$MD5TAG
    docker tag "$DOCKER_REGISTRY:$MD5TAG" "$DOCKER_REGISTRY:ci_latest"
    docker push "$DOCKER_REGISTRY:ci_latest"
    # TensorFlow 1
    docker build -t "$DOCKER_REGISTRY:ci_latest_tf1" -f $ROOT_DIR/docker/toolchain/update_auto_tf1.Dockerfile $ROOT_DIR
    docker push "$DOCKER_REGISTRY:ci_latest_tf1"
else
    echo "Image found! We will just use that one."
fi
