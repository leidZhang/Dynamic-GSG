#!/bin/bash
SCRIPT_DIR="$(dirname "$(realpath "$0")")"

set -e
IMAGE_NAME="dgsg-dev"
TAG="latest"

if [[ "$(docker images -q ${IMAGE_NAME}:${TAG} 2> /dev/null)" == "" ]]; then
    echo "Image ${IMAGE_NAME}:${TAG} not found locally — building now."
    # docker build -t ${IMAGE_NAME}:${TAG} .
    docker compose -f $SCRIPT_DIR/docker-compose.yml build
else
    echo "Image ${IMAGE_NAME}:${TAG} already exists — skipping build."
fi

echo "Configuration complete. Attempting to start the container..."
# Check if a container with the same name is already running
if [[ "$(docker ps -q -f name=dgsg_dev_container 2> /dev/null)" != "" ]]; then
    echo "Container dgsg_dev_container is already running, attaching bash shell..."
else
    echo "Starting container dgsg_dev_container..."
    # Start the container in detached mode
    docker compose -f $SCRIPT_DIR/docker-compose.yml up -d
fi

# Attach to the running container's bash shell
docker compose -f $SCRIPT_DIR/docker-compose.yml exec -it dgsg_dev bash