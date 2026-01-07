#!/bin/bash
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
PROJECT_ROOT_DIR="$(realpath "$SCRIPT_DIR/../..")"

mkdir -p $PROJECT_ROOT_DIR/data
cd $PROJECT_ROOT_DIR/data
wget https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip
unzip Replica.zip
rm Replica.zip