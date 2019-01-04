#!/usr/bin/env bash

paperspace jobs create \
 --container ufoym/deepo \
 --clusterId "cl89n4h5h" \
 --ports 8888:22 \
 --command "./scripts/enable_ssh.sh && TORCH_HOME=/storage/.torch CHECKPOINTS_DIR=/storage/checkpoints SAMPLES_DIR=/artifacts/ TERM=dumb python3 -u mode_pinning.py"
