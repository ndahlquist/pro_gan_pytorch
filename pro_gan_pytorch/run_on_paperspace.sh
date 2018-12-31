#!/usr/bin/env bash

paperspace jobs create \
  --container ufoym/deepo \
  --machineType GPU+ \
  --ports 8888:22 \
  --command "./scripts/enable_ssh.sh && TORCH_HOME=/storage/.torch ARTIFACTS_DIR=/artifacts/ TERM=dumb python3 -u mode_pinning.py"