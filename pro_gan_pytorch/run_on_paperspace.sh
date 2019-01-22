#!/usr/bin/env bash

paperspace jobs create \
  --machineType GPU+ \
  --container ufoym/deepo \
  --ports 8888:22 \
  --project "a_thousand_li" \
  --command "./scripts/enable_ssh.sh && TORCH_HOME=/storage/.torch ARTIFACTS_DIR=/artifacts/ TERM=dumb python3 -u mode_pinning.py"
