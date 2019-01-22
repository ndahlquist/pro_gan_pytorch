#!/usr/bin/env bash

paperspace jobs create \
 --clusterId "clj7lehl0" \
 --container ufoym/deepo \
 --project "a_thousand_li" \
 --command "TORCH_HOME=/persistent/.torch ARTIFACTS_DIR=/artifacts/ TERM=dumb python3 -u mode_pinning.py"
