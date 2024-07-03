#!/usr/bin/env sh

docker compose exec onnxruntime_cuda python3 /test/make_graph.py
docker compose exec onnxruntime_cude python3 /test/run_graph.py
