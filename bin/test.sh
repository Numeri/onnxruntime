#!/usr/bin/env sh

sudo docker compose exec onnxruntime_cuda python3 /test/make_graph.py
sudo docker compose exec onnxruntime_cuda python3 /test/run_graph.py
