#!/usr/bin/env sh

./tools/scripts/compile_triton_kernels.sh

sudo docker compose -f docker-compose.yml up -d --build --remove-orphans --wait
