#!/usr/bin/env bash
# Reproduce everything from raw data (same as `make all`).
set -e
cd "$(dirname "$0")/.."
make all
