#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
cat weight/iter_141000.pth.part* > weight/iter_141000.pth
