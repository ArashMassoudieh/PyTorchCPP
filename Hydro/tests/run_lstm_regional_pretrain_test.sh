#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
libtorch="${LIBTORCH_PATH:-/mnt/3rd900/Projects/libtorch}"
if [[ ! -f "$libtorch/include/torch/csrc/api/include/torch/torch.h" ]]; then
  libtorch=/usr/local/libtorch
fi
if [[ ! -f "$libtorch/include/torch/csrc/api/include/torch/torch.h" ]]; then
  echo "SKIP: LibTorch headers not found; set LIBTORCH_PATH to run lstm_regional_pretrain_test." >&2
  if [[ "${HYDRO_REQUIRE_LIBTORCH_TESTS:-0}" == "1" ]]; then exit 1; fi
  exit 77
fi
binary="${TMPDIR:-/tmp}/hydro_lstm_regional_pretrain_test"
"${CXX:-c++}" -std=c++17 -O0 -Wall -Wextra -fopenmp -D_GLIBCXX_USE_CXX11_ABI="${TORCH_CXX11_ABI:-1}" \
  -I"$repo_root" -I"$repo_root/Hydro" \
  -I"$libtorch/include" -I"$libtorch/include/torch/csrc/api/include" \
  "$repo_root/Hydro/tests/lstm_regional_pretrain_test.cpp" \
  "$repo_root/Hydro/models/lstm_regional_pretrain.cpp" \
  "$repo_root/Hydro/dataset/gistohq_hourly_harmonizer.cpp" \
  "$repo_root/Hydro/dataset/gistohq_package_adapter.cpp" \
  "$repo_root/Hydro/dataset/gistohq_model_rows.cpp" \
  "$repo_root/Hydro/dataset/gistohq_temporal_csv.cpp" \
  "$repo_root/Hydro/dataset/hydro_checksum.cpp" \
  "$repo_root/Hydro/dataset/hydro_dataset_contract.cpp" \
  "$repo_root/Hydro/dataset/ddrr_loader.cpp" \
  -L"$libtorch/lib" -Wl,-rpath,"$libtorch/lib" -ltorch -ltorch_cpu -lc10 -lgomp -lpthread -lcrypto -o "$binary"
"$binary"
