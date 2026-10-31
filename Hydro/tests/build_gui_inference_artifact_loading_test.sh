#!/usr/bin/env bash
# Builds gui_inference_artifact_loading_test against the real compiled GUI
# translation unit (the qmake-generated hydropinnwindow_gistohq_pinn.cpp, not
# a bare hydropinnwindow.cpp -- HydroPINN.pro only compiles the generated
# derivative into the actual binary, see improvement_loop_20261023/51's
# "the generator transform" finding). Requires a prior `qmake && make` in a
# build directory (not this script's job -- see HydroPINN.pro) so the
# needed .o files already exist there.
#
# That `qmake` step is also what generates hydropinnwindow_gistohq_pinn.cpp
# itself, plus Hydro/generated/{ffn_wrapper_reduced_reservoir,
# lstmnetworkwrapper_reduced_reservoir}.cpp -- none of the three are tracked
# in git (Hydro/generated/.gitignore); HydroPINN.pro triggers all three
# generators as qmake-parse-time side effects (system(python3 ...), lines
# ~80-91). Running `qmake` is therefore not optional scaffolding here, it is
# the actual dependency-generation step; `make` alone, without qmake having
# run first in this build_dir, will not produce the required .o files even
# if the generated .cpp sources happen to already exist from a prior run.
# Unlike build_verify_six_candidate_reproduction.sh, this script does not
# invoke the generators itself -- verifying this path from a bare clean
# checkout still requires the full `qmake HydroPINN.pro && make` sequence,
# which has not been separately re-timed/isolated in this continuation.
#
# Usage: Hydro/tests/build_gui_inference_artifact_loading_test.sh <qmake_build_dir> [output_path]
#   <qmake_build_dir> is wherever `qmake HydroPINN.pro && make` was run (the
#   directory containing hydropinnwindow_gistohq_pinn.o and friends).
#   Then run, under an offscreen Qt platform since this environment has no
#   X11 display, with QT_QPA_PLATFORM=offscreen:
#     <output_path> <frozen_shared_scale_job_dir> <hydro_package_path>
#   No modal dialog is used anywhere in this test -- that is the point of it
#   (improvement_loop_20261023/57's GUI-walkthrough work). Neither path is
#   hardcoded here since both live outside this repository.
set -euo pipefail
if [ $# -lt 1 ]; then
  echo "Usage: $0 <qmake_build_dir> [output_path]" >&2
  exit 2
fi
build_dir="$1"
output="${2:-/tmp/gui_inference_artifact_loading_test}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
libtorch="${LIBTORCH_PATH:-/mnt/3rd900/Projects/libtorch}"

required_objs=(hydropinnwindow_gistohq_pinn.o input_output_plot.o sweep_manager.o quick_sweep.o
  full_tuning_pipeline.o ui_cleanup.o batch_results_summary.o batch_results_auto_view.o
  build_identity_ui.o neuralnetworkwrapper.o neuralnetworkfactory.o hyperparameters.o
  Distribution.o Matrix.o Matrix_arma.o Matrix_arma_sp.o QuickSort.o Utilities.o Vector.o
  Vector_arma.o ddrr_loader.o gistohq_hourly_harmonizer.o gistohq_package_adapter.o
  gistohq_model_rows.o gistohq_temporal_csv.o hydro_checksum.o hydro_dataset_contract.o
  lag_builder.o sequence_builder.o experiment_exporter.o experiment_loader.o artifact_loader.o
  inference_runner.o shared_scale_inference.o pinn_wrapper.o physics_config.o rr_physics.o
  ffn_wrapper_reduced_reservoir.o ffn_pinn_wrapper.o lstm_wrapper.o lstm_pinn_wrapper.o
  lstmnetworkwrapper_reduced_reservoir.o moc_hydropinnwindow.o)
for o in "${required_objs[@]}"; do
  if [ ! -f "$build_dir/$o" ]; then
    echo "Missing $build_dir/$o -- run 'qmake $repo_root/HydroPINN.pro && make' in $build_dir first" >&2
    exit 2
  fi
done

"${CXX:-g++}" -pipe -fopenmp -D_GLIBCXX_USE_CXX11_ABI=1 -O2 -std=gnu++1z -D_REENTRANT -fPIC \
  -DJason -DDEBUG_ -DTORCH_SUPPORT -D_arma -DARMA_USE_OPENMP -DQT_NO_KEYWORDS -DQT_NO_DEBUG \
  -DQT_CHARTS_LIB -DQT_WIDGETS_LIB -DQT_GUI_LIB -DQT_CORE_LIB \
  -I"$repo_root" -I"$build_dir" -I"$libtorch/include/torch/csrc/api/include" -I"$libtorch/include" \
  -I"$repo_root/Utilities" -I"$repo_root/Hydro" -I"$repo_root/Hydro/dataset" \
  -I"$repo_root/Hydro/models" -I"$repo_root/Hydro/physics" \
  -I/usr/include/x86_64-linux-gnu/qt5 -I/usr/include/x86_64-linux-gnu/qt5/QtCharts \
  -I/usr/include/x86_64-linux-gnu/qt5/QtWidgets -I/usr/include/x86_64-linux-gnu/qt5/QtGui \
  -I/usr/include/x86_64-linux-gnu/qt5/QtCore -I/usr/lib/x86_64-linux-gnu/qt5/mkspecs/linux-g++ \
  -c "$repo_root/Hydro/tests/gui_inference_artifact_loading_test.cpp" -o "$build_dir/gui_inference_artifact_loading_test.o"

"${CXX:-g++}" -fopenmp -Wl,-rpath,"$libtorch/lib" -Wl,-O1 -o "$output" \
  "$build_dir/gui_inference_artifact_loading_test.o" \
  "${required_objs[@]/#/$build_dir/}" \
  -L"$libtorch/lib" -ltorch -ltorch_cpu -lc10 -lgomp -larmadillo -lcrypto \
  /usr/lib/x86_64-linux-gnu/libQt5Charts.so /usr/lib/x86_64-linux-gnu/libQt5Widgets.so \
  /usr/lib/x86_64-linux-gnu/libQt5Gui.so /usr/lib/x86_64-linux-gnu/libQt5Core.so -lGL -lpthread

echo "Built: $output"
echo "Run:   QT_QPA_PLATFORM=offscreen $output <frozen_shared_scale_job_dir> <hydro_package_path>"
