// Direct test of HydroPINNWindow's real artifact-loading and inference path,
// with NO QFileDialog automation at all -- this is the point of the test.
//
// improvement_loop_20261023/57 found that driving loadInferenceArtifacts()'s
// modal QFileDialog::getExistingDirectory() under a headless Qt platform
// hangs for environment reasons (no window manager / X11 protocol errors
// under the sandboxes available), and that this hang says nothing about
// whether the GUI's own loading/inference logic is correct -- the dialog and
// the logic it calls are different things, and passing a test of the
// underlying HydroInferenceRunner::runSharedScaleAblation (Hydro/tests/
// checkpoint_memory_seek_test.cpp and the standalone verifier) does not by
// itself establish that hydropinnwindow.cpp's own loadInferenceArtifacts()/
// runLoadedInferenceForMode() wiring is bug-free; that assumption is exactly
// what this test exists to stop relying on.
//
// hydropinnwindow.{h,cpp} now expose loadInferenceArtifactsFromDirectory()
// as a public method precisely so this test can call it directly with a
// known directory, then drive the real "Show LSTM + PINN" button (a normal,
// non-modal QPushButton::click() -- no dialog involved) to exercise the real
// runLoadedInferenceForMode() path end to end, through the actual compiled
// GUI code, not a bypass or a reimplementation.
#include "hydropinnwindow.h"
#include <QApplication>
#include <QPushButton>
#include <QCheckBox>
#include <QLineEdit>
#include <QComboBox>
#include <QTextEdit>
#include <cassert>
#include <iostream>

int main(int argc, char** argv) {
    QApplication app(argc, argv);

    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <frozen_shared_scale_job_dir> <hydro_package_path>\n"
                  << "  e.g. a sixyear_corrected_forcing_2023_zero_s42 directory, and the real\n"
                  << "  Hydro package directory its experiment_config.json's hydro_package_path\n"
                  << "  names (loadInferenceArtifactsFromDirectory loads the checkpoint/config; the\n"
                  << "  Data tab's own package-path field, set below, is what the inference run\n"
                  << "  actually reads data from -- this test sets both explicitly rather than\n"
                  << "  assuming one implies the other, after an earlier version of this test\n"
                  << "  hung by leaving the package-path field empty and never noticing)\n";
        return 2;
    }
    const QString artifactDir = QString::fromLocal8Bit(argv[1]);
    const QString packagePath = QString::fromLocal8Bit(argv[2]);

    HydroPINNWindow window;

    // --- Step 1: load artifacts directly, no dialog. ---
    std::cout << "[test] calling loadInferenceArtifactsFromDirectory() directly\n" << std::flush;
    const bool loaded = window.loadInferenceArtifactsFromDirectory(artifactDir);
    std::cout << "[test] loadInferenceArtifactsFromDirectory returned " << (loaded ? "true" : "false") << "\n";
    if (!loaded) {
        std::cerr << "FAIL: artifact load itself failed -- see the appendLog output above for the real "
                     "exception message (loadInferenceArtifactsFromDirectory no longer shows a QMessageBox "
                     "itself; only the dialog-driven loadInferenceArtifacts() wrapper does)\n";
        return 1;
    }

    // --- Step 2: set the "use current data source" checkbox and the Hydro
    // Package fields the loaded config expects, via genuine widget access
    // (object names assigned by Hydro/generate_gistohq_pinn_window.py).
    // Setting the path/profile fields is required: runLoadedInferenceForMode
    // reads the *current* Data-tab config, not the loaded experiment's own
    // stored path, for exactly the on-disk package to read predictors from. ---
    auto* dataSourceCombo = window.findChild<QComboBox*>("HydroDataSourceCombo");
    auto* packagePathEdit = window.findChild<QLineEdit*>("HydroPackagePathEdit");
    auto* packageProfileCombo = window.findChild<QComboBox*>("HydroPackageProfileCombo");
    if (!dataSourceCombo || !packagePathEdit || !packageProfileCombo) {
        std::cerr << "FAIL: HydroDataSourceCombo/HydroPackagePathEdit/HydroPackageProfileCombo not found -- "
                     "are you running the generated hydropinnwindow_gistohq_pinn translation unit, not a "
                     "bare hydropinnwindow.cpp build?\n";
        return 2;
    }
    dataSourceCombo->setCurrentText("Hydro Package");
    packagePathEdit->setText(packagePath);
    packageProfileCombo->setCurrentText("rainfall-runoff");

    QCheckBox* useCurrentCheck = nullptr;
    QPushButton* showLstmPinnBtn = nullptr;
    for (auto* c : window.findChildren<QCheckBox*>()) {
        if (c->text().contains("Run loaded checkpoint on current Data source")) useCurrentCheck = c;
    }
    for (auto* b : window.findChildren<QPushButton*>()) {
        if (b->text() == "Show LSTM + PINN") showLstmPinnBtn = b;
    }
    if (!useCurrentCheck || !showLstmPinnBtn) {
        std::cerr << "FAIL: checkbox or 'Show LSTM + PINN' button not found\n";
        return 2;
    }
    useCurrentCheck->setChecked(true);

    auto* logEdit = window.findChild<QTextEdit*>();
    const QString logBeforeRun = logEdit ? logEdit->toPlainText() : QString();
    std::cout << "[test] log after load (tail):\n"
              << logBeforeRun.right(400).toStdString() << "\n";

    // --- Step 3: click the real button (a normal, synchronous, non-modal
    // click -- this is the actual runLoadedInferenceForMode() path). ---
    std::cout << "[test] clicking 'Show LSTM + PINN' (real inference call, no dialog involved)\n" << std::flush;
    showLstmPinnBtn->click();

    const QString logAfterRun = logEdit ? logEdit->toPlainText() : QString();
    std::cout << "[test] log after run (tail):\n" << logAfterRun.right(600).toStdString() << "\n";

    const bool ran = logAfterRun.contains("shared-scale-ablation");
    const bool noFailure = !logAfterRun.contains("Loaded checkpoint inference failed");
    std::cout << (ran && noFailure ? "GUI_INFERENCE_PATH_PASS\n" : "GUI_INFERENCE_PATH_FAIL\n");
    return (ran && noFailure) ? 0 : 1;
}
