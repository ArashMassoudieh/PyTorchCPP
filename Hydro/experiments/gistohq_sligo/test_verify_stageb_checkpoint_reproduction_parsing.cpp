// Regression tests for verify_stageb_checkpoint_reproduction_parsing.h,
// added per independent review (2026-11-18, checkpoint69): the original
// readExistingPredictions had no fail-closed validation at all. Each guard
// is reproduced here directly against a synthetic predictions.csv before
// being trusted. No torch dependency -- builds in under a second.
#include "verify_stageb_checkpoint_reproduction_parsing.h"

#include <cassert>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>

namespace {

std::string writeTempCsv(const std::string& name, const std::string& content) {
    const auto path = std::filesystem::temp_directory_path() / name;
    std::ofstream f(path);
    f << content;
    f.close();
    return path.string();
}

bool throws(const std::function<void()>& fn) {
    try {
        fn();
        return false;
    } catch (const std::exception&) {
        return true;
    }
}

void test_well_formed_rows_parse_correctly() {
    const std::string path = writeTempCsv("ok.csv",
        "approach,index,split,x,observed,predicted,residual\n"
        "lstm_pinn,0,test,100,5.0,5.1,-0.1\n"
        "lstm_pinn,1,test,101,6.0,5.9,0.1\n");
    auto rows = readExistingPredictions(path, "lstm_pinn");
    assert(rows.size() == 2);
    assert(rows[0].x == 100.0);
    assert(rows[0].observed == 5.0);
    assert(rows[0].predicted == 5.1);
}

void test_negative_x_is_rejected() {
    const std::string path = writeTempCsv("neg.csv",
        "approach,index,split,x,observed,predicted,residual\n"
        "lstm_pinn,0,test,-5,5.0,5.1,-0.1\n");
    assert(throws([&] { readExistingPredictions(path, "lstm_pinn"); }));
}

void test_non_integer_x_is_rejected() {
    const std::string path = writeTempCsv("frac.csv",
        "approach,index,split,x,observed,predicted,residual\n"
        "lstm_pinn,0,test,100.5,5.0,5.1,-0.1\n");
    assert(throws([&] { readExistingPredictions(path, "lstm_pinn"); }));
}

void test_sealed_x_is_rejected() {
    const std::string path = writeTempCsv("sealed.csv",
        "approach,index,split,x,observed,predicted,residual\n"
        "lstm_pinn,0,test,77016,5.0,5.1,-0.1\n");
    assert(throws([&] { readExistingPredictions(path, "lstm_pinn"); }));
}

void test_sealed_x_for_other_approach_is_still_rejected() {
    // The guard must apply BEFORE the approach filter -- a sealed row
    // belonging to a DIFFERENT approach sharing this file must not be
    // silently skipped just because it isn't the approach being verified.
    const std::string path = writeTempCsv("sealed_other.csv",
        "approach,index,split,x,observed,predicted,residual\n"
        "some_other_approach,0,test,77016,1.0,1.0,0.0\n"
        "lstm_pinn,1,test,100,5.0,5.1,-0.1\n");
    assert(throws([&] { readExistingPredictions(path, "lstm_pinn"); }));
}

void test_duplicate_x_for_approach_is_rejected() {
    const std::string path = writeTempCsv("dup.csv",
        "approach,index,split,x,observed,predicted,residual\n"
        "lstm_pinn,0,test,100,5.0,5.1,-0.1\n"
        "lstm_pinn,1,test,100,5.0,5.2,-0.2\n");
    assert(throws([&] { readExistingPredictions(path, "lstm_pinn"); }));
}

void test_malformed_header_is_rejected() {
    const std::string path = writeTempCsv("badheader.csv",
        "approach,index,x,observed,predicted\n"
        "lstm_pinn,0,100,5.0,5.1\n");
    assert(throws([&] { readExistingPredictions(path, "lstm_pinn"); }));
}

void test_non_finite_observed_is_rejected() {
    const std::string path = writeTempCsv("nonfinite.csv",
        "approach,index,split,x,observed,predicted,residual\n"
        "lstm_pinn,0,test,100,nan,5.1,-0.1\n");
    assert(throws([&] { readExistingPredictions(path, "lstm_pinn"); }));
}

void test_no_rows_for_approach_is_rejected() {
    const std::string path = writeTempCsv("norows.csv",
        "approach,index,split,x,observed,predicted,residual\n"
        "lstm_multiscale,0,test,100,5.0,5.1,-0.1\n");
    assert(throws([&] { readExistingPredictions(path, "lstm_pinn"); }));
}

void test_extract_int64_and_string() {
    const std::string json = "{\"train_end_x\": 21887, \"name\": \"foo\"}";
    assert(extractInt64(json, "train_end_x") == 21887);
    assert(extractString(json, "name") == "foo");
    assert(throws([&] { extractInt64(json, "missing_field"); }));
}

void test_parse_frozen_manifest_hashes() {
    const auto dir = std::filesystem::temp_directory_path() / "frozen_manifest_test";
    std::filesystem::create_directories(dir);
    const auto stagedFile = dir / "job.json";
    { std::ofstream f(stagedFile); f << "{}"; }
    const auto manifestPath = dir / "STAGED_MANIFEST_HASHES.txt";
    {
        std::ofstream f(manifestPath);
        f << "ROOT: " << dir.string() << "\n";
        f << "abc123  job.json\n";
    }
    auto hashes = parseFrozenManifestHashes(manifestPath.string());
    const std::string resolvedKey = std::filesystem::weakly_canonical(stagedFile).string();
    assert(hashes.count(resolvedKey) == 1);
    assert(hashes[resolvedKey] == "abc123");
}

}  // namespace

int main() {
    test_well_formed_rows_parse_correctly();
    test_negative_x_is_rejected();
    test_non_integer_x_is_rejected();
    test_sealed_x_is_rejected();
    test_sealed_x_for_other_approach_is_still_rejected();
    test_duplicate_x_for_approach_is_rejected();
    test_malformed_header_is_rejected();
    test_non_finite_observed_is_rejected();
    test_no_rows_for_approach_is_rejected();
    test_extract_int64_and_string();
    test_parse_frozen_manifest_hashes();
    std::cout << "ALL PARSING TESTS PASSED\n";
    return 0;
}
