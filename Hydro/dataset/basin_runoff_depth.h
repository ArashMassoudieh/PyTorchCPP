#pragma once

#include <cmath>
#include <stdexcept>

// Converts between absolute discharge (m^3/s) and basin-normalized runoff
// depth (mm/hour), using the identical formula
// gistohq_hourly_harmonizer.cpp already applies when it builds
// row.observed_runoff_mm_per_hour from row.observed_discharge_m3_per_second
// and config.catchment_area_m2 (confirmed by direct code inspection, not
// assumed) - that conversion is already what every model in this codebase
// is trained against, since loadHydroPackageTensors's y tensor IS that
// depth field, not raw discharge. Multi-basin pooling relies on this: the
// same mm/hour scale is physically comparable across basins of different
// catchment area, which raw discharge in cfs/cms is not.
//
// These two functions exist to (a) make the forward formula's provenance
// explicit and independently unit-tested (round-trip, and against a
// hand-computed reference value) rather than trusting it by inspection
// alone, and (b) provide the inverse (depth -> discharge) for reporting,
// which no existing code path currently needs (NSE and PBIAS are both
// invariant to a constant per-basin multiplicative rescaling, so neither
// metric's value depends on which of the two units was used), but which a
// human reading model output in physically familiar units (cfs/cms) does.
inline double dischargeCubicMetersPerSecondToRunoffMmPerHour(double dischargeM3PerSecond,
                                                             double catchmentAreaM2) {
    if (!std::isfinite(catchmentAreaM2) || catchmentAreaM2 <= 0.0) {
        throw std::invalid_argument("Catchment area must be a positive, finite number of square meters.");
    }
    if (!std::isfinite(dischargeM3PerSecond)) {
        throw std::invalid_argument("Discharge must be finite.");
    }
    return dischargeM3PerSecond * 3600.0 * 1000.0 / catchmentAreaM2;
}

inline double runoffMmPerHourToDischargeCubicMetersPerSecond(double runoffMmPerHour,
                                                             double catchmentAreaM2) {
    if (!std::isfinite(catchmentAreaM2) || catchmentAreaM2 <= 0.0) {
        throw std::invalid_argument("Catchment area must be a positive, finite number of square meters.");
    }
    if (!std::isfinite(runoffMmPerHour)) {
        throw std::invalid_argument("Runoff depth must be finite.");
    }
    return runoffMmPerHour * catchmentAreaM2 / (3600.0 * 1000.0);
}
