#include "lag_builder.h"

std::vector<std::vector<int>> LagBuilder::buildDefaultLags(int seriesCount) const {
    std::vector<std::vector<int>> lags;
    lags.resize(seriesCount);
    for (auto& v : lags) {
        v = {1, 2, 3};
    }
    return lags;
}
