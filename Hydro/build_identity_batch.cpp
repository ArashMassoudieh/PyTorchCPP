#include "build_identity.h"

#include <iostream>

namespace {

struct HydroBatchBuildIdentityPrinter
{
    HydroBatchBuildIdentityPrinter()
    {
        std::cout << "[build] "
                  << hydroBuildIdentity("HydroBatch").toStdString()
                  << std::endl;
    }
};

HydroBatchBuildIdentityPrinter hydroBatchBuildIdentityPrinter;

} // namespace
