#pragma once

#include <QString>

#ifndef HYDRO_GIT_COMMIT
#define HYDRO_GIT_COMMIT "unknown"
#endif

inline QString hydroBuildCommit()
{
    return QStringLiteral(HYDRO_GIT_COMMIT);
}

inline QString hydroBuildTimestamp()
{
    return QStringLiteral(__DATE__ " " __TIME__);
}

inline QString hydroBuildIdentity(const QString& target)
{
    return QString("%1 | commit %2 | built %3")
        .arg(target, hydroBuildCommit(), hydroBuildTimestamp());
}
