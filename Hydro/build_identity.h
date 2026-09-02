#pragma once

#include <QString>

#ifndef HYDRO_GIT_COMMIT
#define HYDRO_GIT_COMMIT "unknown"
#endif

inline QString hydroBuildCommit()
{
    // HYDRO_GIT_COMMIT is supplied by qmake as a quoted compiler define.
    // QStringLiteral() requires a literal token at the call site and is not
    // safe with an externally defined macro on all Qt versions. fromLatin1()
    // accepts the expanded const char[] directly and works with Qt 5 and Qt 6.
    return QString::fromLatin1(HYDRO_GIT_COMMIT);
}

inline QString hydroBuildTimestamp()
{
    return QString::fromLatin1(__DATE__ " " __TIME__);
}

inline QString hydroBuildIdentity(const QString& target)
{
    return QString("%1 | commit %2 | built %3")
        .arg(target, hydroBuildCommit(), hydroBuildTimestamp());
}
