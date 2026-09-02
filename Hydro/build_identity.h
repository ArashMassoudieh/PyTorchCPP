#pragma once

#include <QString>

#ifndef HYDRO_GIT_COMMIT
#define HYDRO_GIT_COMMIT unknown
#endif

#define HYDRO_STRINGIZE_IMPL(x) #x
#define HYDRO_STRINGIZE(x) HYDRO_STRINGIZE_IMPL(x)

inline QString hydroBuildCommit()
{
    // Stringify the compiler definition inside C++ rather than relying on
    // qmake/shell quote escaping. This works whether HYDRO_GIT_COMMIT reaches
    // the compiler as d09ca2b, 2381715, "d09ca2b", or unknown.
    QString value = QString::fromLatin1(HYDRO_STRINGIZE(HYDRO_GIT_COMMIT));
    if (value.size() >= 2 && value.startsWith('"') && value.endsWith('"'))
        value = value.mid(1, value.size() - 2);
    return value;
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
