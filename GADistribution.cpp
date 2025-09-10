// Distribution.cpp: implementation of the GADistribution class.
//
//////////////////////////////////////////////////////////////////////

#include "GADistribution.h"
#include "math.h"
#include <stdlib.h>

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

GADistribution::GADistribution()
{
		
}

GADistribution::GADistribution(int nn)
{
	n = nn;
	e.resize(n);
	s.resize(n);
}

GADistribution::~GADistribution()
{
	
}

GADistribution::GADistribution(const GADistribution &C)
{
	n = C.n;
	e.resize(n);
	s.resize(n);
	for (int i=0; i<n; i++)
	{
		e[i] = C.e[i];
		s[i] = C.s[i];
	}


}

GADistribution GADistribution::operator = (const GADistribution &C)
{
	n = C.n;
	e.resize(n);
	s.resize(n);
	for (int i=0; i<n; i++)
	{
		e[i] = C.e[i];
		s[i] = C.s[i];
	}

	return *this;

}

int GADistribution::GetRand()
{
	double x = GetRndUniF(0,1);
	int ii = 0;
	for (int i=0; i<n-1; i++)
	{	
		if (x<e[i] && x>s[i])
			ii = i;
	}
	return ii;

}


double GADistribution::GetRndUniF(double xmin, double xmax)
{
    double a = double(rand());
    double k = double(RAND_MAX);
    return a/k*(xmax-xmin) + xmin;
}
