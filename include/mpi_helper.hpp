#ifndef MPI_HELPER_HPP
#define MPI_HELPER_HPP

#include <mpi.h>

struct Typecount
{
	MPI_Datatype mpi_type;
	int count;
};

template<typename T>
Typecount get_typecount ( T t )
{
	Typecount tc = { MPI_BYTE, sizeof(T) };
	return tc;
}

template<>
Typecount get_typecount ( int t )
{
	Typecount tc = { MPI_INTEGER, 1 };
	return tc;
}

template<>
Typecount get_typecount ( float t )
{
	Typecount tc = { MPI_FLOAT, 1 };
	return tc;
}

template<>
Typecount get_typecount ( double t )
{
	Typecount tc = { MPI_DOUBLE_PRECISION, 1 };
	return tc;
}

template<>
Typecount get_typecount ( long long t )
{
	Typecount tc = { MPI_LONG_LONG_INT, 1 };
	return tc;
}

#endif
