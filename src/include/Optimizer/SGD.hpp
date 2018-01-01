#ifndef SGD_HPP
#define SGD_HPP

#include <vector>

template<template<typename> class Mat, typename Real>
struct SGD
{
	Real EPS;

	SGD( const std::vector<Mat<Real>>& W, const std::vector<Mat<Real>>& b, Real EPS )
	{
		this->EPS = EPS;
	}

	std::pair<std::vector<Mat<Real>>, std::vector<Mat<Real>>> update ( const std::vector<Mat<Real>>& nabla_W, const std::vector<Mat<Real>>& nabla_b )
	{
		if( nabla_W.size() == 0 ) return std::make_pair(std::vector<Mat<Real>>(), std::vector<Mat<Real>>());

		std::vector<Mat<Real>> update_W(nabla_W.size()), update_b(nabla_b.size());
		for( int i = 0; i < nabla_W.size(); ++i ){
			update_W[i] = -EPS*nabla_W[i];
		}

		for( int i = 0; i < nabla_b.size(); ++i ){
			update_b[i] = -EPS*nabla_b[i];
		}

		return std::make_pair(update_W, update_b);
	}
};

#endif
