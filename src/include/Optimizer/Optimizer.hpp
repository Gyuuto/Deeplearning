#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include <vector>

template<template<typename> class Mat, typename Real>
class Optimizer
{
private:
public:
	virtual void update ( const std::vector<Mat<Real>>& nabla_W, const std::vector<Mat<Real>>& nabla_b, std::vector<Mat<Real>>& update_W, std::vector<Mat<Real>>& update_b ) = 0;
};

#endif
