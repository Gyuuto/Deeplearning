#ifndef RMSPROP_HPP
#define RMSPROP_HPP

#include <vector>

#ifdef USE_GPU
#include "../clDeviceManager.hpp"
#include "../clMatrix.hpp"
#endif
#include "../Matrix.hpp"

template<template<typename> class Mat, typename Real>
struct RMSProp
{
	const Real gamma = 0.9, eps = 1.0E-8;
	Real EPS;
	
	std::vector<Mat<Real>> r_W, r_b;

	RMSProp( const std::vector<Mat<Real>>& W, const std::vector<Mat<Real>>& b, Real EPS );

	std::pair<std::vector<Mat<Real>>, std::vector<Mat<Real>>> update ( const std::vector<Mat<Real>>& nabla_W, const std::vector<Mat<Real>>& nabla_b );
};

template<typename Real>
struct RMSProp<Matrix, Real>
{
	const Real gamma = 0.9, eps = 1.0E-8;
	Real EPS;
	
	std::vector<Matrix<Real>> r_W, r_b;

	RMSProp( const std::vector<Matrix<Real>>& W, const std::vector<Matrix<Real>>& b, Real EPS )
	{
		this->EPS = EPS;

		if( W.size() == 0 || W[0].m == 0 || W[0].n == 0 )
			r_W = std::vector<Matrix<Real>>();
		else
			r_W = std::vector<Matrix<Real>>(W.size(), Matrix<Real>::zeros(W[0].m, W[0].n));

		if( b.size() == 0 || b[0].m == 0 || b[0].n == 0 )
			r_b = std::vector<Matrix<Real>>();
		else
			r_b = std::vector<Matrix<Real>>(b.size(), Matrix<Real>::zeros(b[0].m, b[0].n));
	}

	std::pair<std::vector<Matrix<Real>>, std::vector<Matrix<Real>>> update ( const std::vector<Matrix<Real>>& nabla_W, const std::vector<Matrix<Real>>& nabla_b )
	{
		std::vector<Matrix<Real>> update_W, update_b;
		if( nabla_W.size() == 0 ) update_W = std::vector<Matrix<Real>>();
		else{
			update_W = std::vector<Matrix<Real>>(nabla_W.size());
			
			for( int i = 0; i < nabla_W.size(); ++i ){
				update_W[i] = Matrix<Real>(nabla_W[i].m, nabla_W[i].n);
#pragma omp parallel for
				for( int j = 0; j < nabla_W[i].m; ++j )
					for( int k = 0; k < nabla_W[i].n; ++k ){
						r_W[i](j,k) = gamma*r_W[i](j,k) + (1.0 - gamma)*nabla_W[i](j,k)*nabla_W[i](j,k);
						update_W[i](j,k) = -EPS / (sqrt(r_W[i](j,k)) + eps)*nabla_W[i](j,k);
					}
			}
		}

		if( nabla_b.size() == 0 ) update_b = std::vector<Matrix<Real>>();
		else{
			update_b = std::vector<Matrix<Real>>(nabla_b.size());
			
			for( int i = 0; i < nabla_b.size(); ++i ){
				update_b[i] = Matrix<Real>(nabla_b[i].m, nabla_b[i].n);
#pragma omp parallel for
				for( int j = 0; j < nabla_b[i].m; ++j )
					for( int k = 0; k < nabla_b[i].n; ++k ){
						r_b[i](j,k) = gamma*r_b[i](j,k) + (1.0 - gamma)*nabla_b[i](j,k)*nabla_b[i](j,k);
						update_b[i](j,k) = -EPS / (sqrt(r_b[i](j,k)) + eps)*nabla_b[i](j,k);
					}
			}
		}

		return std::make_pair(update_W, update_b);
	}
};

#ifdef USE_GPU
template<typename Real>
struct RMSProp<clMatrix, Real>
{
	const Real gamma = 0.9, eps = 1.0E-8;
	Real EPS;

	cl_mem cl_gamma, cl_eps, cl_EPS;
	
	std::vector<clMatrix<Real>> r_W, r_b;

	RMSProp( const RMSProp<clMatrix, Real>& rmsprop )
	{
		this->EPS = rmsprop.EPS;
		this->r_W = rmsprop.r_W;
		this->r_b = rmsprop.r_b;

		cl_int err;
		cl_gamma = clCreateBuffer( cl_device_manager.get_context(), CL_MEM_READ_ONLY, sizeof(Real), NULL, &err);
		cl_eps = clCreateBuffer( cl_device_manager.get_context(), CL_MEM_READ_ONLY, sizeof(Real), NULL, &err);
		cl_EPS = clCreateBuffer( cl_device_manager.get_context(), CL_MEM_READ_ONLY, sizeof(Real), NULL, &err);

		clEnqueueWriteBuffer( cl_device_manager.get_queue(), cl_gamma, CL_TRUE, 0, sizeof(Real), &gamma, 0, NULL, NULL );
		clEnqueueWriteBuffer( cl_device_manager.get_queue(), cl_eps, CL_TRUE, 0, sizeof(Real), &eps, 0, NULL, NULL );
		clEnqueueWriteBuffer( cl_device_manager.get_queue(), cl_EPS, CL_TRUE, 0, sizeof(Real), &EPS, 0, NULL, NULL );
	}
	
	RMSProp( const std::vector<clMatrix<Real>>& W, const std::vector<clMatrix<Real>>& b, Real EPS )
	{
		this->EPS = EPS;

		if( W.size() == 0 || W[0].m == 0 || W[0].n == 0 )
			r_W = std::vector<clMatrix<Real>>();
		else
			r_W = std::vector<clMatrix<Real>>(W.size(), clMatrix<Real>::zeros(W[0].m, W[0].n));

		if( b.size() == 0 || b[0].m == 0 || b[0].n == 0 )
			r_b = std::vector<clMatrix<Real>>();
		else
			r_b = std::vector<clMatrix<Real>>(b.size(), clMatrix<Real>::zeros(b[0].m, b[0].n));

		cl_int err;
		cl_gamma = clCreateBuffer( cl_device_manager.get_context(), CL_MEM_READ_ONLY, sizeof(Real), NULL, &err);
		cl_eps = clCreateBuffer( cl_device_manager.get_context(), CL_MEM_READ_ONLY, sizeof(Real), NULL, &err);
		cl_EPS = clCreateBuffer( cl_device_manager.get_context(), CL_MEM_READ_ONLY, sizeof(Real), NULL, &err);

		clEnqueueWriteBuffer( cl_device_manager.get_queue(), cl_gamma, CL_TRUE, 0, sizeof(Real), &gamma, 0, NULL, NULL );
		clEnqueueWriteBuffer( cl_device_manager.get_queue(), cl_eps, CL_TRUE, 0, sizeof(Real), &eps, 0, NULL, NULL );
		clEnqueueWriteBuffer( cl_device_manager.get_queue(), cl_EPS, CL_TRUE, 0, sizeof(Real), &EPS, 0, NULL, NULL );
	}

	~RMSProp ()
	{
		clReleaseMemObject(cl_gamma);
		clReleaseMemObject(cl_eps);
		clReleaseMemObject(cl_EPS);
	}

	const RMSProp& operator = ( const RMSProp<clMatrix, Real>& rmsprop )
	{
		this->EPS = rmsprop.EPS;
		this->r_W = rmsprop.r_W;
		this->r_b = rmsprop.r_b;

		cl_int err;
		clEnqueueWriteBuffer( cl_device_manager.get_queue(), cl_EPS, CL_TRUE, 0, sizeof(Real), &EPS, 0, NULL, NULL );

		return *this;
	}
	
	std::pair<std::vector<clMatrix<Real>>, std::vector<clMatrix<Real>>> update ( const std::vector<clMatrix<Real>>& nabla_W, const std::vector<clMatrix<Real>>& nabla_b )
	{
		std::vector<clMatrix<Real>> update_W, update_b;
		if( nabla_W.size() == 0 ) update_W = std::vector<clMatrix<Real>>();
		else{
			update_W = std::vector<clMatrix<Real>>(nabla_W.size(), clMatrix<Real>(nabla_W[0].m, nabla_W[0].n));
			
			for( int i = 0; i < nabla_W.size(); ++i ){
				cl_device_manager.set_argument( PRG::RMSPROP, 0, &r_W[i].v );
				cl_device_manager.set_argument( PRG::RMSPROP, 1, &update_W[i].v );
				cl_device_manager.set_argument( PRG::RMSPROP, 2, &nabla_W[i].v );
				cl_device_manager.set_argument( PRG::RMSPROP, 3, &cl_gamma );
				cl_device_manager.set_argument( PRG::RMSPROP, 4, &cl_eps );
				cl_device_manager.set_argument( PRG::RMSPROP, 5, &cl_EPS );
				cl_device_manager.run_kernel( PRG::RMSPROP, nabla_W[i].m*nabla_W[i].n, 1 );
			}
		}

		if( nabla_b.size() == 0 ) update_b = std::vector<clMatrix<Real>>();
		else{
			update_b = std::vector<clMatrix<Real>>(nabla_b.size(), clMatrix<Real>(nabla_b[0].m, nabla_b[0].n));
			
			for( int i = 0; i < nabla_b.size(); ++i ){
				cl_device_manager.set_argument( PRG::RMSPROP, 0, &r_b[i].v );
				cl_device_manager.set_argument( PRG::RMSPROP, 1, &update_b[i].v );
				cl_device_manager.set_argument( PRG::RMSPROP, 2, &nabla_b[i].v );
				cl_device_manager.set_argument( PRG::RMSPROP, 3, &cl_gamma );
				cl_device_manager.set_argument( PRG::RMSPROP, 4, &cl_eps );
				cl_device_manager.set_argument( PRG::RMSPROP, 5, &cl_EPS );
				cl_device_manager.run_kernel( PRG::RMSPROP, nabla_b[i].m*nabla_b[i].n, 1 );
			}
		}

		return std::make_pair(update_W, update_b);
	}
};
#endif

#endif
