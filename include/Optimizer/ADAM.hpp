#ifndef ADAM_HPP
#define ADAM_HPP

#include <vector>
#ifdef USE_GPU
#include "../clDeviceManager.hpp"
#include "../clMatrix.hpp"
#endif
#include "../Matrix.hpp"

template<template<typename> class Mat, typename Real>
struct ADAM
{
	const Real beta = 0.9, gamma = 0.999, eps = 1.0E-8;
	Real EPS, beta_ = 1.0, gamma_ = 1.0;
	Real threshold;
	
	std::vector<Mat<Real>> v_W, r_W;
	std::vector<Mat<Real>> v_b, r_b;

	ADAM( const ADAM& adam );
	ADAM( const std::vector<Mat<Real>>& W, const std::vector<Mat<Real>>& b, Real EPS );

	std::pair<std::vector<Mat<Real>>, std::vector<Mat<Real>>> update ( const std::vector<Mat<Real>>& nabla_W, const std::vector<Mat<Real>>& nabla_b );
};

template<typename Real>
struct ADAM<Matrix, Real>
{
	const Real beta = 0.9, gamma = 0.999, eps = 1.0E-8;
	Real EPS, beta_ = 1.0, gamma_ = 1.0;
	Real threshold;
	
	std::vector<Matrix<Real>> v_W, r_W;
	std::vector<Matrix<Real>> v_b, r_b;

	ADAM( const std::vector<Matrix<Real>>& W, const std::vector<Matrix<Real>>& b, Real EPS, Real threshold = -1.0 )
	{
		this->EPS = EPS;
		this->threshold = threshold;

		if( W.size() == 0 || W[0].m == 0 || W[0].n == 0 ){
			v_W = std::vector<Matrix<Real>>();
			r_W = std::vector<Matrix<Real>>();
		}
		else{
			v_W = std::vector<Matrix<Real>>(W.size(), Matrix<Real>::zeros(W[0].m, W[0].n));
			r_W = std::vector<Matrix<Real>>(W.size(), Matrix<Real>::zeros(W[0].m, W[0].n));
		}

		if( b.size() == 0 || b[0].m == 0 || b[0].n == 0 ){
			v_b = std::vector<Matrix<Real>>();
			r_b = std::vector<Matrix<Real>>();
		}
		else{
			v_b = std::vector<Matrix<Real>>(b.size(), Matrix<Real>::zeros(b[0].m, b[0].n));
			r_b = std::vector<Matrix<Real>>(b.size(), Matrix<Real>::zeros(b[0].m, b[0].n));
		}
	}

	std::pair<std::vector<Matrix<Real>>, std::vector<Matrix<Real>>> update ( const std::vector<Matrix<Real>>& nabla_W, const std::vector<Matrix<Real>>& nabla_b )
	{
		beta_ *= beta; gamma_ *= gamma;
		if( nabla_W.size() == 0 ) return std::make_pair(std::vector<Matrix<Real>>(), std::vector<Matrix<Real>>());

		std::vector<Matrix<Real>> update_W(nabla_W.size()), update_b(nabla_b.size());
		for( int i = 0; i < nabla_W.size(); ++i ){
			update_W[i] = Matrix<Real>(nabla_W[i].m, nabla_W[i].n);

			auto tmp_nabla = nabla_W[i];
			if( threshold > 0.0 ) tmp_nabla.clip( threshold );

#pragma omp parallel for
			for( int j = 0; j < update_W[i].m; ++j )
				for( int k = 0; k < update_W[i].n; ++k ){
					v_W[i](j,k) = beta*v_W[i](j,k) + (1.0 - beta)*tmp_nabla(j,k);
					r_W[i](j,k) = gamma*r_W[i](j,k) + (1.0 - gamma)*(tmp_nabla(j,k)*tmp_nabla(j,k));

					auto v_hat = v_W[i](j,k) / (1.0 - beta_);
					auto r_hat = r_W[i](j,k) / (1.0 - gamma_);
					update_W[i](j,k) = -EPS*v_hat / (sqrt(r_hat) + eps);
				}
		}

		for( int i = 0; i < nabla_b.size(); ++i ){
			update_b[i] = Matrix<Real>(nabla_b[i].m, nabla_b[i].n);
#pragma omp parallel for
			for( int j = 0; j < update_b[i].m; ++j )
				for( int k = 0; k < update_b[i].n; ++k ){
					v_b[i](j,k) = beta*v_b[i](j,k) + (1.0 - beta)*nabla_b[i](j,k);
					r_b[i](j,k) = gamma*r_b[i](j,k) + (1.0 - gamma)*(nabla_b[i](j,k)*nabla_b[i](j,k));

					auto v_hat = v_b[i](j,k) / (1.0 - beta_);
					auto r_hat = r_b[i](j,k) / (1.0 - gamma_);
					update_b[i](j,k) = -EPS*v_hat / (sqrt(r_hat) + eps);
				}
		}

		return std::make_pair(update_W, update_b);
	}
};

#ifdef USE_GPU
template<typename Real>
struct ADAM<clMatrix, Real>
{
	const Real beta = 0.9, gamma = 0.999, eps = 1.0E-8;
	Real EPS, beta_ = 1.0, gamma_ = 1.0;
	Real threshold;
	
	cl_mem cl_beta, cl_gamma, cl_eps;
	cl_mem cl_EPS, cl_beta_, cl_gamma_;

	std::vector<clMatrix<Real>> v_W, r_W;
	std::vector<clMatrix<Real>> v_b, r_b;

	ADAM( const ADAM<clMatrix, Real>& adam )
	{
		this->EPS = adam.EPS;
		this->threshold = adam.threshold;

		this->beta_ = adam.beta_;
		this->gamma_ = adam.gamma_;
		this->v_W = adam.v_W; this->r_W = adam.r_W;
		this->v_b = adam.v_b; this->r_b = adam.r_b;
		
		cl_int err;
		cl_beta = clCreateBuffer( cl_device_manager.get_context(), CL_MEM_READ_ONLY, sizeof(Real), NULL, &err);
		cl_gamma = clCreateBuffer( cl_device_manager.get_context(), CL_MEM_READ_ONLY, sizeof(Real), NULL, &err);
		cl_eps = clCreateBuffer( cl_device_manager.get_context(), CL_MEM_READ_ONLY, sizeof(Real), NULL, &err);
		
		cl_EPS = clCreateBuffer( cl_device_manager.get_context(), CL_MEM_READ_ONLY, sizeof(Real), NULL, &err);
		cl_beta_ = clCreateBuffer( cl_device_manager.get_context(), CL_MEM_READ_ONLY, sizeof(Real), NULL, &err);
		cl_gamma_ = clCreateBuffer( cl_device_manager.get_context(), CL_MEM_READ_ONLY, sizeof(Real), NULL, &err);

		clEnqueueWriteBuffer( cl_device_manager.get_queue(), cl_beta, CL_TRUE, 0, sizeof(Real), &beta, 0, NULL, NULL );
		clEnqueueWriteBuffer( cl_device_manager.get_queue(), cl_gamma, CL_TRUE, 0, sizeof(Real), &gamma, 0, NULL, NULL );
		clEnqueueWriteBuffer( cl_device_manager.get_queue(), cl_eps, CL_TRUE, 0, sizeof(Real), &eps, 0, NULL, NULL );
		clEnqueueWriteBuffer( cl_device_manager.get_queue(), cl_EPS, CL_TRUE, 0, sizeof(Real), &EPS, 0, NULL, NULL );
	}

	ADAM( const std::vector<clMatrix<Real>>& W, const std::vector<clMatrix<Real>>& b, Real EPS, Real threshold = -1.0 )
	{
		this->EPS = EPS;
		this->threshold = threshold;
		
		if( W.size() == 0 || W[0].m == 0 || W[0].n == 0 ){
			v_W = std::vector<clMatrix<Real>>();
			r_W = std::vector<clMatrix<Real>>();
		}
		else{
			v_W = std::vector<clMatrix<Real>>(W.size(), clMatrix<Real>::zeros(W[0].m, W[0].n));
			r_W = std::vector<clMatrix<Real>>(W.size(), clMatrix<Real>::zeros(W[0].m, W[0].n));
		}

		if( b.size() == 0 || b[0].m == 0 || b[0].n == 0 ){
			v_b = std::vector<clMatrix<Real>>();
			r_b = std::vector<clMatrix<Real>>();
		}
		else{
			v_b = std::vector<clMatrix<Real>>(b.size(), clMatrix<Real>::zeros(b[0].m, b[0].n));
			r_b = std::vector<clMatrix<Real>>(b.size(), clMatrix<Real>::zeros(b[0].m, b[0].n));
		}

		cl_int err;

		cl_beta = clCreateBuffer( cl_device_manager.get_context(), CL_MEM_READ_ONLY, sizeof(Real), NULL, &err);
		cl_gamma = clCreateBuffer( cl_device_manager.get_context(), CL_MEM_READ_ONLY, sizeof(Real), NULL, &err);
		cl_eps = clCreateBuffer( cl_device_manager.get_context(), CL_MEM_READ_ONLY, sizeof(Real), NULL, &err);
		
		cl_EPS = clCreateBuffer( cl_device_manager.get_context(), CL_MEM_READ_ONLY, sizeof(Real), NULL, &err);
		cl_beta_ = clCreateBuffer( cl_device_manager.get_context(), CL_MEM_READ_ONLY, sizeof(Real), NULL, &err);
		cl_gamma_ = clCreateBuffer( cl_device_manager.get_context(), CL_MEM_READ_ONLY, sizeof(Real), NULL, &err);

		clEnqueueWriteBuffer( cl_device_manager.get_queue(), cl_beta, CL_TRUE, 0, sizeof(Real), &beta, 0, NULL, NULL );
		clEnqueueWriteBuffer( cl_device_manager.get_queue(), cl_gamma, CL_TRUE, 0, sizeof(Real), &gamma, 0, NULL, NULL );
		clEnqueueWriteBuffer( cl_device_manager.get_queue(), cl_eps, CL_TRUE, 0, sizeof(Real), &eps, 0, NULL, NULL );
		clEnqueueWriteBuffer( cl_device_manager.get_queue(), cl_EPS, CL_TRUE, 0, sizeof(Real), &EPS, 0, NULL, NULL );
	}

	~ADAM()
	{
		clReleaseMemObject(cl_beta);
		clReleaseMemObject(cl_gamma);
		clReleaseMemObject(cl_eps);

		clReleaseMemObject(cl_EPS);
		clReleaseMemObject(cl_beta_);
		clReleaseMemObject(cl_gamma_);
	}

	const ADAM& operator = ( const ADAM<clMatrix, Real>& adam )
	{
		this->EPS = adam.EPS;
		this->threshold = adam.threshold;

		this->beta_ = adam.beta_;
		this->gamma_ = adam.gamma_;
		this->v_W = adam.v_W; this->r_W = adam.r_W;
		this->v_b = adam.v_b; this->r_b = adam.r_b;
		
		cl_int err;
		clEnqueueWriteBuffer( cl_device_manager.get_queue(), cl_beta, CL_TRUE, 0, sizeof(Real), &beta, 0, NULL, NULL );
		clEnqueueWriteBuffer( cl_device_manager.get_queue(), cl_gamma, CL_TRUE, 0, sizeof(Real), &gamma, 0, NULL, NULL );
		clEnqueueWriteBuffer( cl_device_manager.get_queue(), cl_eps, CL_TRUE, 0, sizeof(Real), &eps, 0, NULL, NULL );
		clEnqueueWriteBuffer( cl_device_manager.get_queue(), cl_EPS, CL_TRUE, 0, sizeof(Real), &EPS, 0, NULL, NULL );

		return *this;
	}

	std::pair<std::vector<clMatrix<Real>>, std::vector<clMatrix<Real>>> update ( const std::vector<clMatrix<Real>>& nabla_W, const std::vector<clMatrix<Real>>& nabla_b )
	{
		beta_ *= beta; gamma_ *= gamma;
		clEnqueueWriteBuffer( cl_device_manager.get_queue(), cl_beta_, CL_TRUE, 0, sizeof(Real), &beta_, 0, NULL, NULL );
		clEnqueueWriteBuffer( cl_device_manager.get_queue(), cl_gamma_, CL_TRUE, 0, sizeof(Real), &gamma_, 0, NULL, NULL );

		if( nabla_W.size() == 0 ) return std::make_pair(std::vector<clMatrix<Real>>(), std::vector<clMatrix<Real>>());

		std::vector<clMatrix<Real>> update_W(nabla_W.size()), update_b(nabla_b.size());
		for( int i = 0; i < nabla_W.size(); ++i ){
			update_W[i] = clMatrix<Real>(nabla_W[i].m, nabla_W[i].n);

			auto tmp_nabla = nabla_W[i];
			if( threshold > 0.0 ) tmp_nabla.clip(threshold);

			cl_device_manager.set_argument( PRG::ADAM, 0, &v_W[i].v );
			cl_device_manager.set_argument( PRG::ADAM, 1, &r_W[i].v );
			cl_device_manager.set_argument( PRG::ADAM, 2, &update_W[i].v );
			cl_device_manager.set_argument( PRG::ADAM, 3, &tmp_nabla.v );
			cl_device_manager.set_argument( PRG::ADAM, 4, &cl_beta );
			cl_device_manager.set_argument( PRG::ADAM, 5, &cl_gamma );
			cl_device_manager.set_argument( PRG::ADAM, 6, &cl_beta_ );
			cl_device_manager.set_argument( PRG::ADAM, 7, &cl_gamma_ );
			cl_device_manager.set_argument( PRG::ADAM, 8, &cl_EPS );
			cl_device_manager.set_argument( PRG::ADAM, 9, &cl_eps );
			cl_device_manager.run_kernel( PRG::ADAM, nabla_W[i].m*nabla_W[i].n, 1 );
		}

		for( int i = 0; i < nabla_b.size(); ++i ){
			update_b[i] = clMatrix<Real>(nabla_b[i].m, nabla_b[i].n);
			
			cl_device_manager.set_argument( PRG::ADAM, 0, &v_b[i].v );
			cl_device_manager.set_argument( PRG::ADAM, 1, &r_b[i].v );
			cl_device_manager.set_argument( PRG::ADAM, 2, &update_b[i].v );
			cl_device_manager.set_argument( PRG::ADAM, 3, &nabla_b[i].v );
			cl_device_manager.set_argument( PRG::ADAM, 4, &cl_beta );
			cl_device_manager.set_argument( PRG::ADAM, 5, &cl_gamma );
			cl_device_manager.set_argument( PRG::ADAM, 6, &cl_beta_ );
			cl_device_manager.set_argument( PRG::ADAM, 7, &cl_gamma_ );
			cl_device_manager.set_argument( PRG::ADAM, 8, &cl_EPS );
			cl_device_manager.set_argument( PRG::ADAM, 9, &cl_eps );
			cl_device_manager.run_kernel( PRG::ADAM, nabla_b[i].m*nabla_b[i].n, 1 );
		}

		return std::make_pair(update_W, update_b);
	}
};
#endif

#endif
