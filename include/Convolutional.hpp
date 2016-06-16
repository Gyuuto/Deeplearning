#ifndef CONVOLUTIONAL_HPP
#define CONVOLUTIONAL_HPP

#include "Layer.hpp"

class Convolutional : public Layer
{
private:
	int prev_ldu, ldu;
	int m, n, stride, pad;

	std::vector<int> feed_idx, delta_idx;

	Vec r, v;
	double beta_, gamma_;
public:
	Vec bias, d_bias;
	Convolutional( int prev_num_map, int prev_num_unit, int prev_ldu,
				   int num_map, int num_unit, int ldu,
				   int m, int n, int stride, 
				   const std::shared_ptr<Function>& f );

#ifdef USE_MPI
	void init( std::mt19937& mt, MPI_Comm inner_world, MPI_Comm outer_world );
#else
	void init( std::mt19937& mt );
#endif
	void finalize();
	
	std::vector<std::vector<Mat>> calc_gradient ( const std::vector<Mat>& U, const std::vector<Mat>& delta );
	std::vector<Mat> calc_delta ( const std::vector<Mat>& U, const std::vector<Mat>& delta );
	void update_W ( const std::vector<std::vector<Mat>>& dW );

	std::vector<Mat> apply ( const std::vector<Mat>& U, bool use_func = true );
	std::vector<std::vector<Vec>> apply ( const std::vector<std::vector<Vec>>& u, bool use_func = true );
	std::vector<Mat> deconvolution ( const std::vector<Mat>& U );
	std::vector<std::vector<Vec>> deconvolution ( const std::vector<std::vector<Vec>>& u );

	void set_W ( const std::string& filename );
	void output_W ( const std::string& filename );
	
#ifdef USE_MPI
	void param_mix ();
#endif
};

Convolutional::Convolutional( int prev_num_map, int prev_num_unit, int prev_ldu,
							  int num_map, int num_unit, int ldu,
							  int m, int n, int stride, 
							  const std::shared_ptr<Function>& f )
{
	this->prev_num_map = prev_num_map;
	this->prev_num_unit = prev_num_unit;
	this->prev_ldu = prev_ldu;
	
	this->num_map = num_map;
	this->num_unit = num_unit;
	this->ldu = ldu;	

	t_apply = t_delta = t_grad = 0.0;
	t_apply_init = t_apply_gemm = t_apply_repl = 0.0;
	t_delta_init = t_delta_gemm = t_delta_repl = 0.0;
	t_grad_init = t_grad_gemm = t_grad_repl = 0.0;

	this->m = m; this->n = n; this->stride = stride; this->pad = m/2;

	int rank = 0;
#ifdef USE_MPI
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
	if( num_unit%ldu != 0 )
		if( rank == 0 ){
			printf("WARNING : Wrong leading dimension of output on Convolution layer.\n");
			printf("          Layer details : output size[%d x %d], filter size[%d x %d], stride %d, padding %d, number of map %d.\n", num_unit/ldu, ldu, m, n, stride, pad, num_map);
		}
	if( prev_num_unit%prev_ldu != 0 )
		if( rank == 0 ){
			printf("WARNING : Wrong leading dimension of input on Convolution layer.\n");
			printf("          Layer details : output size[%d x %d], filter size[%d x %d], stride %d, padding %d, number of map %d.\n", num_unit/ldu, ldu, m, n, stride, pad, num_map);
		}
	if( ldu != (prev_ldu + 2*pad - n)/stride + 1 )
		if( rank == 0 ){
			printf("WARNING : Wrong output image width on Convolution layer.\n");
			printf("          Estimate width = %d.\n", (prev_ldu + 2*pad - n)/stride + 1);
			printf("          Layer details : output size[%d x %d], filter size[%d x %d], stride %d, padding %d, number of map %d.\n", num_unit/ldu, ldu, m, n, stride, pad, num_map);
		}
	if( num_unit/ldu != (prev_num_unit/prev_ldu + 2*pad - m)/stride + 1 )
		if( rank == 0 ){
			printf("WARNING : Wrong output image height on Convolution layer.\n");
			printf("          Estimate height = %d.\n", (prev_num_unit/prev_ldu + 2*pad - m)/stride + 1);
			printf("          Layer details : output size[%d x %d], filter size[%d x %d], stride %d, padding %d, number of map %d.\n", num_unit/ldu, ldu, m, n, stride, pad, num_map);
		}
	
	func = f;

	beta_ = 1.0; gamma_ = 1.0;
	for( int i = 0; i < num_map; ++i ){
		W.emplace_back(prev_num_map);
		for( int j = 0; j < prev_num_map; ++j ){
			W[i][j] = Mat(this->m, this->n);
		}
	}
}

#ifdef USE_MPI
void Convolutional::init ( std::mt19937& mt, MPI_Comm inner_world, MPI_Comm outer_world )
#else
void Convolutional::init ( std::mt19937& mt )
#endif
{
#ifdef USE_MPI
	this->inner_world = inner_world;
	this->outer_world = outer_world;

	MPI_Comm_rank(inner_world, &rank);
	MPI_Comm_size(inner_world, &nprocs);

#endif

	// calculate indices of feed forward
	{
		int my_size, my_offset;
		const int Y = prev_num_unit/prev_ldu, X = prev_ldu;
		const int gap = prev_ldu + 2*pad;
#ifdef USE_MPI
		my_size = (rank+1)*num_unit/nprocs - rank*num_unit/nprocs;
		my_offset = rank*num_unit/nprocs;
#else
		my_size = num_unit; my_offset = 0;
#endif
		feed_idx.resize(my_size*m*n);
#pragma omp parallel for schedule(auto)
		for( int i = 0; i < my_size; ++i ){
			int x = (i + my_offset)%ldu, y = (i + my_offset)/ldu;
			for( int s = 0; s < n; ++s )
				for( int t = 0; t < m; ++t ){
					int idx = stride*x + s + t*gap + stride*y*gap;
					int nx = idx%gap - pad, ny = idx/gap - pad;

					if( nx < 0 || nx >= X || ny < 0 || ny >= Y ){
						feed_idx[i*m*n + t*n + s] = -1;
						continue;
					}
					feed_idx[i*m*n + t*n + s] = ny*prev_ldu + nx;
				}
		}
	}

	{
		int my_size, my_offset;
#ifdef USE_MPI
		my_size = (rank+1)*prev_num_unit/nprocs - rank*prev_num_unit/nprocs;
		my_offset = rank*prev_num_unit/nprocs;
#else
		my_size = prev_num_unit; my_offset = 0;
#endif

		const int X = prev_ldu, Y = prev_num_unit/prev_ldu;
		const int gap = prev_ldu + 2*pad;
#ifdef USE_MPI
		const int tmp_size = (rank+1)*num_unit/nprocs - rank*num_unit/nprocs; 
		const int tmp_offset = rank*num_unit/nprocs;
#else
		const int tmp_size = num_unit;
		const int tmp_offset = 0;
#endif
		int l_idx = std::max(0, tmp_offset - m*prev_ldu/2);
		int r_idx = std::min(num_unit, tmp_offset + tmp_size + m*prev_ldu/2);
		delta_idx.resize(m*n*(r_idx - l_idx));
#pragma omp parallel for schedule(auto)
		for( int j = l_idx; j < r_idx; ++j ){
			int x = j%ldu, y = j/ldu;
			for( int t = 0; t < m; ++t )
				for( int s = 0; s < n; ++s ){
					int idx = stride*x + s + t*gap + stride*y*gap;
					int nx = idx%gap - pad, ny = idx/gap - pad;

					if( nx < 0 || nx >= X || ny < 0 || ny >= Y ){
						delta_idx[(j-l_idx)*m*n + t*n + s] = -1;
						continue;
					}
					if( ny*prev_ldu + nx < my_offset || my_offset + my_size <= ny*prev_ldu + nx ){
						delta_idx[(j-l_idx)*m*n + t*n + s] = -1;
						continue;
					}
					delta_idx[(j-l_idx)*m*n + t*n + s] = ny*prev_ldu + nx - my_offset;
				}
		}
	}

	const double r = sqrt(6.0/(num_unit + prev_num_unit));
	std::normal_distribution<double> d_rand(0.0, 1.0E-1);

	bias = Vec(num_map, 0.0); d_bias = Vec(num_map, 0.0);
	this->r = Vec(num_map, 0.0); v = Vec(num_map, 0.0);
	for( int i = 0; i < num_map; ++i ){
		for( int j = 0; j < prev_num_map; ++j ){
			for( int k = 0; k < W[i][j].m; ++k )
				for( int l = 0; l < W[i][j].m; ++l )
					W[i][j](k,l) = d_rand(mt);
		}				
	}
}

void Convolutional::finalize ()
{
}

std::vector<std::vector<Convolutional::Mat>> Convolutional::calc_gradient ( const std::vector<Mat>& U, const std::vector<Mat>& delta )
{
	auto tot_beg = std::chrono::system_clock::now();
	auto beg = tot_beg;

	int offset = 0, my_size = U[0].m;
#ifdef USE_MPI
	offset = rank*my_size/nprocs;
	my_size = (rank+1)*my_size/nprocs - rank*my_size/nprocs;
#endif

	std::vector<std::vector<Mat>> nabla(num_map);
	for( int i = 0; i < num_map; ++i ){
		nabla[i] = std::vector<Mat>(prev_num_map);
		for( int j = 0; j < prev_num_map; ++j )
			nabla[i][j] = Mat(W[i][j].m, W[i][j].n);
	}

	std::vector<Mat> U_(prev_num_map);
	for( int i = 0; i < prev_num_map; ++i )
		U_[i] = (*prev_func)(U[i], false);
	auto end = std::chrono::system_clock::now();
	t_grad_init += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

	const int Y = prev_num_unit/prev_ldu, X = prev_ldu;
	const int Y_ = num_unit/ldu, X_ = ldu;
	Mat nabla_mat = Mat::zeros(m*n*num_map, prev_num_map);

	for( int i = 0; i < delta[0].n; ++i ){
		auto beg = std::chrono::system_clock::now();
		Mat delta_mat = Mat::zeros(m*n*num_map, my_size);
		Mat U_mat(my_size, prev_num_map);

		const int gap = prev_ldu + 2*pad;
#ifdef USE_MPI
		const int tmp_size = (rank+1)*num_unit/nprocs - rank*num_unit/nprocs; 
		const int tmp_offset = rank*num_unit/nprocs;
#else
		const int tmp_size = num_unit;
		const int tmp_offset = 0;
#endif
		int l_idx = std::max(0, tmp_offset - m*prev_ldu/2);
		int r_idx = std::min(num_unit, tmp_offset + tmp_size + m*prev_ldu/2);

#pragma omp parallel for schedule(auto) firstprivate(l_idx, r_idx)
		for( int j = l_idx; j < r_idx; ++j ){
			for( int t = 0; t < m; ++ t )
				for( int s = 0; s < n; ++s )
					if( delta_idx[(j-l_idx)*m*n + t*n + s] != -1 )
						for( int k = 0; k < num_map; ++k )
							delta_mat(k*m*n + s*m + t, delta_idx[(j-l_idx)*m*n + t*n + s]) = delta[k](j, i);
		}

#pragma omp parallel for schedule(auto) firstprivate(my_size, offset)
		for( int k = 0; k < my_size; ++k )
			for( int j = 0; j < prev_num_map; ++j )
				U_mat(k, j) = U_[j](offset + k, i);
		auto end = std::chrono::system_clock::now();
		t_grad_init += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;
		beg = std::chrono::system_clock::now();
		nabla_mat += delta_mat*U_mat;
		end = std::chrono::system_clock::now();
		t_grad_gemm += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;
	}

	beg = std::chrono::system_clock::now();
	for( int i = 0; i < num_map; ++i ){
		for( int j = 0; j < prev_num_map; ++j ){
			for( int k = 0; k < m; ++k )
				for( int l = 0; l < n; ++l )
					nabla[i][j](k, l) = nabla_mat(i*m*n + k*n + l, j);
		}
		
		double sum = 0.0;
		for( int k = 0; k < delta[i].m; ++k )
			for( int j = 0; j < delta[i].n; ++j )
				sum += delta[i](k,j);
		d_bias[i] = sum / delta[i].n;
	}
	end = std::chrono::system_clock::now();
	t_grad_repl += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;
	
#ifdef USE_MPI
	for( int i = 0; i < num_map; ++i )
		for( int j = 0; j < prev_num_map; ++j )
			MPI_Allreduce(MPI_IN_PLACE, &nabla[i][j](0,0), m*n, MPI_DOUBLE_PRECISION, MPI_SUM, inner_world);
#endif
	end = std::chrono::system_clock::now();
	t_grad += std::chrono::duration_cast<std::chrono::nanoseconds>(end - tot_beg).count()/1e9;

	return nabla;				
}

std::vector<Convolutional::Mat> Convolutional::calc_delta ( const std::vector<Mat>& U, const std::vector<Mat>& delta )
{
	auto tot_beg = std::chrono::system_clock::now();
	auto beg = tot_beg;

	int my_size = prev_num_unit, my_offset = 0;
#ifdef USE_MPI
	std::vector<int> size(nprocs), offset(nprocs);
	for( int i = 0; i < nprocs; ++i ){
		size[i] = ((i+1)*prev_num_unit/nprocs - i*prev_num_unit/nprocs)*prev_num_map;
		offset[i] = i*prev_num_unit/nprocs*prev_num_map;
	}

	my_offset = offset[rank] / prev_num_map;
	my_size = size[rank] / prev_num_map;
#endif

	const int X = prev_ldu, Y = prev_num_unit/prev_ldu;
	const int X_ = ldu, Y_ = num_unit/ldu;
	std::vector<Mat> tmp(prev_num_map), nx_delta(prev_num_map);

	for( int i = 0; i < prev_num_map; ++i ) tmp[i] = Mat(prev_num_unit, U[0].n);

	Mat kernel(m*n*num_map, prev_num_map);
#pragma omp parallel for schedule(auto)
	for( int i = 0; i < num_map; ++i )
		for( int j = 0; j < prev_num_map; ++j )
			for( int k = 0; k < m; ++ k )
				for( int l = 0; l < n; ++l )
					kernel(i*(m*n) + l*n + k, j) = W[i][j](k, l);
	auto end = std::chrono::system_clock::now();
	t_delta_init += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;
	
	for( int i = 0; i < delta[0].n; ++i ){
		auto beg = std::chrono::system_clock::now();;
		Mat input_image = Mat::zeros(my_size, m*n*num_map);
 
		const int gap = prev_ldu + 2*pad;
#ifdef USE_MPI
		const int tmp_size = (rank+1)*num_unit/nprocs - rank*num_unit/nprocs; 
		const int tmp_offset = rank*num_unit/nprocs;
#else
		const int tmp_size = num_unit;
		const int tmp_offset = 0;
#endif
		int l_idx = std::max(0, tmp_offset - m*prev_ldu/2);
		int r_idx = std::min(num_unit, tmp_offset + tmp_size + m*prev_ldu/2);
#pragma omp parallel for schedule(auto) firstprivate(l_idx, r_idx)
		for( int j = l_idx; j < r_idx; ++j ){
			// int x = j%ldu, y = j/ldu;
			// for( int t = 0; t < m; ++t )
			// 	for( int s = 0; s < n; ++s ){
			// 		int idx = stride*x + s + t*gap + stride*y*gap;
			// 		int nx = idx%gap - pad, ny = idx/gap - pad;

			// 		if( nx < 0 || nx >= X || ny < 0 || ny >= Y ) continue;
			// 		if( ny*prev_ldu + nx < my_offset || my_offset + my_size <= ny*prev_ldu + nx )
			// 			continue;

			// 		for( int k = 0; k < num_map; ++k )
			// 			input_image(ny*prev_ldu + nx - my_offset, m*n*k + t*n + s) = delta[k](j, i);
			// 	}
			for( int t = 0; t < m; ++t )
				for( int s = 0; s < n; ++s )
					if( delta_idx[(j-l_idx)*m*n + t*n + s] != -1 )
						for( int k = 0; k < num_map; ++k )
							input_image(delta_idx[(j-l_idx)*m*n + t*n + s], m*n*k + t*n + s) = delta[k](j, i);
		}
		auto end = std::chrono::system_clock::now();
		t_delta_init += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

		beg = std::chrono::system_clock::now();
#ifdef USE_MPI
		Mat tmp_output_image = input_image * kernel;
		Mat output_image(prev_num_unit, prev_num_map);
		MPI_Allgatherv(&tmp_output_image(0,0), size[rank], MPI_DOUBLE_PRECISION,
					   &output_image(0,0), &size[0], &offset[0], MPI_DOUBLE_PRECISION, inner_world);
#else
		Mat output_image = input_image * kernel;
#endif
		end = std::chrono::system_clock::now();
		t_delta_gemm += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

		beg = std::chrono::system_clock::now();
		for( int j = 0; j < prev_num_map; ++j )
			for( int k = 0; k < prev_num_unit; ++k )
				tmp[j](k, i) = output_image(k, j);
		end = std::chrono::system_clock::now();
		t_delta_repl += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;
	}

	for( int i = 0; i < prev_num_map; ++i )
		nx_delta[i] = Mat::hadamard(tmp[i], (*prev_func)(U[i], true));
	end = std::chrono::system_clock::now();
	t_delta += std::chrono::duration_cast<std::chrono::nanoseconds>(end - tot_beg).count()/1e9;

	return nx_delta;
}

void Convolutional::update_W ( const std::vector<std::vector<Mat>>& dW )
{
	const double a_beta = 0.9, a_gamma = 0.999, a_eps = 1.0E-8;
	beta_ *= a_beta; gamma_ *= a_gamma;
	for( int i = 0; i < num_map; ++i ){
		for( int j = 0; j < prev_num_map; ++j )
			W[i][j] = W[i][j] + dW[i][j];

		v[i] = a_beta*v[i] + (1.0 - a_beta)*d_bias[i];
		r[i] = a_gamma*r[i] + (1.0 - a_gamma)*d_bias[i]*d_bias[i];
		bias[i] -= 0.001*v[i]/(1.0 - beta_)/(sqrt(r[i]/(1.0 - gamma_)+a_eps));
	}
}

std::vector<Convolutional::Mat> Convolutional::apply ( const std::vector<Mat>& U, bool use_func )
{
	auto tot_beg = std::chrono::system_clock::now();
	auto beg = tot_beg;

	int my_size = num_unit, my_offset = 0;
#ifdef USE_MPI
	std::vector<int> size(nprocs), offset(nprocs);
	for( int i = 0; i < nprocs; ++i ){		
		size[i] = ((i+1)*num_unit/nprocs - i*num_unit/nprocs)*num_map;
		offset[i] = i*num_unit/nprocs*num_map;
	}

	my_offset = offset[rank] / num_map;
	my_size = size[rank] / num_map;
#endif

	const int Y = prev_num_unit/prev_ldu, X = prev_ldu;
	std::vector<Mat> ret(num_map);

	for( int i = 0; i < num_map; ++i ) ret[i] = Mat(num_unit, U[0].n);

	Mat kernel(m*n*prev_num_map, num_map);
	for( int i = 0; i < num_map; ++i )
		for( int j = 0; j < prev_num_map; ++j )
			for( int k = 0; k < m; ++ k )
				for( int l = 0; l < n; ++l )
					kernel(j*(m*n) + l*n + k, i) = W[i][j](k, l);
	auto end = std::chrono::system_clock::now();
	t_apply_init += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;
			
	for( int i = 0; i < U[0].n; ++i ){
		auto beg = std::chrono::system_clock::now();
		Mat input_image = Mat::zeros(my_size, m*n*prev_num_map);

#pragma omp parallel for schedule(auto)
		for( int j = 0; j < my_size; ++j ){
			for( int k = 0; k < prev_num_map; ++k )
				for( int s = 0; s < m; ++ s )
					for( int t = 0; t < n; ++ t )
						if( feed_idx[j*m*n + s*n + t] != -1 )
							input_image(j, m*n*k + s*n + t) = U[k](feed_idx[j*m*n + s*n + t], i);
		}
		auto end = std::chrono::system_clock::now();
		t_apply_init += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

		beg = std::chrono::system_clock::now();
#ifdef USE_MPI
		Mat tmp_output_image = input_image * kernel;
		Mat output_image(num_unit, num_map);
		MPI_Allgatherv(&tmp_output_image(0,0), size[rank], MPI_DOUBLE_PRECISION,
					   &output_image(0,0), &size[0], &offset[0], MPI_DOUBLE_PRECISION, inner_world);
#else
		Mat output_image = input_image * kernel;
#endif
		end = std::chrono::system_clock::now();
		t_apply_gemm += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

		beg = std::chrono::system_clock::now();
#pragma omp parallel for schedule(auto)
		for( int k = 0; k < num_unit; ++k )
			for( int j = 0; j < num_map; ++j )
				ret[j](k, i) = output_image(k, j);
		end = std::chrono::system_clock::now();
		t_apply_repl += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;
	}
	
	for( int i = 0; i < num_map; ++i )
		for( int j = 0; j < ret[i].m; ++j )
			for( int k = 0; k < ret[i].n; ++k )
				ret[i](j,k) += bias[i];

	if( use_func )
		for( int i = 0; i < num_map; ++i )
			ret[i] = (*func)(ret[i], false);
	end = std::chrono::system_clock::now();
	t_apply += std::chrono::duration_cast<std::chrono::nanoseconds>(end - tot_beg).count()/1e9;
	
	return ret;
}

std::vector<std::vector<Convolutional::Vec>> Convolutional::apply ( const std::vector<std::vector<Vec>>& u, bool use_func )
{
	std::vector<Mat> tmp(prev_num_map);
	for( int i = 0; i < prev_num_map; ++i )
		tmp[i] = Mat(u[0][0].size(), u.size());

	for( int i = 0; i < prev_num_map; ++i )
		for( int j = 0; j < u[0][0].size(); ++j )
			for( int k = 0; k < u.size(); ++k )
				tmp[i](j,k) = u[k][i][j];
	
	auto U = apply(tmp, use_func);
	std::vector<std::vector<Vec>> ret(U[0].n);
	for( int i = 0; i < U[0].n; ++i ){
		ret[i] = std::vector<Vec>(U.size(), Vec(U[0].m));
		for( int j = 0; j < U.size(); ++j )
			for( int k = 0; k < U[0].m; ++k )
				ret[i][j][k] = U[j](k,i);
	}

	return ret;
}

std::vector<Convolutional::Mat> Convolutional::deconvolution ( const std::vector<Mat>& U )
{
	const int Y = prev_num_unit/prev_ldu, X = prev_ldu;
	std::vector<Mat> ret(prev_num_map);

	int i, j, k, x, y, s, t;
#pragma omp parallel for default(none) \
	private(i,j,k,s,t,y,x) shared(ret, U)
	for( i = 0; i < prev_num_map; ++i ){
		ret[i] = Mat(prev_num_unit, U[0].n);
		for( j = 0; j < num_map; ++j ){
			auto U_ = (*func)(U[j], false);
			for( k = 0; k < U[0].n; ++k )
				for( x = 0; x < X; ++x )
					for( y = 0; y < Y; ++ y ){
						for( s = -m/2; s < (m+1)/2; ++s )
							for( t = -n/2; t < (n+1)/2; ++t ){
								int nx = (x - s),
									ny = (y - t);
								if( nx < 0 || nx >= X || ny < 0 || ny >= Y ) continue;
								nx /= stride; ny /= stride;
								ret[i](x+prev_ldu*y,k) += W[j][i](s+m/2,t+n/2)*(U_(nx+ldu*ny,k) - bias[j]);
							}
					}
		}
	}

	return ret;
}

std::vector<std::vector<Convolutional::Vec>> Convolutional::deconvolution ( const std::vector<std::vector<Vec>>& u )
{
	std::vector<Mat> tmp(num_map);
	for( int i = 0; i < num_map; ++i )
		tmp[i] = Mat(u[0][0].size(), u.size());

	for( int i = 0; i < num_map; ++i )
		for( int j = 0; j < u[0][0].size(); ++j )
			for( int k = 0; k < u.size(); ++k )
				tmp[i](j,k) = u[k][i][j];
	
	auto U = deconvolution(tmp);
	std::vector<std::vector<Vec>> ret(U[0].n);
	for( int i = 0; i < U[0].n; ++i ){
		ret[i] = std::vector<Vec>(U.size(), Vec(U[0].m));
		for( int j = 0; j < U.size(); ++j )
			for( int k = 0; k < U[0].m; ++k )
				ret[i][j][k] = U[j](k,i);
	}

	return ret;	
}

void Convolutional::set_W ( const std::string& filename )
{
	std::ifstream ifs(filename, std::ios::binary);

	for( int i = 0; i < num_map; ++i )
		for( int j = 0; j < prev_num_map; ++j ){
			ifs.read((char*)&W[i][j].m, sizeof(W[i][j].m));
			ifs.read((char*)&W[i][j].n, sizeof(W[i][j].n));
			for( int k = 0; k < W[i][j].m; ++k )
				for( int l = 0; l < W[i][j].n; ++l ){
					ifs.read((char*)&W[i][j](k,l), sizeof(W[i][j](k,l)));
				}
		}
}

void Convolutional::output_W ( const std::string& filename )
{
#ifdef USE_MPI
	if( rank == 0 ){
#endif
		std::ofstream ofs(filename, std::ios::binary);
		
		for( int i = 0; i < num_map; ++i )
			for( int j = 0; j < prev_num_map; ++j ){
				ofs.write((char*)&W[i][j].m, sizeof(W[i][j].m));
				ofs.write((char*)&W[i][j].n, sizeof(W[i][j].n));
				for( int k = 0; k < W[i][j].m; ++k )
					for( int l = 0; l < W[i][j].n; ++l )
						ofs.write((char*)&W[i][j](k,l), sizeof(W[i][j](k,l)));
			}	
#ifdef USE_MPI
	}
#endif
}

#ifdef USE_MPI
void Convolutional::param_mix ()
{
	int nprocs;
	MPI_Comm_size(outer_world, &nprocs);
	if( W.size() == 0 ) return;

	int cnt = W.size()*W[0].size()*W[0][0].m*W[0][0].n + bias.size();
	std::vector<double> w(cnt);

	int idx = 0;
	for( int i = 0; i < W.size(); ++i )
		for( int j = 0; j < W[i].size(); ++j )
			for( int k = 0; k < W[i][j].m; ++k )
				for( int l = 0; l < W[i][j].n; ++l )
					w[idx++] = W[i][j](k,l);

	for( int i = 0; i < bias.size(); ++i ) w[idx++] = bias[i];
	
	MPI_Allreduce(MPI_IN_PLACE, &w[0], cnt, MPI_DOUBLE_PRECISION, MPI_SUM, outer_world);

	idx = 0;
	for( int i = 0; i < W.size(); ++i )
		for( int j = 0; j < W[i].size(); ++j )
			for( int k = 0; k < W[i][j].m; ++k )
				for( int l = 0; l < W[i][j].n; ++l )
					W[i][j](k,l) = w[idx++]/nprocs;
	for( int i = 0; i < bias.size(); ++i ) bias[i] = w[idx++]/nprocs;
}
#endif

#endif
