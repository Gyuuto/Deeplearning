#ifndef POOLING_HPP
#define POOLING_HPP

#include "Layer.hpp"

class Pooling : public Layer
{
private:
	typedef double real;
	
	int prev_ldu, ldu;
	int m, n, stride, pad;
	std::vector<Mat> S;
public:
	Pooling( int prev_num_map, int prev_num_unit, int prev_ldu,
			 int num_map, int num_unit, int ldu,
			 int m, int n, int stride, 
			 const std::shared_ptr<Function<real>>& f );
	
#ifdef USE_MPI
	void init( std::mt19937& m, MPI_Comm inner_world, MPI_Comm outer_world );
#else
	void init( std::mt19937& m );
#endif
	void finalize();

	std::vector<std::vector<Mat>> calc_gradient ( const std::vector<Mat>& U, const std::vector<Mat>& delta );
	std::vector<Mat> calc_delta ( const std::vector<Mat>& U, const std::vector<Mat>& delta );
	void update_W ( const std::vector<std::vector<Mat>>& dW );

	std::vector<Mat> apply ( const std::vector<Mat>& U, bool use_func = true );
	std::vector<std::vector<Vec>> apply ( const std::vector<std::vector<Vec>>& u, bool use_func = true );
	std::vector<Mat> unpooling ( const std::vector<Mat>& U );
	std::vector<std::vector<Vec>> unpooling ( const std::vector<std::vector<Vec>>& u );

	void set_W ( const std::string& filename );
	void output_W ( const std::string& filename );

#ifdef USE_MPI
	void param_mix ();
#endif
};

Pooling::Pooling( int prev_num_map, int prev_num_unit, int prev_ldu,
				  int num_map, int num_unit, int ldu,
				  int m, int n, int stride, 
				  const std::shared_ptr<Function<real>>& f )
{
	this->prev_num_map = prev_num_map;
	this->prev_num_unit = prev_num_unit;
	this->prev_ldu = prev_ldu;

	this->num_map = num_map;
	this->num_unit = num_unit;
	this->ldu = ldu;

	t_apply = t_delta = t_grad = 0.0;
	t_apply_init = t_apply_gemm = t_apply_repl = t_apply_comm = 0.0;
	t_delta_init = t_delta_gemm = t_delta_repl = t_delta_comm = 0.0;
	t_grad_init = t_grad_gemm = t_grad_repl = t_grad_comm = 0.0;

	this->m = m; this->n = n; this->stride = stride; this->pad = 0;

	int rank = 0;
#ifdef USE_MPI
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
	if( num_unit%ldu != 0 )
		if( rank == 0 ){
			printf("WARNING : Wrong leading dimension of output on Pooling layer.\n");
			printf("          Layer details : output size[%d x %d], filter size[%d x %d], stride %d, padding %d, number of map %d.\n", num_unit/ldu, ldu, m, n, stride, pad, num_map);
		}
	if( prev_num_unit%prev_ldu != 0 )
		if( rank == 0 ){
			printf("WARNING : Wrong leading dimension of input on Pooling layer.\n");
			printf("          Layer details : output size[%d x %d], filter size[%d x %d], stride %d, padding %d, number of map %d.\n", num_unit/ldu, ldu, m, n, stride, pad, num_map);
		}
	if( ldu != (prev_ldu + 2*pad - n)/stride + 1 )
		if( rank == 0 ){
			printf("WARNING : Wrong output image width on Pooling layer.\n");
			printf("          Estimate width = %d.\n", (prev_ldu + 2*pad - n)/stride + 1);
			printf("          Layer details : output size[%d x %d], filter size[%d x %d], stride %d, padding %d, number of map %d.\n", num_unit/ldu, ldu, m, n, stride, pad, num_map);
		}
	if( num_unit/ldu != (prev_num_unit/prev_ldu + 2*pad - m)/stride + 1 )
		if( rank == 0 ){
			printf("WARNING : Wrong output image height on Pooling layer.\n");
			printf("          Estimate height = %d.\n", (prev_num_unit/prev_ldu + 2*pad - m)/stride + 1);
			printf("          Layer details : output size[%d x %d], filter size[%d x %d], stride %d, padding %d, number of map %d.\n", num_unit/ldu, ldu, m, n, stride, pad, num_map);
		}

	func = f;
}

#ifdef USE_MPI
void Pooling::init ( std::mt19937& m, MPI_Comm inner_world, MPI_Comm outer_world )
#else
void Pooling::init ( std::mt19937& m )
#endif
{
#ifdef USE_MPI
	this->inner_world = inner_world;
	this->outer_world = outer_world;

	MPI_Comm_rank(inner_world, &rank);
	MPI_Comm_size(inner_world, &nprocs);
#endif
	
	W = std::vector<std::vector<Mat>>();
}

void Pooling::finalize ()
{
	
}

std::vector<std::vector<Pooling::Mat>> Pooling::calc_gradient ( const std::vector<Mat>& U, const std::vector<Mat>& delta )
{
	return std::vector<std::vector<Mat>>();
}

std::vector<Pooling::Mat> Pooling::calc_delta ( const std::vector<Mat>& U, const std::vector<Mat>& delta )
{
	auto tot_beg = std::chrono::system_clock::now();
	auto beg = tot_beg;

	int my_size = num_unit, my_offset = 0;
#ifdef USE_MPI
	my_size = (rank+1)*num_unit/nprocs - rank*num_unit/nprocs;
	my_offset = rank*num_unit/nprocs;
#endif

	const int Y = prev_num_unit/prev_ldu, X = prev_ldu;
	std::vector<Mat> nx_delta(prev_num_map);
	auto end = std::chrono::system_clock::now();
	t_delta_init += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

	for( int i = 0; i < prev_num_map; ++i ){
		beg = std::chrono::system_clock::now();
		auto U_apply = (*prev_func)(U[i], false);
		auto U_diff = (*prev_func)(U[i], true);
		nx_delta[i] = Mat::zeros(U[i].m, U[i].n);

		const int gap = prev_ldu + 2*pad;
#pragma omp parallel for schedule(auto)
		for( int j = 0; j < my_size; ++j ){
			int x = (j + my_offset)%ldu, y = (j + my_offset)/ldu;

			for( int k = 0; k < U_apply.n; ++k ){
				int s_idx = -1;
				double val = -1E100;
				
				for( int s = 0; s < m; ++s )
					for( int t = 0; t < n; ++t ){
						int idx = stride*x + t + s*gap + stride*y*gap;
						int nx = idx%gap - pad, ny = idx/gap - pad;

						if( nx < 0 || nx >= X || ny < 0 || ny >= Y ) continue;

						if( s_idx == -1 || val < U_apply(ny*prev_ldu + nx, k) ){
							val = U_apply(ny*prev_ldu + nx, k);
							s_idx = ny*prev_ldu + nx;
						}
					}
				
				nx_delta[i](s_idx, k) += delta[i](j + my_offset, k) * U_diff(s_idx, k);
			}
		}
		end = std::chrono::system_clock::now();
		t_delta_repl += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;
		
		beg = std::chrono::system_clock::now();
#ifdef USE_MPI
		MPI_Allreduce(MPI_IN_PLACE, &nx_delta[i](0,0), U[i].m*U[i].n, MPI_DOUBLE_PRECISION, MPI_SUM, inner_world);
#endif
		end = std::chrono::system_clock::now();
		t_delta_comm += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;
	}
	end = std::chrono::system_clock::now();
	t_delta += std::chrono::duration_cast<std::chrono::nanoseconds>(end - tot_beg).count()/1e9;
	
	return nx_delta;
}

void Pooling::update_W ( const std::vector<std::vector<Mat>>& dW )
{
	
}

std::vector<Pooling::Mat> Pooling::apply ( const std::vector<Mat>& U, bool use_func )
{
	auto tot_beg = std::chrono::system_clock::now();
	auto beg = tot_beg;

	int my_size = num_unit, my_offset = 0;
#ifdef USE_MPI
	std::vector<int> size(nprocs), offset(nprocs);
	for( int i = 0; i < nprocs; ++i ){
		size[i] = ((i+1)*my_size/nprocs - i*my_size/nprocs)*U[0].n;
		offset[i] = i*my_size/nprocs*U[0].n;
	}

	my_size = size[rank]/U[0].n;
	my_offset = offset[rank]/U[0].n;
#endif
	
	const int Y = prev_num_unit/prev_ldu, X = prev_ldu;
	std::vector<Mat> new_S(num_map, Mat(num_unit, U[0].n));
	std::vector<Mat> ret(num_map);
	auto end = std::chrono::system_clock::now();
	t_apply_init += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

	Mat tmp(my_size, U[0].n);
	for( int i = 0; i < num_map; ++i ){
		beg = std::chrono::system_clock::now();
		ret[i] = Mat(num_unit, U[0].n);
		Mat U_ = (*prev_func)(U[i], false);

		const int gap = prev_ldu + 2*pad;
#pragma omp parallel for schedule(auto)
		for( int j = 0; j < my_size; ++j ){
			int x = (j + my_offset)%ldu, y = (j + my_offset)/ldu;

			for( int k = 0; k < U_.n; ++k ){
				int s_idx = -1;
				double val = -1E100;

				for( int s = 0; s < m; ++s )
					for( int t = 0; t < n; ++t ){
						int idx = stride*x + t + s*gap + stride*y*gap;
						int nx = idx%gap - pad, ny = idx/gap - pad;
						
						if( nx < 0 || nx >= X || ny < 0 || ny >= Y ) continue;

						if( s_idx == -1 || val < U_(ny*prev_ldu + nx, k) ){
							val = U_(ny*prev_ldu + nx, k);
							s_idx = ny*prev_ldu + nx;
						}
					}

				tmp(j, k) = val;
				new_S[i](j + my_offset, k) = s_idx;
			}
		}

		if( use_func )
			tmp = (*func)(tmp, false);
		end = std::chrono::system_clock::now();
		t_apply_repl += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;

		beg = std::chrono::system_clock::now();
#ifdef USE_MPI
		MPI_Allgatherv(&tmp(0,0), size[rank], MPI_DOUBLE_PRECISION,
					   &ret[i](0,0), &size[0], &offset[0], MPI_DOUBLE_PRECISION, inner_world);
		MPI_Allgatherv(MPI_IN_PLACE, size[rank], MPI_DOUBLE_PRECISION,
					   &new_S[i](0,0), &size[0], &offset[0], MPI_DOUBLE_PRECISION, inner_world);
#else
		ret[i] = tmp;
#endif
		end = std::chrono::system_clock::now();
		t_apply_comm += std::chrono::duration_cast<std::chrono::nanoseconds>(end - beg).count()/1e9;
	}

	S = new_S;
	end = std::chrono::system_clock::now();
	t_apply += std::chrono::duration_cast<std::chrono::nanoseconds>(end - tot_beg).count()/1e9;

	return ret;
}

std::vector<std::vector<Pooling::Vec>> Pooling::apply ( const std::vector<std::vector<Vec>>& u, bool use_func )
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

std::vector<Pooling::Mat> Pooling::unpooling ( const std::vector<Mat>& U )
{
	const int Y = prev_num_unit/prev_ldu, X = prev_ldu;
	std::vector<Mat> ret(num_map);

	int i,j,k,y,x;
#pragma omp parallel for default(none) \
	private(i,j,y,x) shared(ret, U)
	for( i = 0; i < num_map; ++i ){
		ret[i] = Mat(prev_num_unit, U[0].n);
		for( j = 0; j < U[0].n; ++j ){
			for( y = 0; y < Y; y += stride )
				for( x = 0; x < X; x += stride ){
					int idx = S[i](x/stride + (y/stride)*ldu, j);
					double val = U[i](x+prev_ldu*y,j);

					ret[i](idx,j) = U[i](x/stride + (y/stride)*ldu,j);
				}
		}
	}

	return ret;
}

std::vector<std::vector<Pooling::Vec>> Pooling::unpooling ( const std::vector<std::vector<Vec>>& u )
{
	std::vector<Mat> tmp(num_map);
	for( int i = 0; i < num_map; ++i )
		tmp[i] = Mat(u[0][0].size(), u.size());

	for( int i = 0; i < num_map; ++i )
		for( int j = 0; j < u[0][0].size(); ++j )
			for( int k = 0; k < u.size(); ++k )
				tmp[i](j,k) = u[k][i][j];
	
	auto U = unpooling(tmp);
	std::vector<std::vector<Vec>> ret(U[0].n);
	for( int i = 0; i < U[0].n; ++i ){
		ret[i] = std::vector<Vec>(U.size(), Vec(U[0].m));
		for( int j = 0; j < U.size(); ++j )
			for( int k = 0; k < U[0].m; ++k )
				ret[i][j][k] = U[j](k,i);
	}

	return ret;
}

void Pooling::set_W ( const std::string& filename )
{
	std::ifstream ifs(filename, std::ios::binary);
}

void Pooling::output_W ( const std::string& filename )
{
	std::ofstream ofs(filename, std::ios::binary);
}

#ifdef USE_MPI
void Pooling::param_mix ()
{

}
#endif

#endif
