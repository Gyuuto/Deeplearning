#ifndef BATCHNORMALIZE_HPP
#define BATCHNORMALIZE_HPP

#include <fstream>
#include "Layer.hpp"

class BatchNormalize : public Layer
{
private:
	const double EPS = 1.0E-8;
	Mat mean, var;
public:
	BatchNormalize( int prev_num_map, int prev_num_unit,
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

	void set_W ( const std::string& filename );
	void output_W ( const std::string& filename );
	
#ifdef USE_MPI
	void param_mix ();
#endif
};

BatchNormalize::BatchNormalize( int prev_num_map, int prev_num_unit,
								const std::shared_ptr<Function>& f )
{
	this->prev_num_map = this->num_map = prev_num_map;
	this->prev_num_unit = this->num_unit = prev_num_unit;
	func = f;

	W = std::vector<std::vector<Mat>>(1, std::vector<Mat>(num_map, Mat(1, 2)));
}

#ifdef USE_MPI
void BatchNormalize::init( std::mt19937& m, MPI_Comm outer_world, MPI_Comm inner_world )
#else
void BatchNormalize::init( std::mt19937& m )
#endif
{
#ifdef USE_MPI
	this->inner_world = inner_world;
	this->outer_world = outer_world;
	MPI_Comm_size(inner_world, &this->nprocs);
	MPI_Comm_rank(inner_world, &this->rank);
#endif

	int seed = time(NULL);
#ifdef USE_MPI
	MPI_Bcast(&seed, 1, MPI_INTEGER, 0, inner_world);
#endif
	
	int offset, my_size;
#ifdef USE_MPI
	my_size = (rank+1)*num_unit/nprocs - rank*num_unit/nprocs;
	offset = rank*num_unit/nprocs;
#else
	offset = 0;
	my_size = num_unit;
#endif
	mean = Mat(num_map, my_size);
	var = Mat(num_map, my_size);

	std::uniform_real_distribution<double> d_rand(-1.0, 1.0);
	for( int i = 0; i < num_map; ++i )
		for( int j = 0; j < 2; ++j ) W[0][i](0,j) = d_rand(m);
}

void BatchNormalize::finalize ()
{

}

std::vector<std::vector<BatchNormalize::Mat>> BatchNormalize::calc_gradient ( const std::vector<Mat>& U, const std::vector<Mat>& delta )
{
	int my_offset, my_size;
#ifdef USE_MPI
	my_size = (rank+1)*prev_num_unit/nprocs - rank*prev_num_unit/nprocs;
	my_offset = rank*prev_num_unit/nprocs;
#else
	my_offset = 0;
	my_size = prev_num_unit;
#endif
	std::vector<std::vector<Mat>> nabla(1, std::vector<Mat>(num_map, Mat::zeros(1, 2)));

	for( int i = 0; i < num_map; ++i ){
		double tmp_nabla1 = 0.0, tmp_nabla2 = 0.0;;
#pragma omp parallel
		{
#pragma omp for schedule(auto) reduction(+:tmp_nabla1)
			for( int j = 0; j < my_size; ++j ){
				for( int k = 0; k < U[i].n; ++k ){
					tmp_nabla1 += delta[i](my_offset+j,k)*(U[i](my_offset+j,k) - mean(i,j))/std::sqrt(var(i,j) + EPS);
				}
			}

#pragma omp for schedule(auto) reduction(+:tmp_nabla2)
			for( int j = 0; j < my_size; ++j ){
				for( int k = 0; k < U[i].n; ++k ){
					tmp_nabla2 += delta[i](my_offset+j,k);
				}
			}
		}

#ifdef USE_MPI
		MPI_Allreduce(MPI_IN_PLACE, &tmp_nabla1, 1, MPI_DOUBLE_PRECISION, MPI_SUM, inner_world);
		MPI_Allreduce(MPI_IN_PLACE, &tmp_nabla2, 1, MPI_DOUBLE_PRECISION, MPI_SUM, inner_world);
#endif
		nabla[0][i](0,0) = tmp_nabla1; nabla[0][i](0,1) = tmp_nabla2;
	}

	return nabla;
}

std::vector<BatchNormalize::Mat> BatchNormalize::calc_delta ( const std::vector<Mat>& U, const std::vector<Mat>& delta )
{
	int my_offset, my_size;
#ifdef USE_MPI
	std::vector<int> size(nprocs), offset(nprocs);
	for( int i = 0; i < nprocs; ++i ){
		size[i] = ((i+1)*prev_num_unit/nprocs - i*prev_num_unit/nprocs)*U[0].n;
		offset[i] = i*prev_num_unit/nprocs*U[0].n;
	}
	my_size = size[rank]/U[0].n;
	my_offset = offset[rank]/U[0].n;
#else
	my_offset = 0;
	my_size = prev_num_unit;
#endif

	std::vector<Mat> nx_delta(prev_num_map, Mat(prev_num_unit, delta[0].n));
	std::vector<Mat> tmp_nx_delta(prev_num_map, Mat(my_size, delta[0].n));
	
	for( int i = 0; i < num_map; ++i ){
		auto U_appl = (*prev_func)(U[i], false);
		auto U_diff = (*prev_func)(U[i], true);
		
#pragma omp parallel for schedule(auto)
		for( int j = 0; j < my_size; ++j )
			for( int k = 0; k < U[i].n; ++k ){
				double tmp1 = 0.0, tmp2 = 0.0;
				for( int l = 0; l < U[i].n; ++l ){
					tmp1 += delta[i](my_offset+j,l);
					tmp2 += delta[i](my_offset+j,l)*(U_appl(my_offset+j,l) - mean(i,j));
				}
				tmp1 /= U[i].n; tmp2 /= U[i].n;

				tmp_nx_delta[i](j,k) = W[0][i](0,0)/sqrt(var(i,j) + EPS)*delta[i](my_offset+j,k)*U_diff(my_offset+j,k)
					- W[0][i](0,0)/sqrt(var(i,j) + EPS)*U_diff(my_offset+j,k)*tmp1
					- W[0][i](0,0)/pow(var(i,j) + EPS, 1.5)*U_diff(my_offset+j,k)*(U_appl(my_offset+j,k) - mean(i,j))*tmp2;
			}
	}

#ifdef USE_MPI
	for( int i = 0; i < num_map; ++i )
		MPI_Allgatherv(&tmp_nx_delta[i](0,0), size[rank], MPI_DOUBLE_PRECISION,
					   &nx_delta[i](0,0), &size[0], &offset[0], MPI_DOUBLE_PRECISION, inner_world);
#endif

	return nx_delta;
}

void BatchNormalize::update_W ( const std::vector<std::vector<Mat>>& dW )
{
	for( int i = 0; i < num_map; ++i )
		W[0][i] += dW[0][i];
}

std::vector<BatchNormalize::Mat> BatchNormalize::apply ( const std::vector<Mat>& U, bool use_func )
{
	int my_offset, my_size;
#ifdef USE_MPI
	std::vector<int> size(nprocs), offset(nprocs);
	for( int i = 0; i < nprocs; ++i ){
		size[i] = ((i+1)*num_unit/nprocs - i*num_unit/nprocs)*U[0].n;
		offset[i] = i*num_unit/nprocs*U[0].n;
	}
	my_size = size[rank]/U[0].n;
	my_offset = offset[rank]/U[0].n;
#else
	my_offset = 0;
	my_size = num_unit;
#endif
	mean = var = Mat::zeros(num_map, my_size);

	std::vector<Mat> ret(num_map, Mat(num_unit, U[0].n));
	std::vector<Mat> tmp_ret(num_map, Mat(my_size, U[0].n));
#pragma omp parallel for schedule(auto)
	for( int i = 0; i < num_map; ++i ){
		for( int j = 0; j < U[i].n; ++j ){
			for( int k = 0; k < my_size; ++k ) mean(i,k) += U[i](my_offset+k, j);
		}
	}
	mean /= U[0].n;
	
#pragma omp parallel for schedule(auto)
	for( int i = 0; i < num_map; ++i ){
		for( int j = 0; j < U[i].n; ++j ){
			for( int k = 0; k < my_size; ++k ){
				double v = (U[i](my_offset+k, j) - mean(i, k));
				var(i,k) += v*v;
			}
		}
	}
	var /= U[0].n;
	
#pragma omp parallel for schedule(auto)
	for( int i = 0; i < num_map; ++i ){
		for( int j = 0; j < U[0].n; ++j ){
			for( int k = 0; k < my_size; ++k ){
				tmp_ret[i](k, j) = W[0][i](0,0)*(U[i](my_offset+k,j) - mean(i,k))/std::sqrt(var(i,k)+EPS) + W[0][i](0,1);
			}
		}
	}

#ifdef USE_MPI
	std::vector<MPI_Request> req(num_map);
#endif
	if( use_func ){
		for( int i = 0; i < num_map; ++i ){
			tmp_ret[i] = (*func)(tmp_ret[i], false);
#ifdef USE_MPI
			MPI_Iallgatherv(&tmp_ret[i](0,0), size[rank], MPI_DOUBLE_PRECISION,
							&ret[i](0,0), &size[0], &offset[0], MPI_DOUBLE_PRECISION, inner_world, &req[i]);
#endif
		}
	}
	else{
		for( int i = 0; i < num_map; ++i ){
#ifdef USE_MPI
			MPI_Iallgatherv(&tmp_ret[i](0,0), size[rank], MPI_DOUBLE_PRECISION,
							&ret[i](0,0), &size[0], &offset[0], MPI_DOUBLE_PRECISION, inner_world, &req[i]);
#endif
		}
	}

#ifdef USE_MPI
	for( int i = 0; i < num_map; ++i ){
		MPI_Status stat;
		MPI_Wait(&req[i], &stat);
	}
#endif
	
	return ret;
}

std::vector<std::vector<BatchNormalize::Vec>> BatchNormalize::apply ( const std::vector<std::vector<Vec>>& u, bool use_func )
{
	std::vector<Mat> tmp(prev_num_map);
	for( int i = 0; i < prev_num_map; ++i )
		tmp[i] = Mat(u[i][0].size(), u.size());

#pragma omp parallel
	{
		for( int i = 0; i < prev_num_map; ++i )
#pragma omp for schedule(auto) nowait
			for( int j = 0; j < u[i][0].size(); ++j )
				for( int k = 0; k < u.size(); ++k )
					tmp[i](j,k) = u[k][i][j];
	}
	
	auto U = apply(tmp, use_func);
	std::vector<std::vector<Vec>> ret(U[0].n, std::vector<Vec>(U.size(), Vec(U[0].m)));
#pragma omp parallel for schedule(auto)
	for( int i = 0; i < U[0].n; ++i ){
		for( int j = 0; j < U.size(); ++j )
			for( int k = 0; k < U[0].m; ++k )
				ret[i][j][k] = U[j](k,i);
	}

	return ret;
}

void BatchNormalize::set_W ( const std::string& filename )
{

}

void BatchNormalize::output_W ( const std::string& filename )
{

}
	
#ifdef USE_MPI
void BatchNormalize::param_mix ()
{

}
#endif

#endif
