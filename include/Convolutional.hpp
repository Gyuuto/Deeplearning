#ifndef CONVOLUTIONAL_HPP
#define CONVOLUTIONAL_HPP

#include "Layer.hpp"

class Convolutional : public Layer
{
private:
	int prev_ldu, ldu;
	int m, n, stride;

	Vec r, v;
	double beta_, gamma_;
public:
	Vec bias, d_bias;
	Convolutional( int prev_num_map, int prev_num_unit, int prev_ldu,
				   int num_map, int num_unit, int ldu,
				   int m, int n, int stride, 
				   const std::shared_ptr<Function>& f );

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

	this->m = m; this->n = n; this->stride = stride;

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
void Convolutional::init ( std::mt19937& m, MPI_Comm inner_world, MPI_Comm outer_world )
#else
void Convolutional::init ( std::mt19937& m )
#endif
{
#ifdef USE_MPI
	this->inner_world = inner_world;
	this->outer_world = outer_world;

	MPI_Comm_rank(inner_world, &rank);
	MPI_Comm_size(inner_world, &nprocs);
#endif

	const double r = sqrt(6.0/(num_unit + prev_num_unit));
	std::normal_distribution<double> d_rand(0.0, 1.0E-1);

	bias = Vec(num_map, 0.0); d_bias = Vec(num_map, 0.0);
	this->r = Vec(num_map, 0.0); v = Vec(num_map, 0.0);
	for( int i = 0; i < num_map; ++i ){
		for( int j = 0; j < prev_num_map; ++j ){
			for( int k = 0; k < W[i][j].m; ++k )
				for( int l = 0; l < W[i][j].m; ++l )
					W[i][j](k,l) = d_rand(m);
		}				
	}
}

void Convolutional::finalize ()
{
}

std::vector<std::vector<Convolutional::Mat>> Convolutional::calc_gradient ( const std::vector<Mat>& U, const std::vector<Mat>& delta )
{
	int offset = 0, my_size = delta[0].m;
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

	const int Y = prev_num_unit/prev_ldu, X = prev_ldu;
	const int Y_ = num_unit/ldu, X_ = ldu;
	int i, j, k, l, s, t, y, x;
	Mat nabla_mat(m*n*num_map, prev_num_map);
	for( i = 0; i < delta[0].n; ++i ){
		Mat delta_mat(m*n*num_map, my_size);
#pragma omp parallel for default(none)					\
	private(j,k,l) shared(i, my_size,offset, delta,delta_mat)
		for( j = 0; j < num_map; ++j )
			for( k = 0; k < m*n; ++k ){
				int s = k%m - (m/2), t = k/m - (n/2);
				for( l = 0; l < my_size; ++l ){
					int nx = (l + offset)%ldu - s, ny = (l + offset)/ldu - t;
					if( nx < 0 || ny < 0 || X_ <= nx || Y_ <= ny ){
						delta_mat(j*m*n + k, l) = 0.0;
						continue;
					}
					delta_mat(j*m*n + k, l) = delta[j](ny*ldu + nx, i);
				}
			}

		Mat U_mat(my_size, prev_num_map);
		for( j = 0; j < prev_num_map; ++j )
#pragma omp parallel for default(none)					\
	private(k) shared(i,j, my_size,offset, U_,U_mat)
			for( k = 0; k < my_size; ++k )
				U_mat(k, j) = U_[j](offset+k, i);

		nabla_mat += delta_mat*U_mat;
	}

#pragma omp parallel for default(none)			\
	private(i,j,k,l) shared(my_size, delta,nabla,nabla_mat)
	for( i = 0; i < num_map; ++i ){
		for( j = 0; j < prev_num_map; ++j ){
			for( k = 0; k < n; ++k )
				for( l = 0; l < m; ++l )
					nabla[i][j](k, l) = nabla_mat(i*m*n + l*n + k, j);
		}
	
		double sum = 0.0;
		for( j = 0; j < delta[i].n; ++j )
			for( k = 0; k < delta[i].m; ++k )
				sum += delta[i](k,j);
		d_bias[i] = sum / delta[i].n;
	}
#ifdef USE_MPI
	for( i = 0; i < num_map; ++i )
		for( j = 0; j < prev_num_map; ++j )
			MPI_Allreduce(MPI_IN_PLACE, &nabla[i][j](0,0), m*n, MPI_DOUBLE_PRECISION, MPI_SUM, inner_world);
#endif
	
	return nabla;				
}

std::vector<Convolutional::Mat> Convolutional::calc_delta ( const std::vector<Mat>& U, const std::vector<Mat>& delta )
{
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

	int i, j, k, l, x, y, s, t;
	for( int i = 0; i < prev_num_map; ++i ) tmp[i] = Mat(prev_num_unit, U[0].n);

	Mat kernel(m*n*num_map, prev_num_map);
#pragma omp parallel for default(none) \
	private(i,j,k,l) shared(kernel)
	for( i = 0; i < num_map; ++i )
		for( j = 0; j < prev_num_map; ++j )
			for( k = 0; k < m; ++ k )
				for( l = 0; l < n; ++l )
					kernel(i*(m*n) + k*m + l, j) = W[i][j](l, k);

	for( i = 0; i < delta[0].n; ++i ){
		Mat input_image(my_size, m*n*num_map);

#pragma omp parallel for default(none) \
	private(j,k,s,t) shared(i, input_image, my_size, my_offset, delta, X_, Y_)
		for( j = 0; j < my_size; ++j ){
			int x = (j + my_offset)%prev_ldu, y = (j + my_offset)/prev_ldu;

			for( s = -m/2; s < (m+1)/2; s += stride )
				for( t = -n/2; t < (n+1)/2; t += stride ){
					int nx = x - s, ny = y - t;
					if( nx < 0 || nx >= X_ || ny < 0 || ny >= Y_ ){
						for( k = 0; k < num_map; ++k )
							input_image(j, m*n*k + (t+(n/2))*n + s+(m/2)) = 0.0;
						continue;
					}
					for( k = 0; k < num_map; ++k )
						input_image(j, m*n*k + (t+(n/2))*n + s+(m/2)) = delta[k](ny*ldu + nx, i);
				}
		}

#ifdef USE_MPI
		Mat tmp_output_image = input_image * kernel;
		Mat output_image(prev_num_unit, prev_num_map);
		MPI_Allgatherv(&tmp_output_image(0,0), size[rank], MPI_DOUBLE_PRECISION,
					   &output_image(0,0), &size[0], &offset[0], MPI_DOUBLE_PRECISION, inner_world);
#else
		Mat output_image = input_image * kernel;
#endif

		for( j = 0; j < prev_num_map; ++j )
#pragma omp parallel for default(none) \
	private(k) shared(i,j, output_image, tmp)
			for( k = 0; k < prev_num_unit; ++k )
				tmp[j](k, i) = output_image(k, j);
	}

	for( i = 0; i < prev_num_map; ++i )
		nx_delta[i] = Mat::hadamard(tmp[i], (*prev_func)(U[i], true));

	return nx_delta;
}

void Convolutional::update_W ( const std::vector<std::vector<Mat>>& dW )
{
	const double a_beta = 0.9, a_gamma = 0.999, a_eps = 1.0E-8;
	for( int i = 0; i < num_map; ++i ){
		for( int j = 0; j < prev_num_map; ++j )
			W[i][j] = W[i][j] + dW[i][j];

		beta_ *= a_beta; gamma_ *= a_gamma;
		v[i] = a_beta*v[i] + (1.0 - a_beta)*d_bias[i];
		r[i] = a_gamma*r[i] + (1.0 - a_gamma)*d_bias[i]*d_bias[i];
		bias[i] -= 0.001*v[i]/(1.0 - beta_)/(sqrt(r[i]/(1.0 - gamma_)+a_eps));
	}
}

std::vector<Convolutional::Mat> Convolutional::apply ( const std::vector<Mat>& U, bool use_func )
{
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

	int i, j, k, l, y, x, s, t;
	for( int i = 0; i < num_map; ++i ) ret[i] = Mat(num_unit, U[0].n);

	Mat kernel(m*n*prev_num_map, num_map);
#pragma omp parallel for default(none) \
	private(i,j,k,l) shared(kernel)
	for( i = 0; i < num_map; ++i )
		for( j = 0; j < prev_num_map; ++j )
			for( k = 0; k < m; ++ k )
				for( l = 0; l < n; ++l )
					kernel(j*(m*n) + k*n + l, i) = W[i][j](l, k);
			
	for( i = 0; i < U[0].n; ++i ){
		Mat input_image(my_size, m*n*prev_num_map);

#pragma omp parallel for default(none) \
	private(j,k,s,t) shared(i, input_image, my_size, my_offset, U, X, Y)
		for( j = 0; j < my_size; ++j ){
			int x = (j + my_offset)%prev_ldu, y = (j + my_offset)/prev_ldu;
			for( s = -m/2; s < (m+1)/2; ++s )
				for( t = -n/2; t < (n+1)/2; ++t ){
					int nx = x + s, ny = y + t;
					if( nx < 0 || nx >= X || ny < 0 || ny >= Y ){
						for( k = 0; k < prev_num_map; ++k )
							input_image(j, m*n*k + (t+(n/2))*n + s+(m/2)) = 0.0;
						continue;
					}
					for( k = 0; k < prev_num_map; ++k )
						input_image(j, m*n*k + (t+(n/2))*m + s+(m/2)) = U[k](ny*prev_ldu + nx, i);
				}
		}

#ifdef USE_MPI
		Mat tmp_output_image = input_image * kernel;
		Mat output_image(num_unit, num_map);
		MPI_Allgatherv(&tmp_output_image(0,0), size[rank], MPI_DOUBLE_PRECISION,
					   &output_image(0,0), &size[0], &offset[0], MPI_DOUBLE_PRECISION, inner_world);
#else
		Mat output_image = input_image * kernel;
#endif

		for( j = 0; j < num_map; ++j )
#pragma omp parallel for default(none) \
	private(k) shared(i,j, output_image, ret)
			for( k = 0; k < num_unit; ++k )
				ret[j](k, i) = output_image(k, j);
	}
	
	for( i = 0; i < num_map; ++i )
#pragma omp parallel for default(none)	\
	private(j,k) shared(i, ret)
		for( j = 0; j < ret[i].m; ++j )
			for( k = 0; k < ret[i].n; ++k )
				ret[i](j,k) += bias[i];


	if( use_func )
		for( i = 0; i < num_map; ++i )
			ret[i] = (*func)(ret[i], false);

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
