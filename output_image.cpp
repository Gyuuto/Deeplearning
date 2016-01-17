// Standard I/O
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdio>
// Standard Library
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cmath>
// Template Class
#include <complex>
#include <string>
#include <vector>
#include <list>
#include <set>
#include <map>
#include <queue>
#include <stack>
// Container Control
#include <algorithm>

#include "Neuralnet.hpp"

#include <core/core.hpp>
#include <imgproc/imgproc.hpp>
#include <highgui/highgui.hpp>
#include <random>
#include <unistd.h>

using namespace std;

#define rep( i, n ) for( int i = 0; i < n; ++i )
#define irep( i, n ) for( int i = n-1; i >= 0; --i )
#define reep( i, s, n ) for ( int i = s; i < n; ++i )
#define ireep( i, n, s ) for ( int i = n-1; i >= s; --i )
#define foreach(itr, x) for( typeof(x.begin()) itr = x.begin(); itr != x.end(); ++itr)

#define mp make_pair
#define pb push_back
#define eb emplace_back
#define all( v ) v.begin(), v.end()
#define fs first
#define sc second
#define vc vector

// for visualizer.html
double SCALE = 1.0;
double OFFSET_X = 0.0;
double OFFSET_Y = 0.0;
#define LINE(x,y,a,b) cerr << "line(" << SCALE*(x) + OFFSET_X << ","	\
	<< SCALE*(y) + OFFSET_Y << ","										\
	<< SCALE*(a) + OFFSET_X << ","										\
	<< SCALE*(b) + OFFSET_Y << ")" << endl;
#define CIRCLE(x,y,r) cerr << "circle(" << SCALE*(x) + OFFSET_X << ","	\
	<< SCALE*(y) + OFFSET_Y << ","										\
	<< SCALE*(r) << ")" << endl;

typedef long long ll;
typedef complex<double> Point;

typedef pair<int, int> pii;
typedef pair<int, pii> ipii;
typedef vector<int> vi;
typedef vector<double> vd;
typedef vector< vector<int> > vii;
typedef vector< vector<double> > vdd;

typedef vector<int>::iterator vi_itr;

const int IINF = 1 << 28;
const double INF = 1e30;
const double EPS = 1e-10;
const double PI = acos(-1.0);

// Direction : L U R D
const int dx[] = { -1, 0, 1, 0};
const int dy[] = { 0, -1, 0, 1 };

int main()
{
	vector<int> num_unit = { 28*28, 400, 28*28 };
	Neuralnet nn(num_unit);

	const double alpha = 1.0;
	auto f = [&](double x) -> double { return 1.0 / (1.0 + exp(-alpha*x)); };
	auto d_f = [&](double x) -> double { return alpha*exp(-alpha*x) / pow(1.0 + exp(-alpha*x), 2.0); };
	auto t_f = [](double x) -> double { return tanh(x); };
	auto t_d_f = [](double x) -> double { return 1 - tanh(x)*tanh(x); };
	auto R_f = [](double x) -> double { return max(0.0, x); };
	auto R_d_f = [](double x) -> double { return (x <= 0.0 ? 0.0 : 1.0); };
	auto o_f = [](double x) -> double { return x; };
	auto o_d_f = [](double x) -> double { return 1.0; };

	for( int i = 0; i < num_unit.size()-2; ++i ){
		nn.set_function(i, R_f, R_d_f);
	}
  	nn.set_function(num_unit.size()-2, R_f, R_d_f);
	nn.set_function(num_unit.size()-1, o_f, o_d_f);

	// nn.set_ALPHA(3.0);
	// nn.set_K(10.0/num_unit[1]);
	nn.set_W( "autoenc_W.dat" );

	vector<int> lab;
	vector<vector<double>> x;
	const int N = 20000;
	ifstream image("train-images-idx3-ubyte", ios_base::in | ios_base::binary);
	ifstream label("train-labels-idx1-ubyte", ios_base::in | ios_base::binary);
	image.seekg(4*4, ios_base::beg);
	label.seekg(4*2, ios_base::beg);
	for( int i = 0; x.size() < N; ++i ){
		// int beg = 4*4 + 28*28*i;
		// image.seekg(beg, ios_base::beg);
		// label.seekg(4*2+i, ios_base::beg);

		unsigned char tmp_lab;
		label.read((char*)&tmp_lab, sizeof(unsigned char));
		lab.pb(tmp_lab);
		
		vector<double> tmp(28*28);
		for( int i = 0; i < 28*28; ++i ){
			unsigned char c;
			image.read((char*)&c, sizeof(unsigned char));
			tmp[i] = (c/255.0);
		}

		// if( tmp_lab > 2 ) continue;
		x.pb(tmp);
	}

	const int X = 16, Y = 8;
	cv::Mat imag(29*Y, 29*X, CV_8U);
	rep(i, 29*Y) rep(j, 29*X) imag.at<unsigned char>(i, j) = 0;
	// rep(i, Y*X){
	// 	double max_val = -1.0E100, min_val = 1.0E100;
	// 	rep(j, 28) rep(k, 28){
	// 		max_val = max(max_val, x[i][j*28+k]);
	// 		min_val = min(min_val, x[i][j*28+k]);
	// 	}
	// 	rep(j, 28) rep(k, 28)
	// 		imag.at<unsigned char>(i/X*29 + j, i%X*29 + k) = (x[i][j*28+k] - min_val)/(max_val - min_val)*255;
	// }
	// cv::imwrite("original.png", imag);

	// ifstream t_image("t10k-images-idx3-ubyte", ios_base::in | ios_base::binary);
	// imag = cv::Mat(29*2, 29*5, CV_8U);
	// int cnt = 0;
	// vi id = { 8, 124, 149, 151, 241, 247, 321, 340, 352, 358 };
	// for( int i = 0; i < X*Y; ++i ){
	// 	int beg = 4*4 + 28*28*i;
	// 	t_image.seekg(beg, ios_base::beg);

	// 	vector<double> tmp(28*28);
	// 	for( int j = 0; j < 28*28; ++j ){
	// 		unsigned char c;
	// 		t_image.read((char*)&c, sizeof(unsigned char));
	// 		tmp[j] = (c/255.0);
	// 	}

	// 	double max_val = -1.0E100, min_val = 1.0E100;
	// 	rep(j, 28) rep(k, 28){
	// 		max_val = max(max_val, tmp[j*28+k]);
	// 		min_val = min(min_val, tmp[j*28+k]);
	// 	}

	// 	rep(j, 28) rep(k, 28){
	// 		imag.at<unsigned char>(i/5*29+j, i%5*29+k) = (tmp[j*28+k]-min_val)/(max_val-min_val)*255;
	// 	}
	// }
	// cv::imwrite("test_imag.png", imag);

	Matrix<double> ave = Matrix<double>::zeros(28*28, 1);
	rep(i, N) ave = ave + Matrix<double>(x[i]);
	ave = 1.0/N * ave;
	
	double sigma = 0.0;
	rep(i, N){
		Matrix<double> y(x[i]);
		sigma += (Matrix<double>::transpose(y - ave)*(y - ave))[0][0];
	}
	sigma /= N;
	sigma = sqrt(sigma);

	rep(i, N){
		rep(j, x[i].size()){
			x[i][j] = (x[i][j] - ave[j][0])/sigma;
		}
	}

	const double NOISE_RATE = 0.0;
	mt19937 m(20160111);
	uniform_real_distribution<double> d_rand(0.0, 1.0);
	vector<vector<double>> y(x.size(), vector<double>(x[0].size(), 0.0));
	rep(i, y.size()){
		rep(j, y[i].size()){
			y[i][j] = x[i][j] - 1.0*(d_rand(m) - NOISE_RATE > 0.0 ? 0.0 : x[i][j]);
		}
	}
	
	// rep(i, Y*X){
	// 	double max_val = -1.0E100, min_val = 1.0E100;
	// 	rep(j, 28) rep(k, 28){
	// 		max_val = max(max_val, x[i][j*28+k]);
	// 		min_val = min(min_val, x[i][j*28+k]);
	// 	}
	// 	rep(j, 28) rep(k, 28)
	// 		imag.at<unsigned char>(i/X*29 + j, i%X*29 + k) = (y[i][j*28+k] - min_val)/(max_val - min_val)*255;
	// }
	// cv::imwrite("original_noised.png", imag);

	// rep(i, Y*X){
	// 	double max_val = -1.0E100, min_val = 1.0E100;
	// 	rep(j, 28) rep(k, 28){
	// 		max_val = max(max_val, x[i][j*28+k]);
	// 		min_val = min(min_val, x[i][j*28+k]);
	// 	}
	// 	rep(j, 28) rep(k, 28)
	// 		imag.at<unsigned char>(i/X*29 + j, i%X*29 + k) = (x[i][j*28+k] - min_val)/(max_val - min_val)*255;
	// }
	// cv::imwrite("original_normalized.png", imag);

	imag = cv::Mat((28*2+1)*Y, 29*X, CV_8U);
	rep(i, (28*2+1)*Y) rep(j, 29*X) imag.at<unsigned char>(i, j) = 0;
	const int OFF = 10000;
	rep(y_, Y){
		rep(x_, X){
			int idx = OFF+y_*X + x_;
			double max_num = -1.0E100, min_num = 1.0E100;
			rep(j, 28) rep(k, 28){
				max_num = max(max_num, x[idx][28*j+k]*sigma+ave[28*j+k][0]);
				min_num = min(min_num, x[idx][28*j+k]*sigma+ave[28*j+k][0]);
			}
			rep(j, 28) rep(k, 28){
				imag.at<unsigned char>(y_*(28*2+1)+j, x_*29+k) = (x[idx][j*28+k]*sigma+ave[28*j+k][0]-min_num)/(max_num-min_num)*255;
			}
		}
	}
	rep(y_, 8){
		rep(x_, 16){
			auto y = nn.apply(x[OFF+y_*16+x_]);
		
			double max_num = -1.0E100, min_num = 1.0E100;
			rep(j, 28) rep(k, 28){
				max_num = max(max_num, y[28*j+k]*sigma+ave[28*j+k][0]);
				min_num = min(min_num, y[28*j+k]*sigma+ave[28*j+k][0]);
			}
			rep(j, 28) rep(k, 28){
				imag.at<unsigned char>(y_*(28*2+1)+28+j, x_*29+k) = (y[j*28+k]*sigma+ave[28*j+k][0]-min_num)/(max_num-min_num)*255;
			}
		}
	}
	cv::imwrite("encode.png", imag);

	for( int n = 0; n < 2; ++n ){
		auto W = (n == 1 ? Matrix<double>::transpose(nn.get_W(n)) : nn.get_W(n));
		const int M_Y = sqrt(num_unit[1]), M_X = ceil((double)num_unit[1]/M_Y);
		imag = cv::Mat(M_Y*29, M_X*29, CV_8U);
		rep(i, 29*M_Y) rep(j, 29*M_X) imag.at<unsigned char>(i, j) = 0;
		rep(i, num_unit[1]){
			double max_num = -1.0E100, ave = 0.0;

			rep(j, 28) rep(k, 28){
				ave += W[i+n][(1-n)+28*j+k];
				max_num = max(max_num, abs(W[i+n][(1-n)+28*j+k]));
			}
			ave /= 28*28;
		
			rep(j, 28) rep(k, 28)
				W[i+n][(1-n)+28*j+k] = (pow((W[i+n][(1-n)+28*j+k] - ave)/max_num,1.0/1.0) + 1.0)/2.0;
			
			rep(j, 28) rep(k, 28){
				imag.at<unsigned char>(i/M_X*29+j, i%M_X*29+k) = W[i+n][(1-n)+28*j+k]*255;
			}

		}
		cv::imwrite("base_W"+to_string((ll)n)+".png", imag);
	}
}
