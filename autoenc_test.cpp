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
	vector<int> num_unit = { 28*28, 400, 10 };
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
		nn.set_function(i, o_f, o_d_f);
	}
  	nn.set_function(num_unit.size()-2, o_f, o_d_f);
	nn.set_function(num_unit.size()-1, o_f, o_d_f);
	
	vector<int> lab;
	vector<vector<double>> x;
	const int N = 10000;
	ifstream image("train-images-idx3-ubyte", ios_base::in | ios_base::binary);
	ifstream label("train-labels-idx1-ubyte", ios_base::in | ios_base::binary);
	for( int i = 0; x.size() < N; ++i ){
		int beg = 4*4 + 28*28*i;
		image.seekg(beg, ios_base::beg);
		label.seekg(4*2+i, ios_base::beg);
		
		unsigned char tmp_lab;
		label.read((char*)&tmp_lab, sizeof(unsigned char));
		lab.pb(tmp_lab);
		
		vector<double> tmp(28*28);
		for( int i = 0; i < 28*28; ++i ){
			unsigned char c;
			image.read((char*)&c, sizeof(unsigned char));
			tmp[i] = (c/255.0);
		}
		
		x.pb(tmp);
	}
	
	Matrix<double> ave(28*28, 1);
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
			y[i][j] = x[i][j] + 0.1*(d_rand(m) - NOISE_RATE > 0.0 ? 0.0 : 2*d_rand(m)-1.0);
		}
	}
	
	nn.set_EPS(1.0E-1);
	nn.set_LAMBDA(1.0E-4);
	nn.set_MU(0.0);
	// nn.set_BETA(1.0);
	// nn.set_RHO(0.1);
	// nn.set_K(10.0/num_unit[1]);
	// nn.set_ALPHA(3.0);
	nn.set_BATCHSIZE(50);
	
	// nn.set_W( "autoenc_W_recog.dat" );
	
	vector<vector<double>> d(x.size(), vector<double>(10, 0.0));
	rep(i, d.size()) d[i][lab[i]] = 1.0;		
	
	nn.learning(y, d, N/50*100);
	
	// rep(i, x.size()){
	// 	auto y = nn.apply(x[i]);
	// 	double err = 0.0;
	// 	rep(j, y.size()) err += (y[j] - x[i][j])*(y[j] - x[i][j]);
	// 	printf("%d %.6E %.6E %.6E\n", i, err, x[i][300], y[300]);
	// }

	ifstream t_image("t10k-images-idx3-ubyte", ios_base::in | ios_base::binary);
	ifstream t_label("t10k-labels-idx1-ubyte", ios_base::in | ios_base::binary);
	int ans_num = 0;
	for( int i = 0; i < 10000; ++i ){
		int beg = 4*4 + 28*28*i;
		t_image.seekg(beg, ios_base::beg);
		t_label.seekg(4*2+i, ios_base::beg);

		unsigned char tmp_lab;
		t_label.read((char*)&tmp_lab, sizeof(unsigned char));
		
		vector<double> tmp(28*28);
		for( int j = 0; j < 28*28; ++j ){
			unsigned char c;
			t_image.read((char*)&c, sizeof(unsigned char));
			tmp[j] = ((c/255.0)-ave[j][0])/sigma;
		}

		auto y = nn.apply(tmp);
		vector<double> prob(10, 0.0);
		int idx;
		double sum = 0.0, max_num = 0.0;
		rep(j, 10) sum += exp(y[j]);
		rep(j, 10){
			prob[j] = exp(y[j]) / sum;
			if( max_num < prob[j] ){
				max_num = prob[j];
				idx = j;
			}
		}
		if( idx == tmp_lab ) ++ans_num;
		// printf("%d : label = %d, select = %d\n\t", i, tmp_lab, idx);
		// rep(j, 10) printf("(%d, %.4E) ", j, prob[j]);
		// puts("");
	}
	printf("Answer rate : %.6E\n", (double)ans_num/10000.0);
}
