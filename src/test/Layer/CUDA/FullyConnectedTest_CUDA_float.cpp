#include <algorithm>
#include <vector>

#include <gtest/gtest.h>
#include <Matrix.hpp>
#include <cudaMatrix.hpp>
#include <Layer/FullyConnected.hpp>

namespace {
    typedef float Real;

    class FullyConnectedTest : public ::testing::Test {
    protected:
        std::shared_ptr<Function<Real>> f;
        std::shared_ptr<Layer<cudaMatrix, Real>> layer;
        cudaMatrix<Real> x, delta;
        
        virtual void SetUp () {
            f = std::shared_ptr<Function<Real>>(new ReLU<Real>());
            layer = std::shared_ptr<Layer<cudaMatrix, Real>>(new FullyConnected<cudaMatrix, Real>(1, 2, 1, 3, f, false));
            
            std::vector<cudaMatrix<Real>> W;
            Matrix<Real> tmp_W(3, 2);
            tmp_W(0,0) = -1; tmp_W(0,1) = 1;
            tmp_W(1,0) = 2; tmp_W(1,1) = -3;
            tmp_W(2,0) = 0; tmp_W(2,1) = -1;
            W.emplace_back(tmp_W);

            this->layer->set_W(W);

            Matrix<Real> tmp_x(2, 2);
            tmp_x(0, 0) = 1; tmp_x(0, 1) = 1;
            tmp_x(1, 0) = 2; tmp_x(1, 1) = 3;

            x = tmp_x;

            Matrix<Real> tmp_delta(3, 2);
            tmp_delta(0, 0) = -0.1; tmp_delta(0, 1) = 0.4;
            tmp_delta(1, 0) = -0.2; tmp_delta(1, 1) = 0.2;
            tmp_delta(2, 0) = 0.1; tmp_delta(2, 1) = -0.1;
            delta = tmp_delta;
        }

        virtual void TearDown () {
            
        }
    };

    TEST_F(FullyConnectedTest, GPU_float_apply_test_without_function) {
        auto y = layer->apply(x, false);

        auto tmp_y = y.get_matrix();
        EXPECT_LT(fabs(tmp_y(0,0) - 1.0), 1.0E-4);
        EXPECT_LT(fabs(tmp_y(0,1) - 2.0), 1.0E-4);
        EXPECT_LT(fabs(tmp_y(1,0) - (-4.0)), 1.0E-4);
        EXPECT_LT(fabs(tmp_y(1,1) - (-7.0)), 1.0E-4);
        EXPECT_LT(fabs(tmp_y(2,0) - (-2.0)), 1.0E-4);
        EXPECT_LT(fabs(tmp_y(2,1) - (-3.0)), 1.0E-4);
    }

    TEST_F(FullyConnectedTest, GPU_float_apply_test_with_function) {
        auto y = layer->apply(x, true);

        auto tmp_y = y.get_matrix();
        EXPECT_LT(fabs(tmp_y(0,0) - 1.0), 1.0E-4);
        EXPECT_LT(fabs(tmp_y(0,1) - 2.0), 1.0E-4);
        EXPECT_LT(fabs(tmp_y(1,0) - 0.0), 1.0E-4);
        EXPECT_LT(fabs(tmp_y(1,1) - 0.0), 1.0E-4);
        EXPECT_LT(fabs(tmp_y(2,0) - 0.0), 1.0E-4);
        EXPECT_LT(fabs(tmp_y(2,1) - 0.0), 1.0E-4);
    }

    TEST_F(FullyConnectedTest, GPU_float_calc_delta_test) {
        cudaMatrix<Real> nx_delta(2, 2);

        layer->calc_delta(x, x, delta, nx_delta);
        
        auto tmp_nx_delta = nx_delta.get_matrix();
        EXPECT_LT(fabs(tmp_nx_delta(0,0) - (-0.3)), 1.0E-4);
        EXPECT_LT(fabs(tmp_nx_delta(0,1) - 0.0), 1.0E-4);
        EXPECT_LT(fabs(tmp_nx_delta(1,0) - 0.8), 1.0E-4);
        EXPECT_LT(fabs(tmp_nx_delta(1,1) - (-0.3)), 1.0E-4);
    }

    TEST_F(FullyConnectedTest, GPU_float_calc_gradient_test) {
        std::vector<cudaMatrix<Real>> nabla_W, nabla_b;
        nabla_W.emplace_back(3, 2);
        
        layer->calc_gradient(x, x, delta, nabla_W, nabla_b);

        auto tmp_nabla_W = nabla_W[0].get_matrix();
        EXPECT_LT(fabs(tmp_nabla_W(0,0) - 0.3), 1.0E-4);
        EXPECT_LT(fabs(tmp_nabla_W(0,1) - 1.0), 1.0E-4);
        EXPECT_LT(fabs(tmp_nabla_W(1,0) - 0.0), 1.0E-4);
        EXPECT_LT(fabs(tmp_nabla_W(1,1) - 0.2), 1.0E-4);
        EXPECT_LT(fabs(tmp_nabla_W(2,0) - 0.0), 1.0E-4);
        EXPECT_LT(fabs(tmp_nabla_W(2,1) - (-0.1)), 1.0E-4);
    }

    class FullyConnectedTest_bias : public ::testing::Test {
    protected:
        std::shared_ptr<Function<Real>> f;
        std::shared_ptr<Layer<cudaMatrix, Real>> layer;
        cudaMatrix<Real> x, delta;
        
        virtual void SetUp () {
            f = std::shared_ptr<Function<Real>>(new ReLU<Real>());
            layer = std::shared_ptr<Layer<cudaMatrix, Real>>(new FullyConnected<cudaMatrix, Real>(1, 2, 1, 3, f, true));
            
            std::vector<cudaMatrix<Real>> W;
            Matrix<Real> tmp_W(3, 2);
            tmp_W(0,0) = -1; tmp_W(0,1) = 1;
            tmp_W(1,0) = 2; tmp_W(1,1) = -3;
            tmp_W(2,0) = 0; tmp_W(2,1) = -1;
            W.emplace_back(tmp_W);
            
            std::vector<cudaMatrix<Real>> b;
            Matrix<Real> tmp_b(3, 1);
            tmp_b(0,0) = 1;
            tmp_b(1,0) = -1;
            tmp_b(2,0) = 1;
            b.emplace_back(tmp_b);

            this->layer->set_W(W);
            this->layer->set_b(b);

            Matrix<Real> tmp_x(2, 2);
            tmp_x(0, 0) = 1; tmp_x(0, 1) = 1;
            tmp_x(1, 0) = 2; tmp_x(1, 1) = 3;
            x = tmp_x;

            Matrix<Real> tmp_delta(3, 2);
            tmp_delta(0, 0) = -0.1; tmp_delta(0, 1) = 0.4;
            tmp_delta(1, 0) = -0.2; tmp_delta(1, 1) = 0.2;
            tmp_delta(2, 0) = 0.1; tmp_delta(2, 1) = -0.1;
            delta = tmp_delta;
        }

        virtual void TearDown () {
            
        }
    };

    TEST_F(FullyConnectedTest_bias, GPU_float_apply_test_without_function) {
        auto y = layer->apply(x, false);

        auto tmp_y = y.get_matrix();
        EXPECT_LT(fabs(tmp_y(0,0) - 2.0), 1.0E-4);
        EXPECT_LT(fabs(tmp_y(0,1) - 3.0), 1.0E-4);
        EXPECT_LT(fabs(tmp_y(1,0) - (-5.0)), 1.0E-4);
        EXPECT_LT(fabs(tmp_y(1,1) - (-8.0)), 1.0E-4);
        EXPECT_LT(fabs(tmp_y(2,0) - (-1.0)), 1.0E-4);
        EXPECT_LT(fabs(tmp_y(2,1) - (-2.0)), 1.0E-4);
    }

    TEST_F(FullyConnectedTest_bias, GPU_float_apply_test_with_function) {
        auto y = layer->apply(x, true);

        auto tmp_y = y.get_matrix();
        EXPECT_LT(fabs(tmp_y(0,0) - 2.0), 1.0E-4);
        EXPECT_LT(fabs(tmp_y(0,1) - 3.0), 1.0E-4);
        EXPECT_LT(fabs(tmp_y(1,0) - 0.0), 1.0E-4);
        EXPECT_LT(fabs(tmp_y(1,1) - 0.0), 1.0E-4);
        EXPECT_LT(fabs(tmp_y(2,0) - 0.0), 1.0E-4);
        EXPECT_LT(fabs(tmp_y(2,1) - 0.0), 1.0E-4);
    }

    TEST_F(FullyConnectedTest_bias, GPU_float_calc_delta_test) {
        cudaMatrix<Real> nx_delta(2, 2);

        layer->calc_delta(x, x, delta, nx_delta);

        auto tmp_nx_delta = nx_delta.get_matrix();
        EXPECT_LT(fabs(tmp_nx_delta(0,0) - (-0.3)), 1.0E-4);
        EXPECT_LT(fabs(tmp_nx_delta(0,1) - 0.0), 1.0E-4);
        EXPECT_LT(fabs(tmp_nx_delta(1,0) - 0.8), 1.0E-4);
        EXPECT_LT(fabs(tmp_nx_delta(1,1) - (-0.3)), 1.0E-4);
    }

    TEST_F(FullyConnectedTest_bias, GPU_float_calc_gradient_test) {
        std::vector<cudaMatrix<Real>> nabla_W, nabla_b;
        nabla_W.emplace_back(3, 2);
        nabla_b.emplace_back(3, 1);
        
        layer->calc_gradient(x, x, delta, nabla_W, nabla_b);

        auto tmp_nabla_W = nabla_W[0].get_matrix();
        EXPECT_LT(fabs(tmp_nabla_W(0,0) - 0.3), 1.0E-4);
        EXPECT_LT(fabs(tmp_nabla_W(0,1) - 1.0), 1.0E-4);
        EXPECT_LT(fabs(tmp_nabla_W(1,0) - 0.0), 1.0E-4);
        EXPECT_LT(fabs(tmp_nabla_W(1,1) - 0.2), 1.0E-4);
        EXPECT_LT(fabs(tmp_nabla_W(2,0) - 0.0), 1.0E-4);
        EXPECT_LT(fabs(tmp_nabla_W(2,1) - (-0.1)), 1.0E-4);

        auto tmp_nabla_b = nabla_b[0].get_matrix();
        EXPECT_LT(fabs(tmp_nabla_b(0,0) - 0.3), 1.0E-4);
        EXPECT_LT(fabs(tmp_nabla_b(1,0) - 0.0), 1.0E-4);
        EXPECT_LT(fabs(tmp_nabla_b(2,0) - 0.0), 1.0E-4);
    }
    
    class FullyConnectedTest_w_map : public ::testing::Test {
    protected:
        std::shared_ptr<Function<Real>> f;
        std::shared_ptr<Layer<cudaMatrix, Real>> layer;
        cudaMatrix<Real> x, delta;
        
        virtual void SetUp () {
            f = std::shared_ptr<Function<Real>>(new ReLU<Real>());
            layer = std::shared_ptr<Layer<cudaMatrix, Real>>(new FullyConnected<cudaMatrix, Real>(1, 2, 2, 3, f, false));
            
            std::vector<cudaMatrix<Real>> W;
            Matrix<Real> tmp_W(2*3, 2);
            tmp_W(0,0) = -1; tmp_W(0,1) = 1;
            tmp_W(1,0) = 2; tmp_W(1,1) = -3;
            tmp_W(2,0) = 0; tmp_W(2,1) = -1;

            tmp_W(3,0) = 1; tmp_W(3,1) = 0;
            tmp_W(4,0) = 0; tmp_W(4,1) = 1;
            tmp_W(5,0) = -2; tmp_W(5,1) = 1;
            W.emplace_back(tmp_W);

            this->layer->set_W(W);

            Matrix<Real> tmp_x(2, 2);
            tmp_x(0, 0) = 1; tmp_x(0, 1) = 1;
            tmp_x(1, 0) = 2; tmp_x(1, 1) = 3;
            x = tmp_x;

            Matrix<Real> tmp_delta(2*3, 2);
            tmp_delta(0, 0) = -0.1; tmp_delta(0, 1) = 0.4;
            tmp_delta(1, 0) = -0.2; tmp_delta(1, 1) = 0.2;
            tmp_delta(2, 0) = 0.1; tmp_delta(2, 1) = -0.1;

            tmp_delta(3, 0) = 0.0; tmp_delta(3, 1) = -0.2;
            tmp_delta(4, 0) = 0.1; tmp_delta(4, 1) = 0.6;
            tmp_delta(5, 0) = 0.2; tmp_delta(5, 1) = -0.3;
            delta = tmp_delta;
        }

        virtual void TearDown () {
            
        }
    };

    TEST_F(FullyConnectedTest_w_map, GPU_float_apply_test_without_function) {
        auto y = layer->apply(x, false);

        auto tmp_y = y.get_matrix();
        EXPECT_LT(fabs(tmp_y(0,0) - 1.0), 1.0E-4);
        EXPECT_LT(fabs(tmp_y(0,1) - 2.0), 1.0E-4);
        EXPECT_LT(fabs(tmp_y(1,0) - (-4.0)), 1.0E-4);
        EXPECT_LT(fabs(tmp_y(1,1) - (-7.0)), 1.0E-4);
        EXPECT_LT(fabs(tmp_y(2,0) - (-2.0)), 1.0E-4);
        EXPECT_LT(fabs(tmp_y(2,1) - (-3.0)), 1.0E-4);

        EXPECT_LT(fabs(tmp_y(3,0) - 1.0), 1.0E-4);
        EXPECT_LT(fabs(tmp_y(3,1) - 1.0), 1.0E-4);
        EXPECT_LT(fabs(tmp_y(4,0) - 2.0), 1.0E-4);
        EXPECT_LT(fabs(tmp_y(4,1) - 3.0), 1.0E-4);
        EXPECT_LT(fabs(tmp_y(5,0) - 0.0), 1.0E-4);
        EXPECT_LT(fabs(tmp_y(5,1) - 1.0), 1.0E-4);
    }

    TEST_F(FullyConnectedTest_w_map, GPU_float_apply_test_with_function) {
        auto y = layer->apply(x, true);

        auto tmp_y = y.get_matrix();
        EXPECT_LT(fabs(tmp_y(0,0) - 1.0), 1.0E-4);
        EXPECT_LT(fabs(tmp_y(0,1) - 2.0), 1.0E-4);
        EXPECT_LT(fabs(tmp_y(1,0) - 0.0), 1.0E-4);
        EXPECT_LT(fabs(tmp_y(1,1) - 0.0), 1.0E-4);
        EXPECT_LT(fabs(tmp_y(2,0) - 0.0), 1.0E-4);
        EXPECT_LT(fabs(tmp_y(2,1) - 0.0), 1.0E-4);

        EXPECT_LT(fabs(tmp_y(3,0) - 1.0), 1.0E-4);
        EXPECT_LT(fabs(tmp_y(3,1) - 1.0), 1.0E-4);
        EXPECT_LT(fabs(tmp_y(4,0) - 2.0), 1.0E-4);
        EXPECT_LT(fabs(tmp_y(4,1) - 3.0), 1.0E-4);
        EXPECT_LT(fabs(tmp_y(5,0) - 0.0), 1.0E-4);
        EXPECT_LT(fabs(tmp_y(5,1) - 1.0), 1.0E-4);
    }

    TEST_F(FullyConnectedTest_w_map, GPU_float_calc_delta_test) {
        cudaMatrix<Real> nx_delta(2, 2);

        layer->calc_delta(x, x, delta, nx_delta);
        
        auto tmp_nx_delta = nx_delta.get_matrix();
        EXPECT_LT(fabs(tmp_nx_delta(0,0) - (-0.7)), 1.0E-4);
        EXPECT_LT(fabs(tmp_nx_delta(0,1) - 0.4), 1.0E-4);
        EXPECT_LT(fabs(tmp_nx_delta(1,0) - 1.4), 1.0E-4);
        EXPECT_LT(fabs(tmp_nx_delta(1,1) - 0.6), 1.0E-4);
    }

    TEST_F(FullyConnectedTest_w_map, GPU_float_calc_gradient_test) {
        std::vector<cudaMatrix<Real>> nabla_W, nabla_b;
        nabla_W.emplace_back(2*3, 2);
        
        layer->calc_gradient(x, x, delta, nabla_W, nabla_b);
        
        auto tmp_nabla_W = nabla_W[0].get_matrix();
        EXPECT_LT(fabs(tmp_nabla_W(0,0) - 0.3), 1.0E-4);
        EXPECT_LT(fabs(tmp_nabla_W(0,1) - 1.0), 1.0E-4);
        EXPECT_LT(fabs(tmp_nabla_W(1,0) - 0.0), 1.0E-4);
        EXPECT_LT(fabs(tmp_nabla_W(1,1) - 0.2), 1.0E-4);
        EXPECT_LT(fabs(tmp_nabla_W(2,0) - 0.0), 1.0E-4);
        EXPECT_LT(fabs(tmp_nabla_W(2,1) - (-0.1)), 1.0E-4);
        EXPECT_LT(fabs(tmp_nabla_W(3,0) - (-0.2)), 1.0E-4);
        EXPECT_LT(fabs(tmp_nabla_W(3,1) - (-0.6)), 1.0E-4);
        EXPECT_LT(fabs(tmp_nabla_W(4,0) - 0.7), 1.0E-4);
        EXPECT_LT(fabs(tmp_nabla_W(4,1) - 2.0), 1.0E-4);
        EXPECT_LT(fabs(tmp_nabla_W(5,0) - (-0.1)), 1.0E-4);
        EXPECT_LT(fabs(tmp_nabla_W(5,1) - (-0.5)), 1.0E-4);
    }

    class FullyConnectedTest_w_map_bias : public ::testing::Test {
    protected:
        std::shared_ptr<Function<Real>> f;
        std::shared_ptr<Layer<cudaMatrix, Real>> layer;
        cudaMatrix<Real> x, delta;
        
        virtual void SetUp () {
            f = std::shared_ptr<Function<Real>>(new ReLU<Real>());
            layer = std::shared_ptr<Layer<cudaMatrix, Real>>(new FullyConnected<cudaMatrix, Real>(1, 2, 2, 3, f, true));
            
            std::vector<cudaMatrix<Real>> W;
            Matrix<Real> tmp_W(2*3, 2);
            tmp_W(0,0) = -1; tmp_W(0,1) = 1;
            tmp_W(1,0) = 2; tmp_W(1,1) = -3;
            tmp_W(2,0) = 0; tmp_W(2,1) = -1;

            tmp_W(3,0) = 1; tmp_W(3,1) = 0;
            tmp_W(4,0) = 0; tmp_W(4,1) = 1;
            tmp_W(5,0) = -2; tmp_W(5,1) = 1;
            W.emplace_back(tmp_W);

            std::vector<cudaMatrix<Real>> b;
            Matrix<Real> tmp_b(2*3, 1);
            tmp_b(0,0) = 1;
            tmp_b(1,0) = -1;
            tmp_b(2,0) = 1;

            tmp_b(3,0) = 1;
            tmp_b(4,0) = -1;
            tmp_b(5,0) = 1;
            b.emplace_back(tmp_b);

            this->layer->set_W(W);
            this->layer->set_b(b);

            Matrix<Real> tmp_x(2, 2);
            tmp_x(0, 0) = 1; tmp_x(0, 1) = 1;
            tmp_x(1, 0) = 2; tmp_x(1, 1) = 3;
            x = tmp_x;

            Matrix<Real> tmp_delta(2*3, 2);
            tmp_delta(0, 0) = -0.1; tmp_delta(0, 1) = 0.4;
            tmp_delta(1, 0) = -0.2; tmp_delta(1, 1) = 0.2;
            tmp_delta(2, 0) = 0.1; tmp_delta(2, 1) = -0.1;

            tmp_delta(3, 0) = 0.0; tmp_delta(3, 1) = -0.2;
            tmp_delta(4, 0) = 0.1; tmp_delta(4, 1) = 0.6;
            tmp_delta(5, 0) = 0.2; tmp_delta(5, 1) = -0.3;
            delta = tmp_delta;
        }

        virtual void TearDown () {
            
        }
    };

    TEST_F(FullyConnectedTest_w_map_bias, GPU_float_apply_test_without_function) {
        auto y = layer->apply(x, false);

        auto tmp_y = y.get_matrix();
        EXPECT_LT(fabs(tmp_y(0,0) - 2.0), 1.0E-4);
        EXPECT_LT(fabs(tmp_y(0,1) - 3.0), 1.0E-4);
        EXPECT_LT(fabs(tmp_y(1,0) - (-5.0)), 1.0E-4);
        EXPECT_LT(fabs(tmp_y(1,1) - (-8.0)), 1.0E-4);
        EXPECT_LT(fabs(tmp_y(2,0) - (-1.0)), 1.0E-4);
        EXPECT_LT(fabs(tmp_y(2,1) - (-2.0)), 1.0E-4);

        EXPECT_LT(fabs(tmp_y(3,0) - 2.0), 1.0E-4);
        EXPECT_LT(fabs(tmp_y(3,1) - 2.0), 1.0E-4);
        EXPECT_LT(fabs(tmp_y(4,0) - 1.0), 1.0E-4);
        EXPECT_LT(fabs(tmp_y(4,1) - 2.0), 1.0E-4);
        EXPECT_LT(fabs(tmp_y(5,0) - 1.0), 1.0E-4);
        EXPECT_LT(fabs(tmp_y(5,1) - 2.0), 1.0E-4);
    }

    TEST_F(FullyConnectedTest_w_map_bias, GPU_float_apply_test_with_function) {
        auto y = layer->apply(x, true);

        auto tmp_y = y.get_matrix();
        EXPECT_LT(fabs(tmp_y(0,0) - 2.0), 1.0E-4);
        EXPECT_LT(fabs(tmp_y(0,1) - 3.0), 1.0E-4);
        EXPECT_LT(fabs(tmp_y(1,0) - 0.0), 1.0E-4);
        EXPECT_LT(fabs(tmp_y(1,1) - 0.0), 1.0E-4);
        EXPECT_LT(fabs(tmp_y(2,0) - 0.0), 1.0E-4);
        EXPECT_LT(fabs(tmp_y(2,1) - 0.0), 1.0E-4);

        EXPECT_LT(fabs(tmp_y(3,0) - 2.0), 1.0E-4);
        EXPECT_LT(fabs(tmp_y(3,1) - 2.0), 1.0E-4);
        EXPECT_LT(fabs(tmp_y(4,0) - 1.0), 1.0E-4);
        EXPECT_LT(fabs(tmp_y(4,1) - 2.0), 1.0E-4);
        EXPECT_LT(fabs(tmp_y(5,0) - 1.0), 1.0E-4);
        EXPECT_LT(fabs(tmp_y(5,1) - 2.0), 1.0E-4);
    }

    TEST_F(FullyConnectedTest_w_map_bias, GPU_float_calc_delta_test) {
        cudaMatrix<Real> nx_delta(2, 2);

        layer->calc_delta(x, x, delta, nx_delta);
        
        auto tmp_nx_delta = nx_delta.get_matrix();
        EXPECT_LT(fabs(tmp_nx_delta(0,0) - (-0.7)), 1.0E-4);
        EXPECT_LT(fabs(tmp_nx_delta(0,1) - 0.4), 1.0E-4);
        EXPECT_LT(fabs(tmp_nx_delta(1,0) - 1.4), 1.0E-4);
        EXPECT_LT(fabs(tmp_nx_delta(1,1) - 0.6), 1.0E-4);
    }

    TEST_F(FullyConnectedTest_w_map_bias, GPU_float_calc_gradient_test) {
        std::vector<cudaMatrix<Real>> nabla_W, nabla_b;
        nabla_W.emplace_back(2*3, 2);
        nabla_b.emplace_back(2*3, 1);
        
        layer->calc_gradient(x, x, delta, nabla_W, nabla_b);
        
        auto tmp_nabla_W = nabla_W[0].get_matrix();
        EXPECT_LT(fabs(tmp_nabla_W(0,0) - 0.3), 1.0E-4);
        EXPECT_LT(fabs(tmp_nabla_W(0,1) - 1.0), 1.0E-4);
        EXPECT_LT(fabs(tmp_nabla_W(1,0) - 0.0), 1.0E-4);
        EXPECT_LT(fabs(tmp_nabla_W(1,1) - 0.2), 1.0E-4);
        EXPECT_LT(fabs(tmp_nabla_W(2,0) - 0.0), 1.0E-4);
        EXPECT_LT(fabs(tmp_nabla_W(2,1) - (-0.1)), 1.0E-4);
        EXPECT_LT(fabs(tmp_nabla_W(3,0) - (-0.2)), 1.0E-4);
        EXPECT_LT(fabs(tmp_nabla_W(3,1) - (-0.6)), 1.0E-4);
        EXPECT_LT(fabs(tmp_nabla_W(4,0) - 0.7), 1.0E-4);
        EXPECT_LT(fabs(tmp_nabla_W(4,1) - 2.0), 1.0E-4);
        EXPECT_LT(fabs(tmp_nabla_W(5,0) - (-0.1)), 1.0E-4);
        EXPECT_LT(fabs(tmp_nabla_W(5,1) - (-0.5)), 1.0E-4);

        auto tmp_nabla_b = nabla_b[0].get_matrix();
        EXPECT_LT(fabs(tmp_nabla_b(0,0) - 0.3), 1.0E-4);
        EXPECT_LT(fabs(tmp_nabla_b(1,0) - 0.0), 1.0E-4);
        EXPECT_LT(fabs(tmp_nabla_b(2,0) - 0.0), 1.0E-4);
        EXPECT_LT(fabs(tmp_nabla_b(3,0) - (-0.2)), 1.0E-4);
        EXPECT_LT(fabs(tmp_nabla_b(4,0) - 0.7), 1.0E-4);
        EXPECT_LT(fabs(tmp_nabla_b(5,0) - (-0.1)), 1.0E-4);
    }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
