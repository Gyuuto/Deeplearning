#include <algorithm>
#include <vector>
#include <random>

#include <gtest/gtest.h>
#include <Layer/FullyConnected.hpp>

namespace {
    typedef float Real;

    class FullyConnectedTest : public ::testing::Test {
    protected:
        std::shared_ptr<Function<Real>> f;
        std::shared_ptr<Layer<Matrix, Real>> layer;
        Matrix<Real> x, delta;
        std::mt19937 mt;
        
        virtual void SetUp () {
            f = std::shared_ptr<Function<Real>>(new ReLU<Real>());
            layer = std::shared_ptr<Layer<Matrix, Real>>(new FullyConnected<Matrix, Real>(1, 2, 1, 3, f, false));
            layer->init(mt);
            layer->set_learning();
            
            std::vector<Matrix<Real>> W;
            W.emplace_back(3, 2);
            W[0](0,0) = -1; W[0](0,1) = 1;
            W[0](1,0) = 2; W[0](1,1) = -3;
            W[0](2,0) = 0; W[0](2,1) = -1;

            this->layer->set_W(W);

            x = Matrix<Real>(2, 2);
            x(0, 0) = 1; x(0, 1) = 1;
            x(1, 0) = 2; x(1, 1) = 3;

            delta = Matrix<Real>(3, 2);
            delta(0, 0) = -0.1; delta(0, 1) = 0.4;
            delta(1, 0) = -0.2; delta(1, 1) = 0.2;
            delta(2, 0) = 0.1; delta(2, 1) = -0.1;
        }

        virtual void TearDown () {
            
        }
    };

    TEST_F(FullyConnectedTest, CPU_float_apply_test_without_function) {
        auto y = layer->apply(x, false);

        EXPECT_LT(fabs(y(0,0) - 1.0), 1.0E-4);
        EXPECT_LT(fabs(y(0,1) - 2.0), 1.0E-4);
        EXPECT_LT(fabs(y(1,0) - (-4.0)), 1.0E-4);
        EXPECT_LT(fabs(y(1,1) - (-7.0)), 1.0E-4);
        EXPECT_LT(fabs(y(2,0) - (-2.0)), 1.0E-4);
        EXPECT_LT(fabs(y(2,1) - (-3.0)), 1.0E-4);
    }

    TEST_F(FullyConnectedTest, CPU_float_apply_test_with_function) {
        auto y = layer->apply(x, true);

        EXPECT_LT(fabs(y(0,0) - 1.0), 1.0E-4);
        EXPECT_LT(fabs(y(0,1) - 2.0), 1.0E-4);
        EXPECT_LT(fabs(y(1,0) - 0.0), 1.0E-4);
        EXPECT_LT(fabs(y(1,1) - 0.0), 1.0E-4);
        EXPECT_LT(fabs(y(2,0) - 0.0), 1.0E-4);
        EXPECT_LT(fabs(y(2,1) - 0.0), 1.0E-4);
    }

    TEST_F(FullyConnectedTest, CPU_float_calc_delta_test) {
        Matrix<Real> nx_delta(2, 2);

        layer->calc_delta(x, x, delta, nx_delta);
        
        EXPECT_LT(fabs(nx_delta(0,0) - (-0.3)), 1.0E-4);
        EXPECT_LT(fabs(nx_delta(0,1) - 0.0), 1.0E-4);
        EXPECT_LT(fabs(nx_delta(1,0) - 0.8), 1.0E-4);
        EXPECT_LT(fabs(nx_delta(1,1) - (-0.3)), 1.0E-4);
    }

    TEST_F(FullyConnectedTest, CPU_float_calc_gradient_test) {
        std::vector<Matrix<Real>> nabla_W, nabla_b;
        nabla_W.emplace_back(3, 2);
        
        layer->calc_gradient(x, x, delta, nabla_W, nabla_b);
        
        EXPECT_LT(fabs(nabla_W[0](0,0) - 0.3), 1.0E-4);
        EXPECT_LT(fabs(nabla_W[0](0,1) - 1.0), 1.0E-4);
        EXPECT_LT(fabs(nabla_W[0](1,0) - 0.0), 1.0E-4);
        EXPECT_LT(fabs(nabla_W[0](1,1) - 0.2), 1.0E-4);
        EXPECT_LT(fabs(nabla_W[0](2,0) - 0.0), 1.0E-4);
        EXPECT_LT(fabs(nabla_W[0](2,1) - (-0.1)), 1.0E-4);
    }

    class FullyConnectedTest_bias : public ::testing::Test {
    protected:
        std::shared_ptr<Function<Real>> f;
        std::shared_ptr<Layer<Matrix, Real>> layer;
        Matrix<Real> x, delta;
        std::mt19937 mt;
        
        virtual void SetUp () {
            f = std::shared_ptr<Function<Real>>(new ReLU<Real>());
            layer = std::shared_ptr<Layer<Matrix, Real>>(new FullyConnected<Matrix, Real>(1, 2, 1, 3, f, true));
            layer->init(mt);
            layer->set_learning();
            
            std::vector<Matrix<Real>> W;
            W.emplace_back(3, 2);
            W[0](0,0) = -1; W[0](0,1) = 1;
            W[0](1,0) = 2; W[0](1,1) = -3;
            W[0](2,0) = 0; W[0](2,1) = -1;

            std::vector<Matrix<Real>> b;
            b.emplace_back(3, 1);
            b[0](0,0) = 1;
            b[0](1,0) = -1;
            b[0](2,0) = 1;

            this->layer->set_W(W);
            this->layer->set_b(b);

            x = Matrix<Real>(2, 2);
            x(0, 0) = 1; x(0, 1) = 1;
            x(1, 0) = 2; x(1, 1) = 3;

            delta = Matrix<Real>(3, 2);
            delta(0, 0) = -0.1; delta(0, 1) = 0.4;
            delta(1, 0) = -0.2; delta(1, 1) = 0.2;
            delta(2, 0) = 0.1; delta(2, 1) = -0.1;
        }

        virtual void TearDown () {
            
        }
    };

    TEST_F(FullyConnectedTest_bias, CPU_float_apply_test_without_function) {
        auto y = layer->apply(x, false);

        EXPECT_LT(fabs(y(0,0) - 2.0), 1.0E-4);
        EXPECT_LT(fabs(y(0,1) - 3.0), 1.0E-4);
        EXPECT_LT(fabs(y(1,0) - (-5.0)), 1.0E-4);
        EXPECT_LT(fabs(y(1,1) - (-8.0)), 1.0E-4);
        EXPECT_LT(fabs(y(2,0) - (-1.0)), 1.0E-4);
        EXPECT_LT(fabs(y(2,1) - (-2.0)), 1.0E-4);
    }

    TEST_F(FullyConnectedTest_bias, CPU_float_apply_test_with_function) {
        auto y = layer->apply(x, true);

        EXPECT_LT(fabs(y(0,0) - 2.0), 1.0E-4);
        EXPECT_LT(fabs(y(0,1) - 3.0), 1.0E-4);
        EXPECT_LT(fabs(y(1,0) - 0.0), 1.0E-4);
        EXPECT_LT(fabs(y(1,1) - 0.0), 1.0E-4);
        EXPECT_LT(fabs(y(2,0) - 0.0), 1.0E-4);
        EXPECT_LT(fabs(y(2,1) - 0.0), 1.0E-4);
    }

    TEST_F(FullyConnectedTest_bias, CPU_float_calc_delta_test) {
        Matrix<Real> nx_delta(2, 2);

        layer->calc_delta(x, x, delta, nx_delta);
        
        EXPECT_LT(fabs(nx_delta(0,0) - (-0.3)), 1.0E-4);
        EXPECT_LT(fabs(nx_delta(0,1) - 0.0), 1.0E-4);
        EXPECT_LT(fabs(nx_delta(1,0) - 0.8), 1.0E-4);
        EXPECT_LT(fabs(nx_delta(1,1) - (-0.3)), 1.0E-4);
    }

    TEST_F(FullyConnectedTest_bias, CPU_float_calc_gradient_test) {
        std::vector<Matrix<Real>> nabla_W, nabla_b;
        nabla_W.emplace_back(3, 2);
        nabla_b.emplace_back(3, 1);
        
        layer->calc_gradient(x, x, delta, nabla_W, nabla_b);
        
        EXPECT_LT(fabs(nabla_W[0](0,0) - 0.3), 1.0E-4);
        EXPECT_LT(fabs(nabla_W[0](0,1) - 1.0), 1.0E-4);
        EXPECT_LT(fabs(nabla_W[0](1,0) - 0.0), 1.0E-4);
        EXPECT_LT(fabs(nabla_W[0](1,1) - 0.2), 1.0E-4);
        EXPECT_LT(fabs(nabla_W[0](2,0) - 0.0), 1.0E-4);
        EXPECT_LT(fabs(nabla_W[0](2,1) - (-0.1)), 1.0E-4);

        EXPECT_LT(fabs(nabla_b[0](0,0) - 0.3), 1.0E-4);
        EXPECT_LT(fabs(nabla_b[0](1,0) - 0.0), 1.0E-4);
        EXPECT_LT(fabs(nabla_b[0](2,0) - 0.0), 1.0E-4);
    }
    
    class FullyConnectedTest_w_map : public ::testing::Test {
    protected:
        std::shared_ptr<Function<Real>> f;
        std::shared_ptr<Layer<Matrix, Real>> layer;
        Matrix<Real> x, delta;
        std::mt19937 mt;
        
        virtual void SetUp () {
            f = std::shared_ptr<Function<Real>>(new ReLU<Real>());
            layer = std::shared_ptr<Layer<Matrix, Real>>(new FullyConnected<Matrix, Real>(1, 2, 2, 3, f, false));
            layer->init(mt);
            layer->set_learning();
            
            std::vector<Matrix<Real>> W;
            W.emplace_back(2*3, 2);
            W[0](0,0) = -1; W[0](0,1) = 1;
            W[0](1,0) = 2; W[0](1,1) = -3;
            W[0](2,0) = 0; W[0](2,1) = -1;

            W[0](3,0) = 1; W[0](3,1) = 0;
            W[0](4,0) = 0; W[0](4,1) = 1;
            W[0](5,0) = -2; W[0](5,1) = 1;

            this->layer->set_W(W);

            x = Matrix<Real>(2, 2);
            x(0, 0) = 1; x(0, 1) = 1;
            x(1, 0) = 2; x(1, 1) = 3;

            delta = Matrix<Real>(2*3, 2);
            delta(0, 0) = -0.1; delta(0, 1) = 0.4;
            delta(1, 0) = -0.2; delta(1, 1) = 0.2;
            delta(2, 0) = 0.1; delta(2, 1) = -0.1;

            delta(3, 0) = 0.0; delta(3, 1) = -0.2;
            delta(4, 0) = 0.1; delta(4, 1) = 0.6;
            delta(5, 0) = 0.2; delta(5, 1) = -0.3;
        }

        virtual void TearDown () {
            
        }
    };

    TEST_F(FullyConnectedTest_w_map, CPU_float_apply_test_without_function) {
        auto y = layer->apply(x, false);

        EXPECT_LT(fabs(y(0,0) - 1.0), 1.0E-4);
        EXPECT_LT(fabs(y(0,1) - 2.0), 1.0E-4);
        EXPECT_LT(fabs(y(1,0) - (-4.0)), 1.0E-4);
        EXPECT_LT(fabs(y(1,1) - (-7.0)), 1.0E-4);
        EXPECT_LT(fabs(y(2,0) - (-2.0)), 1.0E-4);
        EXPECT_LT(fabs(y(2,1) - (-3.0)), 1.0E-4);

        EXPECT_LT(fabs(y(3,0) - 1.0), 1.0E-4);
        EXPECT_LT(fabs(y(3,1) - 1.0), 1.0E-4);
        EXPECT_LT(fabs(y(4,0) - 2.0), 1.0E-4);
        EXPECT_LT(fabs(y(4,1) - 3.0), 1.0E-4);
        EXPECT_LT(fabs(y(5,0) - 0.0), 1.0E-4);
        EXPECT_LT(fabs(y(5,1) - 1.0), 1.0E-4);
    }

    TEST_F(FullyConnectedTest_w_map, CPU_float_apply_test_with_function) {
        auto y = layer->apply(x, true);

        EXPECT_LT(fabs(y(0,0) - 1.0), 1.0E-4);
        EXPECT_LT(fabs(y(0,1) - 2.0), 1.0E-4);
        EXPECT_LT(fabs(y(1,0) - 0.0), 1.0E-4);
        EXPECT_LT(fabs(y(1,1) - 0.0), 1.0E-4);
        EXPECT_LT(fabs(y(2,0) - 0.0), 1.0E-4);
        EXPECT_LT(fabs(y(2,1) - 0.0), 1.0E-4);

        EXPECT_LT(fabs(y(3,0) - 1.0), 1.0E-4);
        EXPECT_LT(fabs(y(3,1) - 1.0), 1.0E-4);
        EXPECT_LT(fabs(y(4,0) - 2.0), 1.0E-4);
        EXPECT_LT(fabs(y(4,1) - 3.0), 1.0E-4);
        EXPECT_LT(fabs(y(5,0) - 0.0), 1.0E-4);
        EXPECT_LT(fabs(y(5,1) - 1.0), 1.0E-4);
    }

    TEST_F(FullyConnectedTest_w_map, CPU_float_calc_delta_test) {
        Matrix<Real> nx_delta(2, 2);

        layer->calc_delta(x, x, delta, nx_delta);
        
        EXPECT_LT(fabs(nx_delta(0,0) - (-0.7)), 1.0E-4);
        EXPECT_LT(fabs(nx_delta(0,1) - 0.4), 1.0E-4);
        EXPECT_LT(fabs(nx_delta(1,0) - 1.4), 1.0E-4);
        EXPECT_LT(fabs(nx_delta(1,1) - 0.6), 1.0E-4);
    }

    TEST_F(FullyConnectedTest_w_map, CPU_float_calc_gradient_test) {
        std::vector<Matrix<Real>> nabla_W, nabla_b;
        nabla_W.emplace_back(2*3, 2);
        
        layer->calc_gradient(x, x, delta, nabla_W, nabla_b);
        
        EXPECT_LT(fabs(nabla_W[0](0,0) - 0.3), 1.0E-4);
        EXPECT_LT(fabs(nabla_W[0](0,1) - 1.0), 1.0E-4);
        EXPECT_LT(fabs(nabla_W[0](1,0) - 0.0), 1.0E-4);
        EXPECT_LT(fabs(nabla_W[0](1,1) - 0.2), 1.0E-4);
        EXPECT_LT(fabs(nabla_W[0](2,0) - 0.0), 1.0E-4);
        EXPECT_LT(fabs(nabla_W[0](2,1) - (-0.1)), 1.0E-4);
        EXPECT_LT(fabs(nabla_W[0](3,0) - (-0.2)), 1.0E-4);
        EXPECT_LT(fabs(nabla_W[0](3,1) - (-0.6)), 1.0E-4);
        EXPECT_LT(fabs(nabla_W[0](4,0) - 0.7), 1.0E-4);
        EXPECT_LT(fabs(nabla_W[0](4,1) - 2.0), 1.0E-4);
        EXPECT_LT(fabs(nabla_W[0](5,0) - (-0.1)), 1.0E-4);
        EXPECT_LT(fabs(nabla_W[0](5,1) - (-0.5)), 1.0E-4);
    }

    class FullyConnectedTest_w_map_bias : public ::testing::Test {
    protected:
        std::shared_ptr<Function<Real>> f;
        std::shared_ptr<Layer<Matrix, Real>> layer;
        Matrix<Real> x, delta;
        std::mt19937 mt;
        
        virtual void SetUp () {
            f = std::shared_ptr<Function<Real>>(new ReLU<Real>());
            layer = std::shared_ptr<Layer<Matrix, Real>>(new FullyConnected<Matrix, Real>(1, 2, 2, 3, f, true));
            layer->init(mt);
            layer->set_learning();
            
            std::vector<Matrix<Real>> W;
            W.emplace_back(2*3, 2);
            W[0](0,0) = -1; W[0](0,1) = 1;
            W[0](1,0) = 2; W[0](1,1) = -3;
            W[0](2,0) = 0; W[0](2,1) = -1;

            W[0](3,0) = 1; W[0](3,1) = 0;
            W[0](4,0) = 0; W[0](4,1) = 1;
            W[0](5,0) = -2; W[0](5,1) = 1;

            std::vector<Matrix<Real>> b;
            b.emplace_back(2*3, 1);
            b[0](0,0) = 1;
            b[0](1,0) = -1;
            b[0](2,0) = 1;

            b[0](3,0) = 1;
            b[0](4,0) = -1;
            b[0](5,0) = 1;

            this->layer->set_W(W);
            this->layer->set_b(b);

            x = Matrix<Real>(2, 2);
            x(0, 0) = 1; x(0, 1) = 1;
            x(1, 0) = 2; x(1, 1) = 3;

            delta = Matrix<Real>(2*3, 2);
            delta(0, 0) = -0.1; delta(0, 1) = 0.4;
            delta(1, 0) = -0.2; delta(1, 1) = 0.2;
            delta(2, 0) = 0.1; delta(2, 1) = -0.1;

            delta(3, 0) = 0.0; delta(3, 1) = -0.2;
            delta(4, 0) = 0.1; delta(4, 1) = 0.6;
            delta(5, 0) = 0.2; delta(5, 1) = -0.3;
        }

        virtual void TearDown () {
            
        }
    };

    TEST_F(FullyConnectedTest_w_map_bias, CPU_float_apply_test_without_function) {
        auto y = layer->apply(x, false);

        EXPECT_LT(fabs(y(0,0) - 2.0), 1.0E-4);
        EXPECT_LT(fabs(y(0,1) - 3.0), 1.0E-4);
        EXPECT_LT(fabs(y(1,0) - (-5.0)), 1.0E-4);
        EXPECT_LT(fabs(y(1,1) - (-8.0)), 1.0E-4);
        EXPECT_LT(fabs(y(2,0) - (-1.0)), 1.0E-4);
        EXPECT_LT(fabs(y(2,1) - (-2.0)), 1.0E-4);

        EXPECT_LT(fabs(y(3,0) - 2.0), 1.0E-4);
        EXPECT_LT(fabs(y(3,1) - 2.0), 1.0E-4);
        EXPECT_LT(fabs(y(4,0) - 1.0), 1.0E-4);
        EXPECT_LT(fabs(y(4,1) - 2.0), 1.0E-4);
        EXPECT_LT(fabs(y(5,0) - 1.0), 1.0E-4);
        EXPECT_LT(fabs(y(5,1) - 2.0), 1.0E-4);
    }

    TEST_F(FullyConnectedTest_w_map_bias, CPU_float_apply_test_with_function) {
        auto y = layer->apply(x, true);

        EXPECT_LT(fabs(y(0,0) - 2.0), 1.0E-4);
        EXPECT_LT(fabs(y(0,1) - 3.0), 1.0E-4);
        EXPECT_LT(fabs(y(1,0) - 0.0), 1.0E-4);
        EXPECT_LT(fabs(y(1,1) - 0.0), 1.0E-4);
        EXPECT_LT(fabs(y(2,0) - 0.0), 1.0E-4);
        EXPECT_LT(fabs(y(2,1) - 0.0), 1.0E-4);

        EXPECT_LT(fabs(y(3,0) - 2.0), 1.0E-4);
        EXPECT_LT(fabs(y(3,1) - 2.0), 1.0E-4);
        EXPECT_LT(fabs(y(4,0) - 1.0), 1.0E-4);
        EXPECT_LT(fabs(y(4,1) - 2.0), 1.0E-4);
        EXPECT_LT(fabs(y(5,0) - 1.0), 1.0E-4);
        EXPECT_LT(fabs(y(5,1) - 2.0), 1.0E-4);
    }

    TEST_F(FullyConnectedTest_w_map_bias, CPU_float_calc_delta_test) {
        Matrix<Real> nx_delta(2, 2);

        layer->calc_delta(x, x, delta, nx_delta);
        
        EXPECT_LT(fabs(nx_delta(0,0) - (-0.7)), 1.0E-4);
        EXPECT_LT(fabs(nx_delta(0,1) - 0.4), 1.0E-4);
        EXPECT_LT(fabs(nx_delta(1,0) - 1.4), 1.0E-4);
        EXPECT_LT(fabs(nx_delta(1,1) - 0.6), 1.0E-4);
    }

    TEST_F(FullyConnectedTest_w_map_bias, CPU_float_calc_gradient_test) {
        std::vector<Matrix<Real>> nabla_W, nabla_b;
        nabla_W.emplace_back(2*3, 2);
        nabla_b.emplace_back(2*3, 1);
        
        layer->calc_gradient(x, x, delta, nabla_W, nabla_b);
        
        EXPECT_LT(fabs(nabla_W[0](0,0) - 0.3), 1.0E-4);
        EXPECT_LT(fabs(nabla_W[0](0,1) - 1.0), 1.0E-4);
        EXPECT_LT(fabs(nabla_W[0](1,0) - 0.0), 1.0E-4);
        EXPECT_LT(fabs(nabla_W[0](1,1) - 0.2), 1.0E-4);
        EXPECT_LT(fabs(nabla_W[0](2,0) - 0.0), 1.0E-4);
        EXPECT_LT(fabs(nabla_W[0](2,1) - (-0.1)), 1.0E-4);
        EXPECT_LT(fabs(nabla_W[0](3,0) - (-0.2)), 1.0E-4);
        EXPECT_LT(fabs(nabla_W[0](3,1) - (-0.6)), 1.0E-4);
        EXPECT_LT(fabs(nabla_W[0](4,0) - 0.7), 1.0E-4);
        EXPECT_LT(fabs(nabla_W[0](4,1) - 2.0), 1.0E-4);
        EXPECT_LT(fabs(nabla_W[0](5,0) - (-0.1)), 1.0E-4);
        EXPECT_LT(fabs(nabla_W[0](5,1) - (-0.5)), 1.0E-4);

        EXPECT_LT(fabs(nabla_b[0](0,0) - 0.3), 1.0E-4);
        EXPECT_LT(fabs(nabla_b[0](1,0) - 0.0), 1.0E-4);
        EXPECT_LT(fabs(nabla_b[0](2,0) - 0.0), 1.0E-4);
        EXPECT_LT(fabs(nabla_b[0](3,0) - (-0.2)), 1.0E-4);
        EXPECT_LT(fabs(nabla_b[0](4,0) - 0.7), 1.0E-4);
        EXPECT_LT(fabs(nabla_b[0](5,0) - (-0.1)), 1.0E-4);
    }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
