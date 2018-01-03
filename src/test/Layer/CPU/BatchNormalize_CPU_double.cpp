#include <algorithm>
#include <vector>
#include <random>

#include <gtest/gtest.h>
#include <Layer/BatchNormalize.hpp>

namespace {
    typedef double Real;

    class BatchNormalizeTest : public ::testing::Test {
    protected:
        std::shared_ptr<Function<Real>> f;
        std::shared_ptr<Layer<Matrix, Real>> layer;
        Matrix<Real> x, delta;
        std::mt19937 mt;

        virtual void SetUp () {
            f = std::shared_ptr<Function<Real>>(new Identity<Real>());
            layer = std::shared_ptr<Layer<Matrix, Real>>(new BatchNormalize<Matrix, Real>(1, 2, f));
            layer->init(mt);
            layer->set_learning();
            
            std::vector<Matrix<Real>> W, b;
            W.emplace_back(1, 1);
            W[0](0,0) = 1;
            b.emplace_back(1, 1);
            b[0](0,0) = 0;
            
            this->layer->set_W(W);
            this->layer->set_b(b);

            x = Matrix<Real>(2, 2);
            x(0, 0) = 1; x(0, 1) = 5;
            x(1, 0) = 2; x(1, 1) = 3;

            delta = Matrix<Real>(2, 2);
            delta(0, 0) = -0.1; delta(0, 1) = 0.4;
            delta(1, 0) = -0.2; delta(1, 1) = 0.2;
        }

        virtual void TearDown () {
            
        }
    };

    TEST_F(BatchNormalizeTest, CPU_double_apply_without_function) {
        auto y = layer->apply(x, false);

        EXPECT_LT(fabs(y(0,0) - (1.0 - 3.0)/sqrt(4.0 + 1.0E-8)), 1.0E-8);
        EXPECT_LT(fabs(y(0,1) - (5.0 - 3.0)/sqrt(4.0 + 1.0E-8)), 1.0E-8);
        EXPECT_LT(fabs(y(1,0) - (2.0 - 2.5)/sqrt(0.25 + 1.0E-8)), 1.0E-8);
        EXPECT_LT(fabs(y(1,1) - (3.0 - 2.5)/sqrt(0.25 + 1.0E-8)), 1.0E-8);
    }

    TEST_F(BatchNormalizeTest, CPU_double_calc_delta) {
        auto y = layer->apply(x, false);

        Matrix<Real> nx_delta(2, 2);

        layer->calc_delta(x, f->operator()(x, true), delta, nx_delta);

        Real sigma[2] = {
            -0.1*1.0 * (1.0 - 3.0) + 0.4*1.0 * (5.0 - 3.0),
            -0.2*1.0 * (2.0 - 2.5) + 0.2*1.0 * (3.0 - 2.5)
        };
        Real mu[2] = {
            -0.1*1.0 + 0.4*1.0,
            -0.2*1.0 + 0.2*1.0
        };

        Real val[4] = {
            -0.1*1.0*1.0/sqrt(4.0 + 1.0E-8) - sigma[0]/2.0*(1.0 - 3.0)*pow((4.0 + 1.0E-8), -3.0/2.0) - mu[0]/2.0*1.0/sqrt(4.0 + 1.0E-8),
            0.4*1.0*1.0/sqrt(4.0 + 1.0E-8) - sigma[0]/2.0*(5.0 - 3.0)*pow((4.0 + 1.0E-8), -3.0/2.0) - mu[0]/2.0*1.0/sqrt(4.0 + 1.0E-8),
            -0.2*1.0*1.0/sqrt(0.25 + 1.0E-8) - sigma[1]/2.0*(2.0 - 2.5)*pow((0.25 + 1.0E-8), -3.0/2.0) - mu[1]/2.0*1.0/sqrt(0.25 + 1.0E-8),
            0.2*1.0*1.0/sqrt(0.25 + 1.0E-8) - sigma[1]/2.0*(3.0 - 2.5)*pow((0.25 + 1.0E-8), -3.0/2.0) - mu[1]/2.0*1.0/sqrt(0.25 + 1.0E-8)
        };
        EXPECT_LT(fabs(nx_delta(0,0) - val[0]), 1.0E-8);
        EXPECT_LT(fabs(nx_delta(0,1) - val[1]), 1.0E-8);
        EXPECT_LT(fabs(nx_delta(1,0) - val[2]), 1.0E-8);
        EXPECT_LT(fabs(nx_delta(1,1) - val[3]), 1.0E-8);
    }

    TEST_F(BatchNormalizeTest, CPU_double_calc_gradient) {
        auto y = layer->apply(x, false);

        std::vector<Matrix<Real>> nabla_W, nabla_b;
        nabla_W.emplace_back(1, 1); nabla_b.emplace_back(1, 1);

        layer->calc_gradient(x, x, delta, nabla_W, nabla_b);

        const Real val_nabla_W = -0.1 * (1.0 - 3.0)/sqrt(4.0 + 1.0E-8) + 0.4 * (5.0 - 3.0)/sqrt(4.0 + 1.0E-8) + -0.2 * (2.0 - 2.5)/sqrt(0.25 + 1.0E-8) + 0.2 * (3.0 - 2.5)/sqrt(0.25 + 1.0E-8);
        const Real val_nabla_b = -0.1 + 0.4 + -0.2 + 0.2;
        EXPECT_LT(fabs(nabla_W[0](0,0) - val_nabla_W), 1.0E-8);
        EXPECT_LT(fabs(nabla_b[0](0,0) - val_nabla_b), 1.0E-8);
    }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
