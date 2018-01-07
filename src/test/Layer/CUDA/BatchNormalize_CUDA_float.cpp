#include <algorithm>
#include <vector>
#include <random>

#include <gtest/gtest.h>
#include <clMatrix.hpp>
#include <Layer/BatchNormalize.hpp>

namespace {
    typedef float Real;
    const double EPS = 1.0E-6;
    
    class BatchNormalizeTest : public ::testing::Test {
    protected:
        std::shared_ptr<Function<Real>> f;
        std::shared_ptr<Layer<clMatrix, Real>> layer;
        clMatrix<Real> x, delta;
        std::mt19937 mt;

        virtual void SetUp () {
            f = std::shared_ptr<Function<Real>>(new Identity<Real>());
            layer = std::shared_ptr<Layer<clMatrix, Real>>(new BatchNormalize<clMatrix, Real>(1, 2, f, 0.999, EPS));
            layer->init(mt);
            layer->set_learning();
            
            std::vector<clMatrix<Real>> W, b;
            Matrix<Real> tmp_W(1, 1);
            tmp_W(0,0) = 1;
            W.emplace_back(tmp_W);

            Matrix<Real> tmp_b(1, 1);
            tmp_b(0,0) = 0;
            b.emplace_back(tmp_b);
            
            this->layer->set_W(W);
            this->layer->set_b(b);

            Matrix<Real> tmp_x(2, 2);
            tmp_x(0, 0) = 1; tmp_x(0, 1) = 5;
            tmp_x(1, 0) = 2; tmp_x(1, 1) = 3;
            x = tmp_x;

            Matrix<Real> tmp_delta(2, 2);
            tmp_delta(0, 0) = -0.1; tmp_delta(0, 1) = 0.4;
            tmp_delta(1, 0) = -0.2; tmp_delta(1, 1) = 0.2;
            delta = tmp_delta;
        }

        virtual void TearDown () {
            
        }
    };

    TEST_F(BatchNormalizeTest, GPU_float_apply_without_function) {
        auto y = layer->apply(x, false);

        auto tmp_y = y.get_matrix();
        EXPECT_LT(fabs(tmp_y(0,0) - (1.0 - 3.0)/sqrt(4.0 + EPS)), 1.0E-4);
        EXPECT_LT(fabs(tmp_y(0,1) - (5.0 - 3.0)/sqrt(4.0 + EPS)), 1.0E-4);
        EXPECT_LT(fabs(tmp_y(1,0) - (2.0 - 2.5)/sqrt(0.25 + EPS)), 1.0E-4);
        EXPECT_LT(fabs(tmp_y(1,1) - (3.0 - 2.5)/sqrt(0.25 + EPS)), 1.0E-4);
    }

    TEST_F(BatchNormalizeTest, GPU_float_calc_delta) {
        auto y = layer->apply(x, false);

        clMatrix<Real> nx_delta(2, 2);

        layer->calc_delta(x, f->operator()(x, true), delta, nx_delta);

        double sigma[2] = {
            -0.1*1.0 * (1.0 - 3.0) + 0.4*1.0 * (5.0 - 3.0),
            -0.2*1.0 * (2.0 - 2.5) + 0.2*1.0 * (3.0 - 2.5)
        };
        double mu[2] = {
            -0.1*1.0 + 0.4*1.0,
            -0.2*1.0 + 0.2*1.0
        };

        double val[4] = {
            -0.1*1.0*1.0/sqrt(4.0 + EPS) - sigma[0]/2.0*(1.0 - 3.0)*pow((4.0 + EPS), -3.0/2.0) - mu[0]/2.0*1.0/sqrt(4.0 + EPS),
            0.4*1.0*1.0/sqrt(4.0 + EPS) - sigma[0]/2.0*(5.0 - 3.0)*pow((4.0 + EPS), -3.0/2.0) - mu[0]/2.0*1.0/sqrt(4.0 + EPS),
            -0.2*1.0*1.0/sqrt(0.25 + EPS) - sigma[1]/2.0*(2.0 - 2.5)*pow((0.25 + EPS), -3.0/2.0) - mu[1]/2.0*1.0/sqrt(0.25 + EPS),
            0.2*1.0*1.0/sqrt(0.25 + EPS) - sigma[1]/2.0*(3.0 - 2.5)*pow((0.25 + EPS), -3.0/2.0) - mu[1]/2.0*1.0/sqrt(0.25 + EPS)
        };

        auto tmp_nx_delta = nx_delta.get_matrix();
        EXPECT_LT(fabs(tmp_nx_delta(0,0) - val[0]), 1.0E-4);
        EXPECT_LT(fabs(tmp_nx_delta(0,1) - val[1]), 1.0E-4);
        EXPECT_LT(fabs(tmp_nx_delta(1,0) - val[2]), 1.0E-4);
        EXPECT_LT(fabs(tmp_nx_delta(1,1) - val[3]), 1.0E-4);
    }

    TEST_F(BatchNormalizeTest, GPU_float_calc_gradient) {
        auto y = layer->apply(x, false);

        std::vector<clMatrix<Real>> nabla_W, nabla_b;
        nabla_W.emplace_back(1, 1); nabla_b.emplace_back(1, 1);

        layer->calc_gradient(x, x, delta, nabla_W, nabla_b);

        const double val_nabla_W = -0.1 * (1.0 - 3.0)/sqrt(4.0 + EPS) + 0.4 * (5.0 - 3.0)/sqrt(4.0 + EPS) + -0.2 * (2.0 - 2.5)/sqrt(0.25 + EPS) + 0.2 * (3.0 - 2.5)/sqrt(0.25 + EPS);
        const double val_nabla_b = -0.1 + 0.4 + -0.2 + 0.2;
        EXPECT_LT(fabs(nabla_W[0].get_element(0,0) - val_nabla_W), 1.0E-4);
        EXPECT_LT(fabs(nabla_b[0].get_element(0,0) - val_nabla_b), 1.0E-4);
    }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
