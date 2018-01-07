#include <algorithm>
#include <vector>
#include <random>

#include <gtest/gtest.h>

#include <Matrix.hpp>
#include <clMatrix.hpp>
#include <Function.hpp>

namespace {
    typedef float Real;

    class SoftmaxTest : public ::testing::Test {
    protected:
        Softmax<Real> f;
        clMatrix<Real> x;
        
        virtual void SetUp () {
            auto tmp_x = Matrix<Real>(2, 2);
            tmp_x(0, 0) = 1; tmp_x(0, 1) = 1;
            tmp_x(1, 0) = 2; tmp_x(1, 1) = 3;
            x = tmp_x;
        }

        virtual void TearDown () {
            
        }
    };

    TEST_F(SoftmaxTest, CPU_float_apply_test) {
        auto y = f(x, false);

        const float denomi[2] = {
            std::exp(1.0f - 2.0f) + std::exp(2.0f - 2.0f),
            std::exp(1.0f - 3.0f) + std::exp(3.0f - 3.0f)
        };
        auto tmp_y = y.get_matrix();
        EXPECT_LT(fabs(tmp_y(0,0) - std::exp(1.0f - 2.0f)/denomi[0]), 1.0E-4);
        EXPECT_LT(fabs(tmp_y(1,0) - std::exp(2.0f - 2.0f)/denomi[0]), 1.0E-4);
        EXPECT_LT(fabs(tmp_y(0,1) - std::exp(1.0f - 3.0f)/denomi[1]), 1.0E-4);
        EXPECT_LT(fabs(tmp_y(1,1) - std::exp(3.0f - 3.0f)/denomi[1]), 1.0E-4);
    }

    TEST_F(SoftmaxTest, CPU_float_diff_test) {
        auto y = f(x, true);

        auto tmp_y = y.get_matrix();
        EXPECT_LT(fabs(tmp_y(0,0) - 1.0f), 1.0E-4);
        EXPECT_LT(fabs(tmp_y(1,0) - 1.0f), 1.0E-4);
        EXPECT_LT(fabs(tmp_y(0,1) - 1.0f), 1.0E-4);
        EXPECT_LT(fabs(tmp_y(1,1) - 1.0f), 1.0E-4);
    }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
