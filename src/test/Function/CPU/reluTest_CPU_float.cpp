#include <algorithm>
#include <vector>
#include <random>

#include <gtest/gtest.h>
#include <Matrix.hpp>
#include <Function.hpp>

namespace {
    typedef float Real;

    class ReLUTest : public ::testing::Test {
    protected:
        ReLU<Real> f;
        Matrix<Real> x;
        
        virtual void SetUp () {
            x = Matrix<Real>(2, 2);
            x(0, 0) = 1; x(0, 1) = -1;
            x(1, 0) = -2; x(1, 1) = 3;
        }

        virtual void TearDown () {
            
        }
    };

    TEST_F(ReLUTest, CPU_float_apply_test) {
        auto y = f(x, false);

        EXPECT_LT(fabs(y(0,0) - 1.0f), 1.0E-4);
        EXPECT_LT(fabs(y(1,0) - 0.0f), 1.0E-4);
        EXPECT_LT(fabs(y(0,1) - 0.0f), 1.0E-4);
        EXPECT_LT(fabs(y(1,1) - 3.0f), 1.0E-4);
    }

    TEST_F(ReLUTest, CPU_float_diff_test) {
        auto y = f(x, true);

        EXPECT_LT(fabs(y(0,0) - 1.0f), 1.0E-4);
        EXPECT_LT(fabs(y(1,0) - 0.0f), 1.0E-4);
        EXPECT_LT(fabs(y(0,1) - 0.0f), 1.0E-4);
        EXPECT_LT(fabs(y(1,1) - 1.0f), 1.0E-4);
    }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
