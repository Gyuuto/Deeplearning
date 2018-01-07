#include <algorithm>
#include <vector>
#include <random>

#include <gtest/gtest.h>
#include <Matrix.hpp>
#include <Function.hpp>

namespace {
    typedef float Real;

    class CrossEntropyTest : public ::testing::Test {
    protected:
        CrossEntropy<Real> f;
        Matrix<Real> x, d;
        
        virtual void SetUp () {
            x = Matrix<Real>(2, 2);
            x(0, 0) = 1; x(0, 1) = 1;
            x(1, 0) = 2; x(1, 1) = 3;

            d = Matrix<Real>(2, 2);
            d(0, 0) = 1; d(0, 1) = 0;
            d(1, 0) = 0; d(1, 1) = 2;
        }

        virtual void TearDown () {
            
        }
    };

    TEST_F(CrossEntropyTest, CPU_float_apply_test) {
        auto y = f(x, d, false);

        const float ans = 2.0f*(
            -1.0*std::log(1.0f) - 0.0f*std::log(1.0f)
            - 0.0f*std::log(2.0f) - 2.0f*std::log(3.0f));
        EXPECT_LT(fabs(y(0,0) - ans), 1.0E-4);
    }

    TEST_F(CrossEntropyTest, CPU_float_diff_test) {
        auto y = f(x, d, true);

        EXPECT_LT(fabs(y(0,0) - 2.0f*(1.0f - 1.0f)), 1.0E-4);
        EXPECT_LT(fabs(y(1,0) - 2.0f*(2.0f - 0.0f)), 1.0E-4);
        EXPECT_LT(fabs(y(0,1) - 2.0f*(1.0f - 0.0f)), 1.0E-4);
        EXPECT_LT(fabs(y(1,1) - 2.0f*(3.0f - 2.0f)), 1.0E-4);
    }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
