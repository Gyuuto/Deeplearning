#include <algorithm>
#include <vector>
#include <random>

#include <gtest/gtest.h>
#include <Matrix.hpp>
#include <clMatrix.hpp>
#include <Function.hpp>

namespace {
    typedef float Real;

    class CrossEntropyTest : public ::testing::Test {
    protected:
        CrossEntropy<Real> f;
        clMatrix<Real> x, d;
        
        virtual void SetUp () {
            auto tmp_x = Matrix<Real>(2, 2);
            tmp_x(0, 0) = 1; tmp_x(0, 1) = 1;
            tmp_x(1, 0) = 2; tmp_x(1, 1) = 3;
            x = tmp_x;

            auto tmp_d = Matrix<Real>(2, 2);
            tmp_d(0, 0) = 1; tmp_d(0, 1) = 0;
            tmp_d(1, 0) = 0; tmp_d(1, 1) = 2;
            d = tmp_d;
        }

        virtual void TearDown () {
            
        }
    };

    TEST_F(CrossEntropyTest, OpenCL_float_apply_test) {
        auto y = f(x, d, false);

        const float ans = 2.0f*(
            -1.0*std::log(1.0f) - 0.0f*std::log(1.0f)
            - 0.0f*std::log(2.0f) - 2.0f*std::log(3.0f));
        auto tmp_y = y.get_matrix();
        EXPECT_LT(fabs(tmp_y(0,0) - ans), 1.0E-4);
    }

    TEST_F(CrossEntropyTest, OpenCL_float_diff_test) {
        auto y = f(x, d, true);

        auto tmp_y = y.get_matrix();
        EXPECT_LT(fabs(tmp_y(0,0) - 2.0f*(1.0f - 1.0f)), 1.0E-4);
        EXPECT_LT(fabs(tmp_y(1,0) - 2.0f*(2.0f - 0.0f)), 1.0E-4);
        EXPECT_LT(fabs(tmp_y(0,1) - 2.0f*(1.0f - 0.0f)), 1.0E-4);
        EXPECT_LT(fabs(tmp_y(1,1) - 2.0f*(3.0f - 2.0f)), 1.0E-4);
    }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
