#include <algorithm>
#include <vector>
#include <random>

#include <gtest/gtest.h>
#include <Matrix.hpp>
#include <Optimizer/ADAM.hpp>

namespace {
    typedef float Real;

    class ADAMTest : public ::testing::Test {
    protected:
        const Real EPS = 1.0E0;
        std::vector<Matrix<Real>> nabla_W, nabla_b;
        
        virtual void SetUp () {
            nabla_W.emplace_back(Matrix<Real>(2, 3));
            nabla_W[0](0, 0) = 1.0f; nabla_W[0](0, 1) = -2.0f; nabla_W[0](0, 2) = -1.0f; 
            nabla_W[0](1, 0) = 0.0f; nabla_W[0](1, 1) = 1.0f; nabla_W[0](1, 2) = 3.0f; 

            nabla_b.emplace_back(Matrix<Real>(2, 1));
            nabla_b[0](0, 0) = 1.0f;
            nabla_b[0](1, 0) = -2.0f;
        }

        virtual void TearDown () {
            
        }
    };

    TEST_F(ADAMTest, CPU_float_update_test) {
        std::vector<Matrix<Real>> update_W, update_b;
        update_W.emplace_back(Matrix<Real>(nabla_W[0].m, nabla_W[0].n));
        update_b.emplace_back(Matrix<Real>(nabla_b[0].m, nabla_b[0].n));

        ADAM<Matrix, Real> adam(nabla_W, nabla_b, EPS);
        adam.update(nabla_W, nabla_b, update_W, update_b);
        
        EXPECT_LT(fabs(update_W[0](0, 0) - (-1.0f)), 1.0E-4);
        EXPECT_LT(fabs(update_W[0](0, 1) - 1.0f), 1.0E-4);
        EXPECT_LT(fabs(update_W[0](0, 2) - 1.0f), 1.0E-4);
        EXPECT_LT(fabs(update_W[0](1, 0) - 0.0f), 1.0E-4);
        EXPECT_LT(fabs(update_W[0](1, 1) - (-1.0f)), 1.0E-4);
        EXPECT_LT(fabs(update_W[0](1, 2) - (-1.0f)), 1.0E-4);

        EXPECT_LT(fabs(update_b[0](0, 0) - (-1.0f)), 1.0E-4);
        EXPECT_LT(fabs(update_b[0](1, 0) - 1.0f), 1.0E-4);
    }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
