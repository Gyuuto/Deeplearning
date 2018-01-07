#include <algorithm>
#include <vector>
#include <random>

#include <gtest/gtest.h>
#include <Matrix.hpp>
#include <cudaMatrix.hpp>
#include <Optimizer/ADAM.hpp>

namespace {
    typedef float Real;

    class ADAMTest : public ::testing::Test {
    protected:
        const Real EPS = 1.0E0;
        std::vector<cudaMatrix<Real>> nabla_W, nabla_b;
        
        virtual void SetUp () {
            Matrix<Real> tmp_nabla_W(2, 3);
            tmp_nabla_W(0, 0) = 1.0f; tmp_nabla_W(0, 1) = -2.0f; tmp_nabla_W(0, 2) = -1.0f; 
            tmp_nabla_W(1, 0) = 0.0f; tmp_nabla_W(1, 1) = 1.0f; tmp_nabla_W(1, 2) = 3.0f; 
            nabla_W.emplace_back(tmp_nabla_W);

            Matrix<Real> tmp_nabla_b(2, 1);
            tmp_nabla_b(0, 0) = 1.0f;
            tmp_nabla_b(1, 0) = -2.0f;
            nabla_b.emplace_back(tmp_nabla_b);
        }

        virtual void TearDown () {
            
        }
    };

    TEST_F(ADAMTest, CUDA_float_update_test) {
        std::vector<cudaMatrix<Real>> update_W, update_b;
        update_W.emplace_back(cudaMatrix<Real>(nabla_W[0].m, nabla_W[0].n));
        update_b.emplace_back(cudaMatrix<Real>(nabla_b[0].m, nabla_b[0].n));

        ADAM<cudaMatrix, Real> adam(nabla_W, nabla_b, EPS);
        adam.update(nabla_W, nabla_b, update_W, update_b);

        auto tmp_W = update_W[0].get_matrix();
        EXPECT_LT(fabs(tmp_W(0, 0) - (-1.0f)), 1.0E-4);
        EXPECT_LT(fabs(tmp_W(0, 1) - 1.0f), 1.0E-4);
        EXPECT_LT(fabs(tmp_W(0, 2) - 1.0f), 1.0E-4);
        EXPECT_LT(fabs(tmp_W(1, 0) - 0.0f), 1.0E-4);
        EXPECT_LT(fabs(tmp_W(1, 1) - (-1.0f)), 1.0E-4);
        EXPECT_LT(fabs(tmp_W(1, 2) - (-1.0f)), 1.0E-4);

        auto tmp_b = update_b[0].get_matrix();
        EXPECT_LT(fabs(tmp_b(0, 0) - (-1.0f)), 1.0E-4);
        EXPECT_LT(fabs(tmp_b(1, 0) - 1.0f), 1.0E-4);
    }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
