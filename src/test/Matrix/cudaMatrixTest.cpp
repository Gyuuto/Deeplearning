#include <algorithm>
#include <vector>

#include <gtest/gtest.h>
#include <Matrix.hpp>
#include <cudaMatrix.hpp>
#include <Layer/FullyConnected.hpp>

namespace {
    typedef float Real;

    class cudaMatrixTest : public ::testing::Test {
    protected:
        cudaMatrix<Real> A, B, C;
        
        virtual void SetUp () {
            Matrix<Real> tmp_A(2, 2);
            tmp_A(0, 0) = 1; tmp_A(0, 1) = 2;
            tmp_A(1, 0) = 3; tmp_A(1, 1) = 4;
            
            A = cudaMatrix<Real>(tmp_A);

            Matrix<Real> tmp_B(2, 2);
            tmp_B(0, 0) = 2; tmp_B(0, 1) = 2;
            tmp_B(1, 0) = 1; tmp_B(1, 1) = 5;
            
            B = cudaMatrix<Real>(tmp_B);

            Matrix<Real> tmp_C(2, 3);
            tmp_C(0, 0) = 2; tmp_C(0, 1) = 2; tmp_C(0, 2) = 1;
            tmp_C(1, 0) = 1; tmp_C(1, 1) = 5; tmp_C(1, 2) = 3;
            
            C = cudaMatrix<Real>(tmp_C);
        }

        virtual void TearDown () {
            
        }
    };

    TEST_F(cudaMatrixTest, get_matrix) {
        auto tmp_A = A.get_matrix();

        EXPECT_EQ(tmp_A(0,0), 1.0);
        EXPECT_EQ(tmp_A(0,1), 2.0);
        EXPECT_EQ(tmp_A(1,0), 3.0);
        EXPECT_EQ(tmp_A(1,1), 4.0);
    }
    
    TEST_F(cudaMatrixTest, eye_square_matrix) {
        auto tmp_A = cudaMatrix<Real>::eye(2, 2).get_matrix();

        EXPECT_EQ(tmp_A(0,0), 1.0);
        EXPECT_EQ(tmp_A(0,1), 0.0);
        EXPECT_EQ(tmp_A(1,0), 0.0);
        EXPECT_EQ(tmp_A(1,1), 1.0);
    }

    TEST_F(cudaMatrixTest, eye_rect_matrix) {
        auto tmp_A = cudaMatrix<Real>::eye(3, 4).get_matrix();

        EXPECT_EQ(tmp_A(0,0), 1.0);
        EXPECT_EQ(tmp_A(0,1), 0.0);
        EXPECT_EQ(tmp_A(0,2), 0.0);
        EXPECT_EQ(tmp_A(0,3), 0.0);
        EXPECT_EQ(tmp_A(1,0), 0.0);
        EXPECT_EQ(tmp_A(1,1), 1.0);
        EXPECT_EQ(tmp_A(1,2), 0.0);
        EXPECT_EQ(tmp_A(1,3), 0.0);
        EXPECT_EQ(tmp_A(2,0), 0.0);
        EXPECT_EQ(tmp_A(2,1), 0.0);
        EXPECT_EQ(tmp_A(2,2), 1.0);
        EXPECT_EQ(tmp_A(2,3), 0.0);
    }

    TEST_F(cudaMatrixTest, ones) {
        auto tmp_A = cudaMatrix<Real>::ones(3, 4).get_matrix();

        EXPECT_EQ(tmp_A(0,0), 1.0);
        EXPECT_EQ(tmp_A(0,1), 1.0);
        EXPECT_EQ(tmp_A(0,2), 1.0);
        EXPECT_EQ(tmp_A(0,3), 1.0);
        EXPECT_EQ(tmp_A(1,0), 1.0);
        EXPECT_EQ(tmp_A(1,1), 1.0);
        EXPECT_EQ(tmp_A(1,2), 1.0);
        EXPECT_EQ(tmp_A(1,3), 1.0);
        EXPECT_EQ(tmp_A(2,0), 1.0);
        EXPECT_EQ(tmp_A(2,1), 1.0);
        EXPECT_EQ(tmp_A(2,2), 1.0);
        EXPECT_EQ(tmp_A(2,3), 1.0);
    }

    TEST_F(cudaMatrixTest, zeros) {
        auto tmp_A = cudaMatrix<Real>::zeros(3, 4).get_matrix();

        EXPECT_EQ(tmp_A(0,0), 0.0);
        EXPECT_EQ(tmp_A(0,1), 0.0);
        EXPECT_EQ(tmp_A(0,2), 0.0);
        EXPECT_EQ(tmp_A(0,3), 0.0);
        EXPECT_EQ(tmp_A(1,0), 0.0);
        EXPECT_EQ(tmp_A(1,1), 0.0);
        EXPECT_EQ(tmp_A(1,2), 0.0);
        EXPECT_EQ(tmp_A(1,3), 0.0);
        EXPECT_EQ(tmp_A(2,0), 0.0);
        EXPECT_EQ(tmp_A(2,1), 0.0);
        EXPECT_EQ(tmp_A(2,2), 0.0);
        EXPECT_EQ(tmp_A(2,3), 0.0);
    }

    TEST_F(cudaMatrixTest, hadamard) {
        auto tmp = cudaMatrix<Real>::hadamard(A, B).get_matrix();

        EXPECT_EQ(tmp(0,0), 2.0);
        EXPECT_EQ(tmp(0,1), 4.0);
        EXPECT_EQ(tmp(1,0), 3.0);
        EXPECT_EQ(tmp(1,1), 20.0);
    }    

    TEST_F(cudaMatrixTest, norm_fro) {
        auto tmp = cudaMatrix<Real>::norm_fro(A);

        EXPECT_EQ(tmp, 30.0);
    }

    TEST_F(cudaMatrixTest, plus) {
        auto ans = A + B;
        auto tmp = ans.get_matrix();
        
        EXPECT_EQ(tmp(0,0), 3.0);
        EXPECT_EQ(tmp(0,1), 4.0);
        EXPECT_EQ(tmp(1,0), 4.0);
        EXPECT_EQ(tmp(1,1), 9.0);
    }

    TEST_F(cudaMatrixTest, minus) {
        auto ans = A - B;
        auto tmp = ans.get_matrix();
        
        EXPECT_EQ(tmp(0,0), -1.0);
        EXPECT_EQ(tmp(0,1), 0.0);
        EXPECT_EQ(tmp(1,0), 2.0);
        EXPECT_EQ(tmp(1,1), -1.0);
    }

    TEST_F(cudaMatrixTest, mult_normal) {
        auto ans = A * B;
        auto tmp = ans.get_matrix();
        
        EXPECT_EQ(tmp(0,0), 4.0);
        EXPECT_EQ(tmp(0,1), 12.0);
        EXPECT_EQ(tmp(1,0), 10.0);
        EXPECT_EQ(tmp(1,1), 26.0);
    }

    TEST_F(cudaMatrixTest, mult_trans_TN) {
        auto ans = cudaMatrix<Real>::transpose(A) * B;
        auto tmp = ans.get_matrix();
        
        EXPECT_EQ(tmp(0,0), 5.0);
        EXPECT_EQ(tmp(0,1), 17.0);
        EXPECT_EQ(tmp(1,0), 8.0);
        EXPECT_EQ(tmp(1,1), 24.0);
    }

    TEST_F(cudaMatrixTest, mult_trans_NT) {
        auto ans = A * cudaMatrix<Real>::transpose(B);
        auto tmp = ans.get_matrix();
        
        EXPECT_EQ(tmp(0,0), 6.0);
        EXPECT_EQ(tmp(0,1), 11.0);
        EXPECT_EQ(tmp(1,0), 14.0);
        EXPECT_EQ(tmp(1,1), 23.0);
    }

    TEST_F(cudaMatrixTest, mult_trans_TT) {
        auto ans = cudaMatrix<Real>::transpose(A) * cudaMatrix<Real>::transpose(B);
        auto tmp = ans.get_matrix();
        
        EXPECT_EQ(tmp(0,0), 8.0);
        EXPECT_EQ(tmp(0,1), 16.0);
        EXPECT_EQ(tmp(1,0), 12.0);
        EXPECT_EQ(tmp(1,1), 22.0);
    }

    TEST_F(cudaMatrixTest, mult_rect) {
        auto ans = A * C;
        auto tmp = ans.get_matrix();
        
        EXPECT_EQ(tmp(0,0), 4.0);
        EXPECT_EQ(tmp(0,1), 12.0);
        EXPECT_EQ(tmp(0,2), 7.0);
        EXPECT_EQ(tmp(1,0), 10.0);
        EXPECT_EQ(tmp(1,1), 26.0);
        EXPECT_EQ(tmp(1,2), 15.0);
    }    

    TEST_F(cudaMatrixTest, mult_rect_trans) {
        auto ans = cudaMatrix<Real>::transpose(C) * A;
        auto tmp = ans.get_matrix();
        
        EXPECT_EQ(tmp(0,0), 5.0);
        EXPECT_EQ(tmp(0,1), 8.0);
        EXPECT_EQ(tmp(1,0), 17.0);
        EXPECT_EQ(tmp(1,1), 24.0);
        EXPECT_EQ(tmp(2,0), 10.0);
        EXPECT_EQ(tmp(2,1), 14.0);
    }

    TEST_F(cudaMatrixTest, mult_inplace) {
        cudaMatrix<Real> D(2, 3);
        A.mult(1.0f, C, 0.0f, D);
        auto tmp = D.get_matrix();
        
        EXPECT_EQ(tmp(0,0), 4.0);
        EXPECT_EQ(tmp(0,1), 12.0);
        EXPECT_EQ(tmp(0,2), 7.0);
        EXPECT_EQ(tmp(1,0), 10.0);
        EXPECT_EQ(tmp(1,1), 26.0);
        EXPECT_EQ(tmp(1,2), 15.0);
    }

    TEST_F(cudaMatrixTest, sub) {
        auto ans = A.sub(0, 1, 2, 1);
        auto tmp = ans.get_matrix();
        
        EXPECT_EQ(tmp(0,0), 2.0);
        EXPECT_EQ(tmp(1,0), 4.0);
    }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
