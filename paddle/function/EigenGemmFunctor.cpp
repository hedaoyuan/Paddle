/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "unsupported/Eigen/CXX11/Tensor"

namespace paddle {
typedef Eigen::TensorMap<Eigen::Tensor<float, 2, Eigen::RowMajor, int>,
                         Eigen::Aligned>
    Matrix;

template <class T>
void eigen_gemm(const int M,
                const int N,
                const int K,
                const T alpha,
                const T* A,
                const int lda,
                const T* B,
                const int ldb,
                const T beta,
                T* C,
                const int ldc) {
  Eigen::array<int, 2> sizeA;
  sizeA[0] = M;
  sizeA[1] = K;
  Eigen::array<int, 2> sizeB;
  sizeB[0] = K;
  sizeB[1] = N;
  Eigen::array<int, 2> sizeC;
  sizeC[0] = M;
  sizeC[1] = N;

  const Matrix EigenA(const_cast<float*>(A), sizeA);
  const Matrix EigenB(const_cast<float*>(B), sizeB);
  Matrix EigenC(C, sizeC);

  Eigen::DefaultDevice device;
  typedef typename Eigen::Tensor<T, 2>::DimensionPair DimPair;
  Eigen::array<DimPair, 1> dims;
  dims[0] = DimPair(1, 0);
  EigenC.device(device) = EigenA.contract(EigenB, dims);
}

template void eigen_gemm<float>(const int M,
                                const int N,
                                const int K,
                                const float alpha,
                                const float* A,
                                const int lda,
                                const float* B,
                                const int ldb,
                                const float beta,
                                float* C,
                                const int ldc);

}  // namespace paddle
