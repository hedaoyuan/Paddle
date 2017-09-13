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

#pragma once

#include "paddle/framework/eigen.h"
#include "paddle/framework/op_registry.h"
#include "paddle/operators/math/im2col.h"
#include "paddle/operators/math/math_function.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename Place, typename T>
class GemmConvKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const Tensor* input = context.Input<Tensor>("Input");
    // The filter will be reshaped in the calculations,
    // so here use an assignment operation,
    // that avoids modifying the variable in the Scope.
    Tensor filter = *context.Input<Tensor>("Filter");
    Tensor* output = context.Output<Tensor>("Output");
    output->mutable_data<T>(context.GetPlace());

    std::vector<int> strides = context.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = context.Attr<std::vector<int>>("paddings");

    int batch_size = input->dims()[0];
    int input_channels = input->dims()[1];
    int filter_height = filter.dims()[filter.dims().size() - 2];
    int filter_width = filter.dims()[filter.dims().size() - 1];
    int output_channels = output->dims()[1];
    int output_height = output->dims()[2];
    int output_width = output->dims()[3];

    paddle::operators::math::Im2ColFunctor<
        paddle::operators::math::ColFormat::kCFO, Place, T>
        im2col;
    // use col_shape in the im2col calculation
    framework::DDim col_shape = {input_channels, filter_height, filter_width,
                                 output_height, output_width};
    // use col_matrix_shape in the gemm calculation
    framework::DDim col_matrix_shape = {
        input_channels * filter_height * filter_width,
        output_height * output_width};
    Tensor col;
    col.mutable_data<T>(col_shape, context.GetPlace());
    // col_matrix shares the same piece of data with col,
    // but will be reshaped into a two-dimensional matrix shape
    // to call the matrix multiplication interface.
    Tensor col_matrix = col;
    col_matrix.Resize(col_matrix_shape);

    framework::DDim input_shape = {input->dims()[1], input->dims()[2],
                                   input->dims()[3]};
    framework::DDim filter_matrix_shape = {filter.dims()[0],
                                           filter.numel() / filter.dims()[0]};
    filter.Resize(filter_matrix_shape);

    framework::DDim output_matrix_shape = {output_channels,
                                           output_height * output_width};

    auto* device_context =
        const_cast<platform::DeviceContext*>(context.device_context_);

    // convolution operator: im2col + gemm
    for (int i = 0; i < batch_size; i++) {
      // im2col
      Tensor in_slice = input->Slice<T>(i, i + 1).Resize(input_shape);
      im2col(in_slice, col, strides[0], strides[1], paddings[0], paddings[1],
             device_context);

      // gemm
      Tensor out_slice = output->Slice<T>(i, i + 1).Resize(output_matrix_shape);
      math::matmul<Place, T>(filter, false, col_matrix, false, T(1.0),
                             &out_slice, T(0.0), device_context);
    }
  }
};

template <typename Place, typename T>
class GemmConvGradKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const Tensor* input = context.Input<Tensor>("Input");
    const Tensor* output_grad =
        context.Input<Tensor>(framework::GradVarName("Output"));
    Tensor* input_grad =
        context.Output<Tensor>(framework::GradVarName("Input"));
    Tensor* filter_grad_ =
        context.Output<Tensor>(framework::GradVarName("Filter"));
    input_grad->mutable_data<T>(context.GetPlace());
    filter_grad_->mutable_data<T>(context.GetPlace());

    // The filter and filter_grad will be reshaped in the calculations,
    // so here use an assignment operation,
    // that avoids modifying the variable in the Scope.
    Tensor filter = *context.Input<Tensor>("Filter");
    Tensor filter_grad = *filter_grad_;

    std::vector<int> strides = context.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = context.Attr<std::vector<int>>("paddings");

    int batch_size = input->dims()[0];
    int input_channels = input->dims()[1];
    int filter_height = filter.dims()[filter.dims().size() - 2];
    int filter_width = filter.dims()[filter.dims().size() - 1];
    int output_height = output_grad->dims()[2];
    int output_width = output_grad->dims()[3];

    paddle::operators::math::Col2ImFunctor<
        paddle::operators::math::ColFormat::kCFO, Place, T>
        col2im;
    paddle::operators::math::Im2ColFunctor<
        paddle::operators::math::ColFormat::kCFO, Place, T>
        im2col;
    // use col_shape in the im2col and col2im calculation
    framework::DDim col_shape = {input_channels, filter_height, filter_width,
                                 output_height, output_width};
    // use col_matrix_shape in the gemm calculation
    framework::DDim col_matrix_shape = {
        input_channels * filter_height * filter_width,
        output_height * output_width};
    Tensor col;
    col.mutable_data<T>(col_shape, context.GetPlace());
    // col_matrix shares the same piece of data with col,
    // but will be reshaped into a two-dimensional matrix shape
    // to call the matrix multiplication interface.
    Tensor col_matrix = col;
    col_matrix.Resize(col_matrix_shape);

    framework::DDim input_shape = {input->dims()[1], input->dims()[2],
                                   input->dims()[3]};
    framework::DDim output_matrix_shape = {
        output_grad->dims()[1],
        output_grad->dims()[2] * output_grad->dims()[3]};

    framework::DDim filter_matrix_shape = {filter.dims()[0],
                                           filter.numel() / filter.dims()[0]};
    filter.Resize(filter_matrix_shape);
    filter_grad.Resize(filter_matrix_shape);

    auto t1 = framework::EigenVector<T>::Flatten(filter_grad);
    t1.device(context.GetEigenDevice<Place>()) = t1.constant(static_cast<T>(0));
    auto t2 = framework::EigenVector<T>::Flatten(*input_grad);
    t2.device(context.GetEigenDevice<Place>()) = t2.constant(static_cast<T>(0));

    auto* device_context =
        const_cast<platform::DeviceContext*>(context.device_context_);

    // convolution backward input operator:  gemm + col2im
    // convolution backward weight operator: im2col + gemm
    for (int i = 0; i < batch_size; i++) {
      // gemm
      Tensor out_slice =
          output_grad->Slice<T>(i, i + 1).Resize(output_matrix_shape);
      math::matmul<Place, T>(filter, true, out_slice, false, T(1.0),
                             &col_matrix, T(0.0), device_context);

      // col2im
      Tensor in_grad_slice = input_grad->Slice<T>(i, i + 1).Resize(input_shape);
      col2im(in_grad_slice, col, strides[0], strides[1], paddings[0],
             paddings[1], device_context);

      // im2col
      Tensor in_slice = input->Slice<T>(i, i + 1).Resize(input_shape);
      im2col(in_slice, col, strides[0], strides[1], paddings[0], paddings[1],
             device_context);

      // gemm
      math::matmul<Place, T>(out_slice, false, col_matrix, true, T(1.0),
                             &filter_grad, T(1.0), device_context);
    }
  }
};

}  // namespace operators
}  // namespace paddle