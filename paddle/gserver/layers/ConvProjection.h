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

#include "ConvBaseProjection.h"
#include "paddle/math/MathUtils.h"

namespace paddle {

/**
 * @brief Convolution projection do the same calculation with CudnnConvLayer.
 */
class ConvProjection : public ConvBaseProjection {
public:
  /**
   * Constructor.
   */
  ConvProjection(const ProjectionConfig& config,
                 ParameterPtr parameter,
                 bool useGpu)
      : ConvBaseProjection(config, parameter, useGpu) {}

  ~ConvProjection() {}

  virtual void forward();
  virtual void backward(const UpdateCallback& callback);
  virtual size_t calOutputSize();
  virtual size_t calInputSize();
};

}  // namespace paddle
