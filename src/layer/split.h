// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#ifndef LAYER_SPLIT_H
#define LAYER_SPLIT_H

#include "layer.h"

namespace ncnn {

class Split : public Layer
{
public:
    Split();

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

#if NCNN_VULKAN
    virtual int forward(const std::vector<VkMat>& bottom_blobs, std::vector<VkMat>& top_blobs, VkCompute& cmd, const Option& opt) const;
    virtual int forward(const std::vector<VkImageMat>& bottom_blobs, std::vector<VkImageMat>& top_blobs, VkCompute& cmd, const Option& opt) const;
#endif // NCNN_VULKAN
    
#if NCNN_CNNCACHE
    virtual int forward_roi(std::vector<MRect>& bottom_padroi, std::vector<MRect>& top_roi, std::vector<MRect>& top_padroi) const;
    //virtual int forward_cached(const Mat& bottom_blob, Mat& top_blob, const Option& opt, MRect& bottom_padroi, MRect& top_roi, MRect& top_padroi, Mat& cached_blob) const;
    virtual bool needs_cache() const;
#endif
public:
};

} // namespace ncnn

#endif // LAYER_SPLIT_H
