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

#ifndef NCNN_LAYER_H
#define NCNN_LAYER_H

#include "mat.h"
#include "modelbin.h"
#include "option.h"
#include "paramdict.h"
#include "platform.h"
#include "mrect.h"
#include <math.h>

#if NCNN_VULKAN
#include "command.h"
#include "pipeline.h"

#include <vulkan/vulkan.h>
#endif // NCNN_VULKAN

namespace ncnn {

class Layer
{
public:
    // empty
    Layer();
    // virtual destructor
    virtual ~Layer();

    // load layer specific parameter from parsed dict
    // return 0 if success
    virtual int load_param(const ParamDict& pd);

    // load layer specific weight data from model binary
    // return 0 if success
    virtual int load_model(const ModelBin& mb);

    // layer implementation specific setup
    // return 0 if success
    virtual int create_pipeline(const Option& opt);

    // layer implementation specific clean
    // return 0 if success
    virtual int destroy_pipeline(const Option& opt);

public:
    // one input and one output blob
    bool one_blob_only;

    // support inplace inference
    bool support_inplace;

    // support vulkan compute
    bool support_vulkan;

    // accept input blob with packed storage
    bool support_packing;

    // accept bf16
    bool support_bf16_storage;

    // accept fp16
    bool support_fp16_storage;

    // shader image storage
    bool support_image_storage;

    // TODO drop these fields
    bool use_int8_inference;
    bool support_weight_fp16_storage;

public:
    // implement inference
    // return 0 if success
    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

    // implement inplace inference
    // return 0 if success
    virtual int forward_inplace(std::vector<Mat>& bottom_top_blobs, const Option& opt) const;
    virtual int forward_inplace(Mat& bottom_top_blob, const Option& opt) const;
#if NCNN_CNNCACHE
    virtual int forward_roi(std::vector<MRect>& bottom_padrois, std::vector<MRect>& top_rois, std::vector<MRect>& top_padrois) const;
    virtual int forward_roi(MRect& bottom_padroi, MRect& top_roi, MRect& top_padroi) const;
    virtual int forward_cached(const Mat& bottom_blob, Mat& top_blob, const Option& opt, MRect& bottom_padroi, MRect& top_roi, MRect& top_padroi, Mat& cached_blob, std::vector<Mat>& temp_top) const;
    virtual bool needs_cache() const {return false;}
#endif


#if NCNN_VULKAN
public:
    // upload weight blob from host to device
    virtual int upload_model(VkTransfer& cmd, const Option& opt);

public:
    // implement inference
    // return 0 if success
    virtual int forward(const std::vector<VkMat>& bottom_blobs, std::vector<VkMat>& top_blobs, VkCompute& cmd, const Option& opt) const;
    virtual int forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const;

    // implement inference
    // return 0 if success
    virtual int forward(const std::vector<VkImageMat>& bottom_blobs, std::vector<VkImageMat>& top_blobs, VkCompute& cmd, const Option& opt) const;
    virtual int forward(const VkImageMat& bottom_blob, VkImageMat& top_blob, VkCompute& cmd, const Option& opt) const;

    // implement inplace inference
    // return 0 if success
    virtual int forward_inplace(std::vector<VkMat>& bottom_top_blobs, VkCompute& cmd, const Option& opt) const;
    virtual int forward_inplace(VkMat& bottom_top_blob, VkCompute& cmd, const Option& opt) const;

    // implement inplace inference
    // return 0 if success
    virtual int forward_inplace(std::vector<VkImageMat>& bottom_top_blobs, VkCompute& cmd, const Option& opt) const;
    virtual int forward_inplace(VkImageMat& bottom_top_blob, VkCompute& cmd, const Option& opt) const;

public:
    // assigned immediately after creating this layer
    const VulkanDevice* vkdev;
#endif // NCNN_VULKAN

public:
    // layer type index
    int typeindex;
#if NCNN_STRING
    // layer type name
    std::string type;
    // layer name
    std::string name;
#endif // NCNN_STRING
    // blob index which this layer needs as input
    std::vector<int> bottoms;
    // blob index which this layer produces as output
    std::vector<int> tops;
    // shape hint
    std::vector<Mat> bottom_shapes;
    std::vector<Mat> top_shapes;
};

// layer factory function
typedef Layer* (*layer_creator_func)();

struct layer_registry_entry
{
#if NCNN_STRING
    // layer type name
    const char* name;
#endif // NCNN_STRING
    // layer factory entry
    layer_creator_func creator;
};

#if NCNN_STRING
// get layer type from type name
int layer_to_index(const char* type);
// create layer from type name
Layer* create_layer(const char* type);
#endif // NCNN_STRING
// create layer from layer type
Layer* create_layer(int index);

#define DEFINE_LAYER_CREATOR(name)        \
    ::ncnn::Layer* name##_layer_creator() \
    {                                     \
        return new name;                  \
    }

} // namespace ncnn

#endif // NCNN_LAYER_H
