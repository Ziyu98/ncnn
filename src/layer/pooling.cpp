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

#include "pooling.h"

#include "layer_type.h"

#include <float.h>

namespace ncnn {

Pooling::Pooling()
{
    one_blob_only = true;
    support_inplace = false;
}

int Pooling::load_param(const ParamDict& pd)
{
    pooling_type = pd.get(0, 0);
    kernel_w = pd.get(1, 0);
    kernel_h = pd.get(11, kernel_w);
    stride_w = pd.get(2, 1);
    stride_h = pd.get(12, stride_w);
    pad_left = pd.get(3, 0);
    pad_right = pd.get(14, pad_left);
    pad_top = pd.get(13, pad_left);
    pad_bottom = pd.get(15, pad_top);
    global_pooling = pd.get(4, 0);
    pad_mode = pd.get(5, 0);
    avgpool_count_include_pad = pd.get(6, 0);

    return 0;
}

int Pooling::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    // max value in NxN window
    // avg value in NxN window

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;

    //     NCNN_LOGE("Pooling     input %d x %d  pad = %d %d %d %d  ksize=%d %d  stride=%d %d", w, h, pad_left, pad_right, pad_top, pad_bottom, kernel_w, kernel_h, stride_w, stride_h);
    if (global_pooling)
    {
        top_blob.create(channels, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        int size = w * h;

        if (pooling_type == PoolMethod_MAX)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);

                float max = ptr[0];
                for (int i = 0; i < size; i++)
                {
                    max = std::max(max, ptr[i]);
                }

                top_blob[q] = max;
            }
        }
        else if (pooling_type == PoolMethod_AVE)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);

                float sum = 0.f;
                for (int i = 0; i < size; i++)
                {
                    sum += ptr[i];
                }

                top_blob[q] = sum / size;
            }
        }

        return 0;
    }

    Mat bottom_blob_bordered;
    make_padding(bottom_blob, bottom_blob_bordered, opt);
    if (bottom_blob_bordered.empty())
        return -100;

    w = bottom_blob_bordered.w;
    h = bottom_blob_bordered.h;

    int outw = (w - kernel_w) / stride_w + 1;
    int outh = (h - kernel_h) / stride_h + 1;

    top_blob.create(outw, outh, channels, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    const int maxk = kernel_w * kernel_h;

    // kernel offsets
    std::vector<int> _space_ofs(maxk);
    int* space_ofs = &_space_ofs[0];
    {
        int p1 = 0;
        int p2 = 0;
        int gap = w - kernel_w;
        for (int i = 0; i < kernel_h; i++)
        {
            for (int j = 0; j < kernel_w; j++)
            {
                space_ofs[p1] = p2;
                p1++;
                p2++;
            }
            p2 += gap;
        }
    }

    if (pooling_type == PoolMethod_MAX)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const Mat m = bottom_blob_bordered.channel(q);
            float* outptr = top_blob.channel(q);

            for (int i = 0; i < outh; i++)
            {
                for (int j = 0; j < outw; j++)
                {
                    const float* sptr = m.row(i * stride_h) + j * stride_w;

                    float max = sptr[0];

                    for (int k = 0; k < maxk; k++)
                    {
                        float val = sptr[space_ofs[k]];
                        max = std::max(max, val);
                    }

                    outptr[j] = max;
                }

                outptr += outw;
            }
        }
    }
    else if (pooling_type == PoolMethod_AVE)
    {
        if (avgpool_count_include_pad == 0)
        {
            int wtailpad = 0;
            int htailpad = 0;

            if (pad_mode == 0) // full padding
            {
                wtailpad = bottom_blob_bordered.w - bottom_blob.w - pad_left - pad_right;
                htailpad = bottom_blob_bordered.h - bottom_blob.h - pad_top - pad_bottom;
            }

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const Mat m = bottom_blob_bordered.channel(q);
                float* outptr = top_blob.channel(q);

                for (int i = 0; i < outh; i++)
                {
                    int sy0 = i * stride_h;

                    for (int j = 0; j < outw; j++)
                    {
                        int sx0 = j * stride_w;

                        float sum = 0;
                        int area = 0;

                        for (int ki = 0; ki < kernel_h; ki++)
                        {
                            int sy = sy0 + ki;

                            if (sy < pad_top)
                                continue;

                            if (sy >= h - pad_bottom - htailpad)
                                break;

                            for (int kj = 0; kj < kernel_w; kj++)
                            {
                                int sx = sx0 + kj;

                                if (sx < pad_left)
                                    continue;

                                if (sx >= w - pad_right - wtailpad)
                                    break;

                                float val = m.row(sy)[sx];
                                sum += val;
                                area += 1;
                            }
                        }

                        outptr[j] = sum / area;
                    }

                    outptr += outw;
                }
            }
        }
        else // if (avgpool_count_include_pad == 1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const Mat m = bottom_blob_bordered.channel(q);
                float* outptr = top_blob.channel(q);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        const float* sptr = m.row(i * stride_h) + j * stride_w;

                        float sum = 0;

                        for (int k = 0; k < maxk; k++)
                        {
                            float val = sptr[space_ofs[k]];
                            sum += val;
                        }

                        outptr[j] = sum / maxk;
                    }

                    outptr += outw;
                }
            }
        }
    }

    return 0;
}

void Pooling::make_padding(const Mat& bottom_blob, Mat& bottom_blob_bordered, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;

    bottom_blob_bordered = bottom_blob;

    float pad_value = 0.f;
    if (pooling_type == PoolMethod_MAX)
    {
        pad_value = bottom_blob.elemsize == 1 ? -128.f : -FLT_MAX;
    }
    else if (pooling_type == PoolMethod_AVE)
    {
        pad_value = 0.f;
    }

    int wtailpad = 0;
    int htailpad = 0;

    if (pad_mode == 0) // full padding
    {
        int wtail = (w + pad_left + pad_right - kernel_w) % stride_w;
        int htail = (h + pad_top + pad_bottom - kernel_h) % stride_h;

        if (wtail != 0)
            wtailpad = stride_w - wtail;
        if (htail != 0)
            htailpad = stride_h - htail;

        Option opt_b = opt;
        opt_b.blob_allocator = opt.workspace_allocator;
        copy_make_border(bottom_blob, bottom_blob_bordered, pad_top, pad_bottom + htailpad, pad_left, pad_right + wtailpad, BORDER_CONSTANT, pad_value, opt_b);
    }
    else if (pad_mode == 1) // valid padding
    {
        Option opt_b = opt;
        opt_b.blob_allocator = opt.workspace_allocator;
        copy_make_border(bottom_blob, bottom_blob_bordered, pad_top, pad_bottom, pad_left, pad_right, BORDER_CONSTANT, pad_value, opt_b);
    }
    else if (pad_mode == 2) // tensorflow padding=SAME or onnx padding=SAME_UPPER
    {
        int wpad = kernel_w + (w - 1) / stride_w * stride_w - w;
        int hpad = kernel_h + (h - 1) / stride_h * stride_h - h;
        if (wpad > 0 || hpad > 0)
        {
            Option opt_b = opt;
            opt_b.blob_allocator = opt.workspace_allocator;
            copy_make_border(bottom_blob, bottom_blob_bordered, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, BORDER_CONSTANT, pad_value, opt_b);
        }
    }
    else if (pad_mode == 3) // onnx padding=SAME_LOWER
    {
        int wpad = kernel_w + (w - 1) / stride_w * stride_w - w;
        int hpad = kernel_h + (h - 1) / stride_h * stride_h - h;
        if (wpad > 0 || hpad > 0)
        {
            Option opt_b = opt;
            opt_b.blob_allocator = opt.workspace_allocator;
            copy_make_border(bottom_blob, bottom_blob_bordered, hpad - hpad / 2, hpad / 2, wpad - wpad / 2, wpad / 2, BORDER_CONSTANT, pad_value, opt_b);
        }
    }
}

#if NCNN_CNNCACHE
bool Pooling::needs_cache() const {return false;}
int Pooling::forward_roi(MRect& bottom_padroi, MRect& top_roi, MRect& top_padroi) const
{
    top_roi.forward_in_conv_or_pool(bottom_padroi, pad_left, kernel_w, stride_w);
    top_padroi.pad_in_conv_or_pool(top_roi, pad_left, kernel_w);
    if (top_padroi.size() > 1) {
        size_t maxx = 0;
        size_t max = top_padroi.changed_vecs.size();
        while (maxx != max) {
            max = top_padroi.changed_vecs.size();
            maxx = top_padroi.changed_vecs.size();
            for (size_t i = 0; i < maxx; i++) {
                for (size_t j = i + 1; j < maxx; j++) {
                    const struct rect temp1 = top_padroi.changed_vecs[i];
                    const struct rect temp2 = top_padroi.changed_vecs[j];
                    if ((temp2.x1 >= temp1.x1 && temp2.y1 >= temp1.y1 && temp2.x1 <= temp1.x2 && temp2.y1 <= temp1.y2) 
                        || (temp1.x1 >= temp2.x1 && temp1.y1 >= temp2.y1 && temp1.x1 <= temp2.x2 && temp1.y1 <= temp2.y2)) {
                        // intersected
                        int x1 = std::min(temp1.x1, temp2.x1);
                        int y1 = std::min(temp1.y1, temp2.y1);
                        int x2 = std::max(temp1.x2, temp2.x2);
                        int y2 = std::max(temp1.y2, temp2.y2);
                        auto begin = top_padroi.changed_vecs.begin();
                        // ???? wrong!!!!
                        //top_padroi.changed_vecs.erase(begin + i);
                        top_padroi.changed_vecs[i] = rect(x1, y1, x2, y2);
                        top_padroi.changed_vecs.erase(begin + j);
                        
                        j--;
                        maxx--;
                    }

                }
            }
        }
    }
    //top_padroi.pad_in_conv_or_pool(top_roi, pad_left, kernel_w);
    //NCNN_LOGE("forward_roi pad = %d %d  ksize=%d %d  stride=%d %d", pad_left, pad_top, kernel_w, kernel_h, stride_w, stride_h);
    //NCNN_LOGE("IN POOLING LAYER, pad = %d %d  ksize=%d %d  stride=%d %d", pad_left, pad_top, kernel_w, kernel_h, stride_w, stride_h);
    //NCNN_LOGE("ROI IS: %d, %d, %d, %d", top_roi.changed_vecs[0].x1, top_roi.changed_vecs[0].y1, top_roi.changed_vecs[0].x2, top_roi.changed_vecs[0].y2);
    //NCNN_LOGE("PAD ROI IS: %d, %d, %d, %d", top_padroi.changed_vecs[0].x1, top_padroi.changed_vecs[0].y1, top_padroi.changed_vecs[0].x2, top_padroi.changed_vecs[0].y2);
    /*NCNN_LOGE("in pooling, top_roi info: ");
    top_roi.info();
    NCNN_LOGE("in pooling, top_padroi info: ");
    top_padroi.info();*/
    return 0;
}

#endif

} // namespace ncnn
