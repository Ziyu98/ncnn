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

#include "concat.h"

namespace ncnn {

Concat::Concat()
{
    one_blob_only = false;
    support_inplace = false;
}

int Concat::load_param(const ParamDict& pd)
{
    axis = pd.get(0, 0);

    return 0;
}

int Concat::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    int dims = bottom_blobs[0].dims;
    size_t elemsize = bottom_blobs[0].elemsize;

    if (dims == 1) // axis == 0
    {
        // concat vector
        // total length
        int top_w = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob = bottom_blobs[b];
            top_w += bottom_blob.w;
        }

        Mat& top_blob = top_blobs[0];
        top_blob.create(top_w, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        unsigned char* outptr = top_blob;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob = bottom_blobs[b];

            int w = bottom_blob.w;

            const unsigned char* ptr = bottom_blob;
            memcpy(outptr, ptr, w * elemsize);

            outptr += w * elemsize;
        }

        return 0;
    }

    if (dims == 2 && axis == 0)
    {
        // concat image
        int w = bottom_blobs[0].w;

        // total height
        int top_h = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob = bottom_blobs[b];
            top_h += bottom_blob.h;
        }

        Mat& top_blob = top_blobs[0];
        top_blob.create(w, top_h, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        unsigned char* outptr = top_blob;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob = bottom_blobs[b];

            int size = w * bottom_blob.h;

            const unsigned char* ptr = bottom_blob;
            memcpy(outptr, ptr, size * elemsize);

            outptr += size * elemsize;
        }

        return 0;
    }

    if (dims == 2 && axis == 1)
    {
        // interleave image row
        int h = bottom_blobs[0].h;

        // total width
        int top_w = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob = bottom_blobs[b];
            top_w += bottom_blob.w;
        }

        Mat& top_blob = top_blobs[0];
        top_blob.create(top_w, h, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            unsigned char* outptr = top_blob.row<unsigned char>(i);
            for (size_t b = 0; b < bottom_blobs.size(); b++)
            {
                const Mat& bottom_blob = bottom_blobs[b];

                const unsigned char* ptr = bottom_blob.row<const unsigned char>(i);
                memcpy(outptr, ptr, bottom_blob.w * elemsize);

                outptr += bottom_blob.w * elemsize;
            }
        }

        return 0;
    }

    if (dims == 3 && axis == 0)
    {
        // concat dim
        int w = bottom_blobs[0].w;
        int h = bottom_blobs[0].h;

        // total channels
        int top_channels = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob = bottom_blobs[b];
            top_channels += bottom_blob.c;
        }

        Mat& top_blob = top_blobs[0];
        top_blob.create(w, h, top_channels, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        int q = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob = bottom_blobs[b];

            int channels = bottom_blob.c;
            size_t size = bottom_blob.cstep * channels;

            const unsigned char* ptr = bottom_blob;
            unsigned char* outptr = top_blob.channel(q);
            memcpy(outptr, ptr, size * elemsize);

            q += channels;
        }

        return 0;
    }

    if (dims == 3 && axis == 1)
    {
        // interleave dim height
        int w = bottom_blobs[0].w;
        int channels = bottom_blobs[0].c;

        // total height
        int top_h = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob = bottom_blobs[b];
            top_h += bottom_blob.h;
        }

        Mat& top_blob = top_blobs[0];
        top_blob.create(w, top_h, channels, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            unsigned char* outptr = top_blob.channel(q);

            for (size_t b = 0; b < bottom_blobs.size(); b++)
            {
                const Mat& bottom_blob = bottom_blobs[b];

                int size = bottom_blob.w * bottom_blob.h;

                const unsigned char* ptr = bottom_blob.channel(q);
                memcpy(outptr, ptr, size * elemsize);

                outptr += size * elemsize;
            }
        }

        return 0;
    }

    if (dims == 3 && axis == 2)
    {
        // interleave dim width
        int h = bottom_blobs[0].h;
        int channels = bottom_blobs[0].c;

        // total height
        int top_w = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob = bottom_blobs[b];
            top_w += bottom_blob.w;
        }

        Mat& top_blob = top_blobs[0];
        top_blob.create(top_w, h, channels, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            unsigned char* outptr = top_blob.channel(q);

            for (int i = 0; i < h; i++)
            {
                for (size_t b = 0; b < bottom_blobs.size(); b++)
                {
                    const Mat& bottom_blob = bottom_blobs[b];

                    const unsigned char* ptr = bottom_blob.channel(q).row<const unsigned char>(i);
                    memcpy(outptr, ptr, bottom_blob.w * elemsize);

                    outptr += bottom_blob.w * elemsize;
                }
            }
        }

        return 0;
    }

    return 0;
}

#if NCNN_CNNCACHE
bool Concat::needs_cache() const {return false;}
int Concat::forward_roi(std::vector<MRect>& bottom_padroi, std::vector<MRect>& top_roi, std::vector<MRect>& top_padroi) const
{
    NCNN_LOGE("IN FORWARD ROI OF CONCAT");
    //NCNN_LOGE("IN FORWARD ROI OF CONCAT, INPUT LAYERSIZE=%d", bottom_padroi[0].layersize);
    top_roi.resize(1);
    top_roi[0].layersize = bottom_padroi[0].layersize;
    top_padroi.resize(1);
    top_padroi[0].layersize = bottom_padroi[0].layersize;
    if (!bottom_padroi[1].size()) {
        for (MRect& roi: top_roi) {
            roi.copyFrom(bottom_padroi[0]);
        }
        for (MRect& roi: top_padroi) {
            roi.copyFrom(bottom_padroi[0]);
        }
    }
    else{
        MRect& mr = top_roi[0];
        MRect& mr2 = top_padroi[0];
        for (size_t i = 0, max = bottom_padroi[0].size(); i < max; i++) {
            int x1 = bottom_padroi[0].changed_vecs[i].x1;
            int y1 = bottom_padroi[0].changed_vecs[i].y1;
            int x2 = bottom_padroi[0].changed_vecs[i].x2;
            int y2 = bottom_padroi[0].changed_vecs[i].y2;
            for (size_t j = 1, maxx = bottom_padroi.size(); j < maxx; j++) {

                const struct rect temp = bottom_padroi[j].changed_vecs[i];
                if (temp.x1 <= x1 && temp.y1 <= y1) {
                    x1 = temp.x1;
                    y1 = temp.y1;
                }
                if (temp.x2 >= x2 && temp.y2 >= y2) {
                    x2 = temp.x2;
                    y2 = temp.y2;
                }
            }
            mr.add_rect(x1, y1, x2, y2);
            mr2.add_rect(x1, y1, x2, y2);
        }
    }
    NCNN_LOGE("END OF CONCAT");
    //top_roi.forward_in_conv_or_pool(bottom_padroi, pad_left, kernel_w, stride_w);
    //top_padroi.pad_in_conv_or_pool(top_roi, pad_left, kernel_w);
    //NCNN_LOGE("IN CONCAT LAYER");
    //NCNN_LOGE("ROI IS: %d, %d, %d, %d", top_roi[0].changed_vecs[0].x1, top_roi[0].changed_vecs[0].y1, top_roi[0].changed_vecs[0].x2, top_roi[0].changed_vecs[0].y2);
    //NCNN_LOGE("PAD ROI IS: %d, %d, %d, %d", top_padroi[0].changed_vecs[0].x1, top_padroi[0].changed_vecs[0].y1, top_padroi[0].changed_vecs[0].x2, top_padroi[0].changed_vecs[0].y2);
    return 0;
}

#endif

} // namespace ncnn
