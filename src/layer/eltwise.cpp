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

#include "eltwise.h"

namespace ncnn {

Eltwise::Eltwise()
{
    one_blob_only = false;
    support_inplace = false; // TODO inplace reduction
}

int Eltwise::load_param(const ParamDict& pd)
{
    op_type = pd.get(0, 0);
    coeffs = pd.get(1, Mat());

    return 0;
}

int Eltwise::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    //NCNN_LOGE("IN ELEWISE FORWARD");
    const Mat& bottom_blob = bottom_blobs[0];
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int size = w * h;

    Mat& top_blob = top_blobs[0];
    top_blob.create(w, h, channels, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    if (op_type == Operation_PROD)
    {
        // first blob
        const Mat& bottom_blob1 = bottom_blobs[1];
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const float* ptr = bottom_blob.channel(q);
            const float* ptr1 = bottom_blob1.channel(q);
            float* outptr = top_blob.channel(q);

            for (int i = 0; i < size; i++)
            {
                outptr[i] = ptr[i] * ptr1[i];
            }
        }

        for (size_t b = 2; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob1 = bottom_blobs[b];
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_blob1.channel(q);
                float* outptr = top_blob.channel(q);

                for (int i = 0; i < size; i++)
                {
                    outptr[i] *= ptr[i];
                }
            }
        }
    }
    else if (op_type == Operation_SUM)
    {
        if (coeffs.w == 0)
        {
            // first blob
            const Mat& bottom_blob1 = bottom_blobs[1];
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);
                const float* ptr1 = bottom_blob1.channel(q);
                float* outptr = top_blob.channel(q);

                for (int i = 0; i < size; i++)
                {
                    outptr[i] = ptr[i] + ptr1[i];
                }
            }

            for (size_t b = 2; b < bottom_blobs.size(); b++)
            {
                const Mat& bottom_blob1 = bottom_blobs[b];
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const float* ptr = bottom_blob1.channel(q);
                    float* outptr = top_blob.channel(q);

                    for (int i = 0; i < size; i++)
                    {
                        outptr[i] += ptr[i];
                    }
                }
            }
        }
        else
        {
            // first blob
            const Mat& bottom_blob1 = bottom_blobs[1];
            float coeff0 = coeffs[0];
            float coeff1 = coeffs[1];
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);
                const float* ptr1 = bottom_blob1.channel(q);
                float* outptr = top_blob.channel(q);

                for (int i = 0; i < size; i++)
                {
                    outptr[i] = ptr[i] * coeff0 + ptr1[i] * coeff1;
                }
            }

            for (size_t b = 2; b < bottom_blobs.size(); b++)
            {
                const Mat& bottom_blob1 = bottom_blobs[b];
                float coeff = coeffs[b];
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const float* ptr = bottom_blob1.channel(q);
                    float* outptr = top_blob.channel(q);

                    for (int i = 0; i < size; i++)
                    {
                        outptr[i] += ptr[i] * coeff;
                    }
                }
            }
        }
    }
    else if (op_type == Operation_MAX)
    {
        // first blob
        const Mat& bottom_blob1 = bottom_blobs[1];
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const float* ptr = bottom_blob.channel(q);
            const float* ptr1 = bottom_blob1.channel(q);
            float* outptr = top_blob.channel(q);

            for (int i = 0; i < size; i++)
            {
                outptr[i] = std::max(ptr[i], ptr1[i]);
            }
        }

        for (size_t b = 2; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob1 = bottom_blobs[b];
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_blob1.channel(q);
                float* outptr = top_blob.channel(q);

                for (int i = 0; i < size; i++)
                {
                    outptr[i] = std::max(outptr[i], ptr[i]);
                }
            }
        }
    }

    return 0;
}


#if NCNN_CNNCACHE
bool Eltwise::needs_cache() const {return false;}
int Eltwise::forward_roi(std::vector<MRect>& bottom_padroi, std::vector<MRect>& top_roi, std::vector<MRect>& top_padroi) const
{
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
    /*std::string ret("in eltwise, the input layers are:");
    for (auto temp : bottoms)
    {
        ret += std::to_string(temp);
        ret += " ";
    }
    ret += "the output layers are:";
    for (auto temp : tops){
        ret += std::to_string(temp);
        ret += " ";
    }
    NCNN_LOGE("%s", ret.c_str());
    NCNN_LOGE("top_roi info: ");
    top_roi[0].info();*/
    //NCNN_LOGE("top_padroi info: ");
    //top_padroi[0].info();
    return 0;
}

#endif

} // namespace ncnn
