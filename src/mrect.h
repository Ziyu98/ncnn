#ifndef NCNN_MRECT_H
#define NCNN_MRECT_H

#if NCNN_CNNCACHE

namespace ncnn {



inline bool skip_reuse(bool* cached_map, int outw, int outh) {
    int changed_pixel = 0;
    for (int i = 0; i < outw * outh; i ++) {
        changed_pixel += cached_map[i] ? 1 : 0;
    }
    float ratio = 1.0 * changed_pixel / outw / outh;
    if (ratio > 0.8) return true;
    else return false;
}

struct rect{
    int x1;
    int y1;
    int x2;
    int y2;
    rect(int arg0, int arg1,int arg2, int arg3) {
        x1 = arg0;
        y1 = arg1;
        x2 = arg2;
        y2 = arg3;
    }
    rect() {}
};

class MRect
{

public:

    MRect() {}

    void set_offset(int x, int y) {
        x_offset = x;
        y_offset = y;
    }

    void add_rect(int arg0, int arg1, int arg2, int arg3) {
        changed_vecs.push_back(rect(arg0, arg1, arg2, arg3));
    }

    void copyFrom(MRect other) {
        x_offset = other.x_offset;
        y_offset = other.y_offset;
        layersize = other.layersize;
        changed_vecs.resize(0);
    	for (struct rect r: other.changed_vecs) {
            this->changed_vecs.push_back(r);
        }
    }

    std::string info() {
        std::string ret("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
        NCNN_LOGE("ROI INFO\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");
        //char a[64];
        for (unsigned i = 0; i < changed_vecs.size(); i ++) {
            //if (i > 0)
                //ret += ", ";
            struct rect r1 = changed_vecs[i];
            NCNN_LOGE("(%d,%d,%d,%d)", r1.x1, r1.y1, r1.x2, r1.y2);
            //sprintf(a, "(%d,%d,%d,%d)", r1.x1, r1.y1, r1.x2, r1.y2);
            //ret += a;
        }
        //ret += "-----------------------------------------------\n";
        NCNN_LOGE("--------------------------------------------------");
        return ret;
    }

    int size() {
        return changed_vecs.size();
    }

    void forward_rect_conv_or_pool(
        struct rect& r1, struct rect& r2, int pad, int ksize, int stride) {
        
        // TODO: not correct yet
        // int off = stride > 1 ? 1 : 0;
        if (pad >= 0) { // SAME
            r1.x1 = std::max(0, (r2.x1 + pad) / stride);
            r1.y1 = std::max(0, (r2.y1 + pad) / stride);
            r1.x2 = (r2.x2 - pad) / stride;
            r1.y2 = (r2.y2 - pad) / stride;
        }
        else if (pad == -233) { // VALID
            r1.x1 = std::max(0, (r2.x1 + ksize / 2) / stride);
            r1.y1 = std::max(0, (r2.y1 + ksize / 2) / stride);
            r1.x2 = (r2.x2 - ksize / 2) / stride;
            r1.y2 = (r2.y2 - ksize / 2) / stride;
        }
        // LOGI("Yes (%d,%d,%d,%d) -> (%d,%d,%d,%d)\n",
        //     r2.x1, r2.y1, r2.x2, r2.y2, r1.x1, r1.y1, r1.x2, r1.y2);
    }

    int forward_in_conv_or_pool(MRect& bottom_mrect, int pad, int ksize, int stride) {
        // offset
        x_offset = bottom_mrect.x_offset / stride;
        y_offset = bottom_mrect.y_offset / stride;
        layersize = bottom_mrect.layersize / stride;
        NCNN_LOGE("IN FORWARD CONV OR POOL, INPUT LAYERSIZE=%d, OUTPUT LAYERSIZE=%d", bottom_mrect.layersize, layersize);

        size_t size = bottom_mrect.size();
        changed_vecs.resize(size);
        for (size_t i = 0; i < size; i ++) {
            forward_rect_conv_or_pool(
                changed_vecs[i], bottom_mrect.changed_vecs[i], pad, ksize, stride);
        }
        return 0;
    }

    void pad_rect_conv_or_pool(
        struct rect& r1, struct rect& r2, int pad, int ksize, int layersize) {
        if (pad > 0) {
            r1.x1 = std::max(0, r2.x1 - ksize + 1);
            r1.y1 = std::max(0, r2.y1 - ksize + 1);
            r1.x2 = std::min(r2.x2 + ksize - 1, layersize - 1);
            r1.y2 = std::min(r2.y2 + ksize - 1, layersize - 1);
        }
        else {
            r1.x1 = r2.x1;
            r1.y1 = r2.y1;
            r1.x2 = r2.x2;
            r1.y2 = r2.y2;
        }
    }

    int pad_in_conv_or_pool(MRect& top_mrect, int pad, int ksize) {
        size_t size = top_mrect.size();
        changed_vecs.resize(size);
        layersize = top_mrect.layersize;
        for (size_t i = 0; i < size; i++) {
            pad_rect_conv_or_pool(
                changed_vecs[i], top_mrect.changed_vecs[i], pad, ksize, top_mrect.layersize);
        }
        return 0;
    }

    int x_offset;
    int y_offset;
    int layersize;
    std::vector<struct rect> changed_vecs;
};

} // namespace ncnn

#endif // NCNN_CNNCACHE

#endif // NCNN_MRECT_H
