#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <arm_neon.h>

template <typename F>
double benchmark(F fun, int n = 100)
{
    double start;
    std::vector<double> timings;
    for (int i = 0; i < n; i++)
    {
        start = cv::getTickCount();
        fun();
        timings.push_back((cv::getTickCount() - start) / cv::getTickFrequency());
    }

    std::sort(timings.begin(), timings.end());
    return timings[timings.size() / 2];
}

void blend(const cv::Mat &img1, const cv::Mat &img2, cv::Mat &dst, uchar alpha)
{
    CV_Assert(img1.type() == dst.type());
    CV_Assert(img2.type() == dst.type());
    CV_Assert(dst.type() == CV_8UC1);
    CV_Assert(img1.size() == dst.size());
    CV_Assert(img2.size() == dst.size());

    #pragma omp parallel for num_threads(4)
    for (int y = 0; y < dst.rows; y++)
    {
        const uchar *row1 = img1.ptr<uchar>(y);
        const uchar *row2 = img2.ptr<uchar>(y);
        uchar *row_dst = dst.ptr<uchar>(y);

        for (int x = 0; x < dst.cols; x++)
        {
            row_dst[x] = ((255 - alpha) * row1[x] + alpha * row2[x]) / 255;
        }
    }
}

void blend_neon(const cv::Mat &img1, const cv::Mat &img2, cv::Mat &dst, uchar alpha)
{
    CV_Assert(img1.type() == dst.type());
    CV_Assert(img2.type() == dst.type());
    CV_Assert(dst.type() == CV_8UC1);
    CV_Assert(img1.size() == dst.size());
    CV_Assert(img2.size() == dst.size());
    CV_Assert(dst.cols % 8 == 0);

    uint8x8_t v255 = vdup_n_u8(255);
    uint8x8_t valpha = vdup_n_u8(alpha);
    uint8x8_t vbeta = vsub_u8(v255, valpha);

    #pragma omp parallel for num_threads(4)
    for (int y = 0; y < dst.rows; y++)
    {
        const uchar *row1 = img1.ptr<uchar>(y);
        const uchar *row2 = img2.ptr<uchar>(y);
        uchar *row_dst = dst.ptr<uchar>(y);

        for (int x = 0; x < dst.cols; x+=8)
        {
            uint8x8_t vrow1 = vld1_u8(row1 + x);
            uint8x8_t vrow2 = vld1_u8(row2 + x);

            uint16x8_t vmul1 = vmull_u8(vbeta, vrow1);
            uint16x8_t vmul2 = vmull_u8(valpha, vrow2);

            uint16x8_t vsum = vaddq_u16(vmul1, vmul2);
            uint8x8_t vres = vshrn_n_u16(vsum, 8);

            vst1_u8(row_dst + x, vres);
        }
    }
}

int main()
{
    cv::Mat img1 = cv::imread("lena.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat img2, dst;
    cv::Canny(img1, img2, 100, 200);
    dst.create(img1.size(), img1.type());

    int step = 10;
    for(uchar alpha = 0 ;; alpha += step)
    {
        std::cout << "CPU:       " << benchmark([&](){
            blend(img1, img2, dst, alpha);
        }) << std::endl;
        std::cout << "NEON:      " << benchmark([&](){
            blend_neon(img1, img2, dst, alpha);
        }) << std::endl;

        cv::imshow("dst", dst);
        char c = cv::waitKey(1);

        if (c == 27) // 27 is ESC code
            break;

        if ((alpha + step > 255) || (alpha + step < 0))
            step = -step;
    }
}
