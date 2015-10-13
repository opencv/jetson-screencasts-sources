#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/highgui/highgui.hpp>

template <typename F>
double benchmark(const F &fun, int n = 100)
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

int main()
{
    cv::Mat K = (cv::Mat_<float>(3,3) <<
                 1063.3341955043343, 0, 687.1716218551032,
                 0, 1063.5952314639012, 481.7283833013738,
                 0, 0, 1);

    cv::Mat D = (cv::Mat_<float>(5,1) << -0.2989368115367147, 0.1040355421814525, 0.000985076576025588, -0.0006065352713033385, 0.1024473704796796);

    {
        cv::Mat src_bgr = cv::imread("left01.jpg");
        cv::Mat src;
        cv::cvtColor(src_bgr, src, CV_BGR2GRAY);

        cv::Mat map1, map2;
        cv::initUndistortRectifyMap(K, D, cv::Mat::eye(3, 3, CV_32FC1), K, src_bgr.size(), CV_16SC2, map1, map2);

        cv::Mat dst;
        std::cout << "CPU:       " << benchmark([&](){
                cv::remap(src, dst, map1, map2, cv::INTER_LINEAR);
            }) << std::endl;

        cv::imshow("img", dst);
    }

    {
        cv::Mat src_bgr = cv::imread("left01.jpg");
        cv::Mat src;
        cv::cvtColor(src_bgr, src, CV_BGR2GRAY);

        cv::Mat map1, map2;
        cv::initUndistortRectifyMap(K, D, cv::Mat::eye(3, 3, CV_32FC1), K, src_bgr.size(), CV_32FC1, map1, map2);
        cv::gpu::GpuMat gpu_map1, gpu_map2;
        gpu_map1.upload(map1);
        gpu_map2.upload(map2);

        cv::gpu::GpuMat gpu_src(src.size(), CV_8UC1);
        cv::gpu::GpuMat gpu_dst(src.size(), CV_8UC1);

        cv::Mat dst;
        std::cout << "GPU:       " << benchmark([&](){
                gpu_src.upload(src);
                cv::gpu::remap(gpu_src, gpu_dst, gpu_map1, gpu_map2, cv::INTER_LINEAR);
                gpu_dst.download(dst);
            }) << std::endl;

        cv::imshow("img", dst);
    }

    {
        cv::Mat src_bgr = cv::imread("left01.jpg");

        cv::Mat map1, map2;
        cv::initUndistortRectifyMap(K, D, cv::Mat::eye(3, 3, CV_32FC1), K, src_bgr.size(), CV_32FC1, map1, map2);
        cv::gpu::GpuMat gpu_map1, gpu_map2;
        gpu_map1.upload(map1);
        gpu_map2.upload(map2);

        cv::gpu::CudaMem cudamem_src(src_bgr.size(), CV_8UC1, cv::gpu::CudaMem::ALLOC_ZEROCOPY);
        cv::gpu::CudaMem cudamem_dst(src_bgr.size(), CV_8UC1, cv::gpu::CudaMem::ALLOC_ZEROCOPY);

        cv::gpu::GpuMat gpu_src = cudamem_src.createGpuMatHeader();
        cv::gpu::GpuMat gpu_dst = cudamem_dst.createGpuMatHeader();

        cv::Mat src = cudamem_src.createMatHeader();
        cv::Mat dst = cudamem_dst.createMatHeader();

        cv::cvtColor(src_bgr, src, CV_BGR2GRAY);

        std::cout << "Zero-Copy: " << benchmark([&](){
                cv::gpu::remap(gpu_src, gpu_dst, gpu_map1, gpu_map2, cv::INTER_LINEAR);
            }) << std::endl;

        cv::imshow("img", dst);
    }

    cv::waitKey();
}
