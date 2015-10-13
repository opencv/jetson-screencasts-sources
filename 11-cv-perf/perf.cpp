#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/video.hpp>
#include <iostream>
#include <iomanip>

std::ostream &format()
{
    return std::cout << std::left << std::setw(13) << std::setprecision(4);
}

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

int main()
{
    cv::Mat img = cv::imread("lena.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat src, dst;

    format() << "resize";
    format() << "cvtColor";
    format() << "Sobel";
    format() << "StereoBM";
    format() << "HoughLinesP";
    format() << "calcOpticalFlowPyrLK";
    format() << std::endl;

    src.create(cv::Size(1280, 768), CV_8UC1);
    format() << benchmark([&](){
            cv::resize(img, src, src.size());
        });

    format() << benchmark([&](){
            cv::cvtColor(src, dst, CV_GRAY2BGR);
        });

    format() << benchmark([&](){
            cv::Sobel(src, dst, CV_16S, 1, 1);
        });

    cv::StereoBM bm(cv::StereoBM::BASIC_PRESET, 64);
    format() << benchmark([&](){
            bm(src, src, dst);
        }, 5);

    cv::Mat img_edges;
    cv::Canny(src, img_edges, 100, 150);
    std::vector<cv::Vec4i> lines;
    format() << benchmark([&](){
            cv::HoughLinesP(img_edges, lines, 1, CV_PI/180, 50, 25, 2);
        }, 10);

    cv::OrbFeatureDetector detector(5000);
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors0, descriptors1;
    std::vector<cv::Point2f> points, points_prev;
    std::vector<uchar> status;
    std::vector<float> err;
    cv::Mat cars0 = cv::imread("cars-0.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat cars1 = cv::imread("cars-1.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    detector(cars0, cv::Mat(), keypoints, descriptors0);
    cv::KeyPoint::convert(keypoints, points_prev);
    format() << benchmark([&](){
            cv::calcOpticalFlowPyrLK(cars0, cars1,
                                     points_prev,
                                     points,
                                     status,
                                     err);
        }, 10);

    format() << std::endl;
}
