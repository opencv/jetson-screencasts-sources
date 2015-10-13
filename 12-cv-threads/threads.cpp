#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

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
    cv::Mat img = cv::imread("lena.jpg");
    cv::Mat map_x(img.size(), CV_32FC1), map_y(img.size(), CV_32FC1);

    std::vector<double> timings;
    std::cout << benchmark(
        [&]()
        {
            #pragma omp parallel for num_threads(4)
            for (int i = 0; i < img.rows; i++)
            {
                for (int j = 0; j < img.cols; j++)
                {
                    int x = j - img.cols / 2;
                    int y = i - img.rows / 2;

                    float r = sqrt(x * x + y * y);
                    float a = atan2(y, x);
                    float a2 = a + r * 0.004;

                    map_x.at<float>(i, j) = r * cos(a2) + img.cols / 2;
                    map_y.at<float>(i, j) = r * sin(a2) + img.rows / 2;
                }
            }
        }, 10) << std::endl;

    cv::Mat img_processed;
    cv::remap(img, img_processed, map_x, map_y, cv::INTER_LINEAR);

    cv::imshow("img", img_processed);
    cv::waitKey(0);
}
