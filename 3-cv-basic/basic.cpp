#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

int main()
{
    cv::Mat img = cv::imread("lena.jpg");
    cv::imshow("img", img);

    cv::Mat img_processed;
    cv::resize(img, img_processed, cv::Size(img.cols / 2, img.rows / 2));
    cv::imshow("resized", img_processed);

    cv::Mat R = cv::getRotationMatrix2D(cv::Point2f(img.cols / 2, img.rows / 2),
                                        45, 1);
    cv::warpAffine(img, img_processed, R, img.size());

    for (int i = 1; i < 20; i+=2)
    {
        cv::Mat img_blurred;
        cv::blur(img, img_blurred, cv::Size(i, i));

        cv::Canny(img_blurred, img_processed, 100, 150);

        cv::imshow("blurred", img_blurred);
        cv::imshow("processed", img_processed);
        cv::waitKey();
    }
}
