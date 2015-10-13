#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <cmath>

int main(int argc, char** argv)
{
    cv::VideoCapture input ("signs.avi") ;

    cv::Mat img, img_gray, img_edges;

    for (;;)
    {
        if (!input.read(img))
            break;

        cv::cvtColor(img, img_gray, CV_BGR2GRAY);
        cv::Canny(img_gray, img_edges, 230, 460);

        std::vector<cv::Vec4i> lines;
        cv::HoughLinesP(img_edges, lines, 1, CV_PI/180, 50, 25, 2);

        for (size_t i = 0; i < lines.size(); i++)
        {
            cv::Vec4i l = lines[i];
            cv::line(img,
                     cv::Point(l[0], l[1]),
                     cv::Point(l[2], l[3]),
                     cv::Scalar(0,255,255),
                     3);
        }

        cv::GaussianBlur(img_gray, img_gray, cv::Size(9, 9), 2, 2);

        std::vector<cv::Vec3f> circles;
        cv::HoughCircles(img_gray, circles, CV_HOUGH_GRADIENT,
                         1.0f, 10.f, 100, 25, 1, 25);

        for (size_t i = 0; i < circles.size(); i++)
        {
            cv::Vec3f c = circles[i];
            cv::circle(img,
                       cv::Point(c[0], c[1]),
                       c[2],
                       cv::Scalar(0,255,0),
                       3);
        }

        cv::imshow("img", img);
        char c = cv::waitKey(1);
        if (c == 27) // 27 is ESC code
            break;
    }

    return 0;
}
