#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <iostream>

int main()
{
    cv::Mat left_right, left, right;
    cv::Mat left_gray, right_gray;

    cv::StereoBM bm(cv::StereoBM::BASIC_PRESET, 64);
    cv::Mat disparity;

    cv::Matx33f P(573.835, 0, 306,
                  0, 573.835, 256,
                  0, 0, 1);
    float baseline = 0.18;

    cv::VideoCapture left_right_video("left_right.mp4");

    for (;;)
    {
        if (!left_right_video.read(left_right))
            break;

        left  = left_right(cv::Rect(0, 0, left_right.cols / 2, left_right.rows));
        right = left_right(cv::Rect(left_right.cols / 2, 0, left_right.cols / 2, left_right.rows));

        cv::cvtColor(left, left_gray, CV_BGR2GRAY);
        cv::cvtColor(right, right_gray, CV_BGR2GRAY);

        bm(left_gray, right_gray, disparity, CV_32F);

        for (int y = 0; y < left.rows; y++)
        {
            for (int x = 0; x < left.cols; x++)
            {
                float d = disparity.ptr<float>(y)[x];
                float distance = P(0, 0) * baseline / d;

                cv::Vec3f point_3d = P.inv() * cv::Vec3f(x, y, 1) * distance;

                if (d > 0 && std::abs(point_3d[1] - 0.4) < 0.2)
                {
                    float safe_distance = 30;
                    float alpha = distance / (float)safe_distance;
                    left.ptr<cv::Vec3b>(y)[x] =
                        (1 - alpha) * cv::Vec3b(0, 0, 255) +
                        alpha * cv::Vec3b(0, 255, 0);
                }
            }
        }

        cv::imshow("disparity", disparity / 64);
        cv::imshow("left", left);
        char c = cv::waitKey(1);
        if (c == 27) // 27 is ESC code
            break;
    }
}
