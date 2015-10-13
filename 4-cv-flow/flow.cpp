#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/video.hpp>
#include <iostream>

int main()
{
    cv::VideoCapture input("cars.mp4");
    cv::Mat img, img_prev;
    std::vector<cv::Point2f> points, points_prev;
    std::vector<uchar> status;
    std::vector<float> error;

    cv::OrbFeatureDetector detector;
    std::vector<cv::KeyPoint> keypoints;

    input.read(img);
    detector(img, cv::Mat(), keypoints);
    cv::KeyPoint::convert(keypoints, points);
    img.copyTo(img_prev);
    points_prev = points;

    for (;;)
    {
        if (!input.read(img))
            break;

        cv::calcOpticalFlowPyrLK(
            img_prev, img,
            points_prev, points,
            status, error);

        img.copyTo(img_prev);
        for (size_t i = 0; i < points.size(); i++)
        {
            cv::circle(img, points[i], 3,
                       cv::Scalar(0, 0,
                                  cv::norm(points_prev[i] - points[i]) * 20), 3);
        }
        points_prev = points;

        cv::imshow("img", img);
        char c = cv::waitKey(1);

        if (c == 27) // 27 is ESC code
            break;
    }
}
