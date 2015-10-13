#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <iostream>
#include <algorithm>

int main()
{
    cv::VideoCapture input("cars.mp4");

    cv::Mat img, img_gray;

    cv::OrbFeatureDetector detector(5000);
    std::vector<cv::KeyPoint> img_keypoints, car_keypoints;
    cv::Mat img_descriptors, car_descriptors;

    input.read(img);
    cv::Mat car;
    img(cv::Rect(720, 320, 150, 100)).copyTo(car);

    detector(car, cv::Mat(), car_keypoints, car_descriptors);
    drawKeypoints(car, car_keypoints, car);

    for (;;)
    {
        if (!input.read(img))
            break;

        detector(img, cv::Mat(), img_keypoints, img_descriptors);
        drawKeypoints(img, img_keypoints, img);

        cv::BFMatcher matcher;
        std::vector< cv::DMatch > matches;
        matcher.match(car_descriptors, img_descriptors, matches);

        std::vector<cv::Point2f> car_points, img_points;
        for (int i = 0; i < matches.size(); i++)
        {
            car_points.push_back(car_keypoints[matches[i].queryIdx].pt);
            img_points.push_back(img_keypoints[matches[i].trainIdx].pt);
        }

        cv::Matx33f H = cv::findHomography(car_points, img_points, CV_RANSAC);

        std::vector<cv::Point> car_border, img_border;
        car_border.push_back(cv::Point(0, 0));
        car_border.push_back(cv::Point(0, car.rows));
        car_border.push_back(cv::Point(car.cols, car.rows));
        car_border.push_back(cv::Point(car.cols, 0));

        for (size_t i = 0; i < car_border.size(); i++)
        {
            cv::Vec3f p = H * cv::Vec3f(car_border[i].x, car_border[i].y, 1);
            img_border.push_back(cv::Point(p[0] / p[2], p[1] / p[2]));
        }

        cv::polylines(img, img_border, true, CV_RGB(0, 255, 0));

        cv::Mat img_matches;
        cv::drawMatches(car, car_keypoints, img, img_keypoints,
                        matches, img_matches);

        cv::imshow("img_matches", img_matches);
        char c = cv::waitKey(30);

        if (c == 27) // 27 is ESC code
            break;
    }
}
