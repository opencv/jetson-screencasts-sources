#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <iostream>

int main()
{
    cv::Mat image[2], image_gray[2];
    std::vector< cv::Point2f > corners_image[2];
    std::vector< std::vector<cv::Point2f> > points_image[2];
    cv::Size pattern_size(9, 6);
    cv::VideoCapture input[2];
    input[0].open("left%02d.jpg");
    input[1].open("right%02d.jpg");

    for (;;)
    {
        if (!input[0].read(image[0]) || !input[1].read(image[1]))
            break;

        for (int i = 0; i < 2; i++)
            cv::cvtColor(image[i], image_gray[i], CV_BGR2GRAY);

        bool found[2];
        for (int i = 0; i < 2; i++)
            found[i] = cv::findChessboardCorners(image_gray[i], pattern_size, corners_image[i]);

        if (found[0] && found[1])
        {
            for (int i = 0; i < 2; i++)
            {
                cv::drawChessboardCorners(image[i], pattern_size, corners_image[i], found[i]);
                points_image[i].push_back(corners_image[i]);
            }
        }

        cv::imshow("left", image[0]);
        cv::imshow("right", image[1]);
        char c = cv::waitKey(0);
        if (c == 27) // 27 is ESC code
            break;
    }

    std::vector< cv::Point3f > corners_world;
    for (int i = 0; i < pattern_size.height; i++)
        for (int j = 0; j < pattern_size.width; j++)
            corners_world.push_back(cv::Point3f(j, i, 0));
    std::vector< std::vector<cv::Point3f> >
        points_world(points_image[0].size(), corners_world);

    cv::Mat K[2], D[2];
    cv::Mat R, T, E, F;
    double rms = cv::stereoCalibrate(points_world, points_image[0], points_image[1],
                                     K[0], D[0],
                                     K[1], D[1],
                                     image_gray[0].size(), R, T, E, F,
                                     cv::TermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 100, 1e-5),
                                     CV_CALIB_FIX_ASPECT_RATIO +
                                     CV_CALIB_ZERO_TANGENT_DIST +
                                     CV_CALIB_SAME_FOCAL_LENGTH
                                     );

    std::cout << "Reprojection error: " << rms << std::endl;

    cv::Mat Q;
    cv::Mat Rrect[2], P[2];
    cv::stereoRectify(K[0], D[0],
                      K[1], D[1],
                      image_gray[0].size(), R, T, Rrect[0], Rrect[1], P[0], P[1], Q,
                      cv::CALIB_ZERO_DISPARITY, 0
                      );

    cv::Mat left_right_rectified(cv::Size(image_gray[0].cols * 2, image_gray[0].rows), CV_8UC1);
    cv::Mat image_rectified[2];
    cv::Mat map[2][2];
    image_rectified[0] = left_right_rectified(cv::Rect(0, 0, left_right_rectified.cols / 2, left_right_rectified.rows));
    image_rectified[1] = left_right_rectified(cv::Rect(left_right_rectified.cols / 2, 0, left_right_rectified.cols / 2, left_right_rectified.rows));
    for (int i = 0; i < 2; i++)
    {
        cv::initUndistortRectifyMap(K[i], D[i], Rrect[i], P[i],
                                    image_gray[i].size(),
                                    CV_16SC2, map[i][0], map[i][1]);

        cv::remap(image_gray[i], image_rectified[i], map[i][0], map[i][1], CV_INTER_LINEAR);
    }

    bool found[2];
    for (int i = 0; i < 2; i++)
        found[i] = cv::findChessboardCorners(image_rectified[i], pattern_size, corners_image[i]);

    float epipolar_error = 0.0f;
    for (int i = 0; i < corners_image[0].size(); i++)
    {
        cv::Point2f
            pt1 = corners_image[0][i],
            pt2 = corners_image[1][i] + cv::Point2f(left_right_rectified.cols / 2, 0);

        cv::line(left_right_rectified, pt1, pt2, 255);
        epipolar_error += std::abs(pt1.y - pt2.y);
    }

    epipolar_error /= corners_image[0].size();
    std::cout << "Epipolar error: " << epipolar_error << std::endl;

    cv::imshow("rectified", left_right_rectified);
    cv::waitKey(0);
}
