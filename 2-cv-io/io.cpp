#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>


int main()
{
    cv::Mat img = cv::imread("lena.jpg");

    cv::Mat dst;
    cv::Sobel(img, dst, CV_32F, 1, 1);

    cv::imwrite("lena_sobel.png", dst);
    cv::imshow("dst", dst / 256);
    cv::waitKey();

    cv::VideoCapture input("cars.mp4");
    cv::VideoWriter output
        ("cars_sobel.avi",
         CV_FOURCC('X', 'V', 'I', 'D'),
         30,
         cv::Size(input.get(CV_CAP_PROP_FRAME_WIDTH),
                  input.get(CV_CAP_PROP_FRAME_HEIGHT)));
    for (;;)
    {
        if (!input.read(img))
            break;

        cv::Sobel(img, dst, CV_8U, 1, 1);

        output.write(dst);

        cv::imshow("dst", img);
        char c = cv::waitKey(30);

        if (c == ' ' || c == 27) // 27 is ESC code
            break;
    }
}
