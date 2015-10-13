#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>

int main()
{
    cv::Mat img = cv::imread("lena.jpg", CV_LOAD_IMAGE_GRAYSCALE);

    cv::CascadeClassifier face_detector("/usr/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml");

    cv::imshow("before", img);
    equalizeHist(img, img);

    std::vector<cv::Rect> faces;
    face_detector.detectMultiScale(img, faces);

    for (size_t i = 0; i < faces.size(); i++)
        cv::rectangle(img, faces[i], cv::Scalar(255));

    cv::imshow("img", img);
    cv::waitKey();
}
