#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/calib3d.hpp>

using namespace cv;
using namespace std;
        
int main() {
    namedWindow("Teste", WINDOW_NORMAL);
    Mat image1, image2, imageAux;
    VideoCapture cap(0); // rename video
    if (!cap.isOpened()) { //verifica se cap abriu como esperado
        cout << "camera ou arquivo em falta";
        return 1;
    }

    image1 = imread("dc.jpg", IMREAD_GRAYSCALE); // rename image

    if (image1.empty()) { //verifica a imagem1
        cout << "imagem 1 vazia";
        return 1;
    }

    while (true) {
        cap >> image2;
        if (image2.empty()) {
            cout << "imagem 2 vazia";
            return 1;
        }

        cvtColor(image2, image2, COLOR_BGR2GRAY);

        vector<KeyPoint> kp1, kp2;
        Mat descriptor1, descriptor2;
    
        Ptr<Feature2D> orb = ORB::create(400);
//        Ptr<Feature2D> orb = xfeatures2d::SURF::create(400);
        orb->detectAndCompute(image1, Mat(), kp1, descriptor1);
        orb->detectAndCompute(image2, Mat(), kp2, descriptor2);
        descriptor1.convertTo(descriptor1, CV_32F); descriptor2.convertTo(descriptor2, CV_32F);
        
       if ( descriptor1.empty() )
           break;
        if ( descriptor2.empty() )
            continue;
/*
        Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
        vector< vector<DMatch> > knn_matches;
        matcher->knnMatch( descriptor1, descriptor2, knn_matches, 2 );*/
        
        FlannBasedMatcher matcher;
        vector< DMatch > matches;
        matcher.match( descriptor1, descriptor2, matches );

        const float ratio_thresh = 2;
        vector<DMatch> good_matches;
        for (size_t i = 0; i < matches.size(); i++) {
            if (matches[i].distance < ratio_thresh * matches[i].distance) {
                good_matches.push_back(matches[i]);
            }
        }

        drawMatches( image1, kp1, image2, kp2, good_matches, imageAux, Scalar::all(-1),
                        Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

        imshow("Teste", imageAux);

        if (waitKey(1) == 27) {
            break;
        }
    }

    destroyAllWindows();
    return 0;
}
