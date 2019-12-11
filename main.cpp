#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/types_c.h>

using namespace cv;
using namespace std;
        
int main() {
    namedWindow("Window", WINDOW_NORMAL);
    Mat image1, image2, imageAux;
    //VideoCapture cap(); // webcam capture video
    VideoCapture cap("./assets/VIDEO"); // mp4 capture video
    if (!cap.isOpened()) { // verify cap
        cout << "camera ou arquivo em falta";
        return 1;
    }

    image1 = imread("./assets/IMAGEM", IMREAD_GRAYSCALE); // read image

    if (image1.empty()) { // verify imagem1
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
    
        Ptr<Feature2D> orb = ORB::create(400); // using ORB
        orb->detectAndCompute(image1, Mat(), kp1, descriptor1);
        orb->detectAndCompute(image2, Mat(), kp2, descriptor2);
        descriptor1.convertTo(descriptor1, CV_32F);
        descriptor2.convertTo(descriptor2, CV_32F);
        
       if ( descriptor1.empty() )
           break;
        if ( descriptor2.empty() )
            continue;
        
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
        //-- Homography
        vector<Point2f> obj;
        vector<Point2f> scene;
        
        for (int i = 0; i < good_matches.size(); i++) {
            obj.push_back(kp1[good_matches[i].queryIdx].pt);
            scene.push_back(kp2[good_matches[i].trainIdx].pt);
        }
        
        Mat H = findHomography( obj, scene, RANSAC );
        
        vector<Point2f> obj_corners(4);
        obj_corners[0] = cvPoint(0, 0);
        obj_corners[1] = cvPoint(image1.cols,0);
        obj_corners[2] = cvPoint(image1.cols, image1.rows);
        obj_corners[3] = cvPoint(0, image1.rows);
        vector<Point2f> scene_corners(4);
        
        perspectiveTransform(obj_corners, scene_corners, H);
        
        //-- Draw lines between the corners (the mapped object in the scene - image2 )
        line( imageAux, scene_corners[0] + Point2f( image1.cols, 0), scene_corners[1] + Point2f( image1.cols, 0), Scalar(0, 255, 0), 4 );
        line( imageAux, scene_corners[1] + Point2f( image1.cols, 0), scene_corners[2] + Point2f( image1.cols, 0), Scalar( 0, 255, 0), 4 );
        line( imageAux, scene_corners[2] + Point2f( image1.cols, 0), scene_corners[3] + Point2f( image1.cols, 0), Scalar( 0, 255, 0), 4 );
        line( imageAux, scene_corners[3] + Point2f( image1.cols, 0), scene_corners[0] + Point2f( image1.cols, 0), Scalar( 0, 255, 0), 4 );
        
        //-- Show image
        imshow("Window", imageAux);

        if (waitKey(1) == 27) {
            break;
        }
    }

    destroyAllWindows();
    return 0;
}
