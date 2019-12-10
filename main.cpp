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
	VideoCapture cap(""); // rename video
	if (!cap.isOpened()) { //verifica se cap abriu como esperado
		cout << "camera ou arquivo em falta";
		return 1;
	}

	image1 = imread("", IMREAD_GRAYSCALE); // rename image

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
	
//		Ptr<Feature2D> orb = ORB::create(400);
		Ptr<Feature2D> orb = xfeatures2d::SURF::create(400);
		orb->detectAndCompute(image1, Mat(), kp1, descriptor1);
		orb->detectAndCompute(image2, Mat(), kp2, descriptor2);

        Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
        vector< vector<DMatch> > knn_matches;
        matcher->knnMatch( descriptor1, descriptor2, knn_matches, 2 );

        const float ratio_thresh = 0.7f;
        vector<DMatch> good_matches;
        for (size_t i = 0; i < knn_matches.size(); i++) {
            if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance) {
                good_matches.push_back(knn_matches[i][0]);
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