
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;

// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        pt.x = Y.at<double>(0, 0) / Y.at<double>(2, 0); //Y.at<double>(0, 2); // pixel coordinates
        pt.y = Y.at<double>(1, 0) / Y.at<double>(2, 0); //Y.at<double>(0, 2);

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which match the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}

void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    //cv::namedWindow(windowName, 1);
	cv::namedWindow(windowName, 2);  // display two image windows
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}

// associate a given bounding box with the keypoints it1 contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    // ...
	// students implementation

	std::vector<double> distance;

	for (auto it = kptMatches.begin(); it != kptMatches.end(); it++)
	{
		const auto &kptCurr = kptsCurr[it->trainIdx];

		if (boundingBox.roi.contains(kptCurr.pt))
		{
			const auto &kptPrev = kptsPrev[it->queryIdx];

			distance.push_back(cv::norm(kptCurr.pt - kptPrev.pt));
		}
	}

	int distanceNum = distance.size();
	double distanceMean = std::accumulate(distance.begin(), distance.end(), 0.0) / distanceNum;

	for (auto it = kptMatches.begin(); it != kptMatches.end(); it++)
	{
		const auto &kptCurr = kptsCurr[it->trainIdx];

		if (boundingBox.roi.contains(kptCurr.pt))
		{
			int kptPrevIdx = it->queryIdx;
			const auto &kptPrev = kptsPrev[kptPrevIdx];

			if (cv::norm(kptCurr.pt - kptPrev.pt) < distanceMean * 1.3)
			{
				boundingBox.keypoints.push_back(kptCurr);
				boundingBox.kptMatches.push_back(*it);
			}
		}
	}

	std::cout << "Mean value (distance): " << distanceMean << std::endl;
	std::cout << "Number of keypoints (before filtering): " << distanceNum << std::endl;
	std::cout << "Number of keypoints  (after filtering): " << boundingBox.keypoints.size() << std::endl;
}

// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // ...
	// students implementation

	vector<double> distanceRatios; 

	for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1)
	{
		for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2)
		{
			double distanceCurr = cv::norm(kptsCurr.at(it1->trainIdx).pt - kptsCurr.at(it2->trainIdx).pt);
			double distancePrev = cv::norm(kptsPrev.at(it1->queryIdx).pt - kptsPrev.at(it2->queryIdx).pt);

			if (distancePrev > std::numeric_limits<double>::epsilon() && distanceCurr >= 100.0)
			{
				double distanceRatio = distanceCurr / distancePrev;
				distanceRatios.push_back(distanceRatio);
			}
		}
	}

	if (distanceRatios.size() == 0)
	{
		TTC = NAN;
	}
	else
	{
		std::sort(distanceRatios.begin(), distanceRatios.end());

		long medianIdx = floor(distanceRatios.size() / 2.0);
		double medianDistanceRatio = distanceRatios.size() % 2 == 0 ? (distanceRatios[medianIdx - 1] + distanceRatios[medianIdx]) / 2.0 : distanceRatios[medianIdx];

		TTC = -1.0 / frameRate / (1.0 - medianDistanceRatio);
	}
}

void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    // ...
	// students implementation
        
	double y = 0.63; // = (laneWidth - 0.2) / 2; double laneWidth = 1.46;

	auto compareY = [&y](const LidarPoint &lidarPoint) {return abs(lidarPoint.y) >= y; };

	lidarPointsPrev.erase(std::remove_if(lidarPointsPrev.begin(), lidarPointsPrev.end(), compareY), lidarPointsPrev.end());
	lidarPointsCurr.erase(std::remove_if(lidarPointsCurr.begin(), lidarPointsCurr.end(), compareY), lidarPointsCurr.end());

	unsigned int maxClosestPointsNum = 150;
	auto compareX = [](const LidarPoint &lidarPoint1, const LidarPoint &lidarPoint2) { return lidarPoint1.x > lidarPoint2.x; };

	if (lidarPointsPrev.size() < maxClosestPointsNum)
	{
		maxClosestPointsNum = lidarPointsPrev.size();
	}

	std::make_heap(lidarPointsPrev.begin(), lidarPointsPrev.end(), compareX);
	std::sort_heap(lidarPointsPrev.begin(), lidarPointsPrev.begin() + maxClosestPointsNum, compareX);

	if (lidarPointsCurr.size() < maxClosestPointsNum)
	{
		maxClosestPointsNum = lidarPointsCurr.size();
	}

	std::make_heap(lidarPointsCurr.begin(), lidarPointsCurr.end(), compareX);
	std::sort_heap(lidarPointsCurr.begin(), lidarPointsCurr.begin() + maxClosestPointsNum, compareX);

	auto sumX = [](const double sum, const LidarPoint &lidarPoint) { return sum + lidarPoint.x; };

	double xMeanPrev = std::accumulate(lidarPointsPrev.begin(), lidarPointsPrev.begin() + maxClosestPointsNum, 0.0, sumX) / maxClosestPointsNum;
	double xMeanCurr = std::accumulate(lidarPointsCurr.begin(), lidarPointsCurr.begin() + maxClosestPointsNum, 0.0, sumX) / maxClosestPointsNum;

	TTC = xMeanCurr * 1.0 / frameRate / (xMeanPrev - xMeanCurr);
}

void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    // ...
	// students implementation

	for (auto it1 = prevFrame.boundingBoxes.begin(); it1 != prevFrame.boundingBoxes.end(); it1++)
	{
		std::vector<vector<cv::DMatch>::iterator> match;

		for (auto it2 = matches.begin(); it2 != matches.end(); it2++)
		{
			if (it1->roi.contains(prevFrame.keypoints.at(it2->queryIdx).pt))
			{
				match.push_back(it2);
			}
		}

		std::multimap<int, int> bestMatches;

		for (auto it3 = match.begin(); it3 != match.end(); it3++)
		{
			for (auto it4 = currFrame.boundingBoxes.begin(); it4 != currFrame.boundingBoxes.end(); it4++)
			{
				if (it4->roi.contains(currFrame.keypoints.at((*it3)->trainIdx).pt))
				{
					bestMatches.insert(std::pair<int, int>(it4->boxID, (*it3)->trainIdx));
				}
			}
		}

		int idx = std::numeric_limits<int>::max();
		int bestMatchesCount = 0;

		if (bestMatches.size() > 0)
		{
			for (auto it5 = bestMatches.begin(); it5 != bestMatches.end(); it5++)
			{
				if (bestMatches.count(it5->first) > bestMatchesCount)
				{
					bestMatchesCount = bestMatches.count(it5->first);
					idx = it5->first;
				}
			}

			bbBestMatches.insert(std::pair<int, int>(it1->boxID, idx));
		}
	}
}
