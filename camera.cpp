#include <iostream>
#include <sstream>
#include <time.h>
#include <stdio.h>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

void myCalibrateCamera(
        vector<vector<Point3f>>& ObjectPoints, 
        vector<vector<Point2f>>& ImagePoints, 
        Size imageSize, 
        Mat& cameraMatrix, 
        Mat& distCoeffs, 
        vector<Mat>& rvecs, 
        vector<Mat>& tvecs, 
        int flags = 0)
{
    Mat P;

    for(int i = 0; i < ObjectPoints.size(); i++)
    {
        for(int j = 0; j < ObjectPoints[i].size(); j++)
        {
            Mat row1 = Mat::zeros(1, 12, CV_64F);
            Mat row2 = Mat::zeros(1, 12, CV_64F);

            Point3f objPoint = ObjectPoints[i][j];
            Point2f imgPoint = ImagePoints[i][j];

            row1.at<double>(0, 0) = objPoint.x;
            row1.at<double>(0, 1) = objPoint.y;
            row1.at<double>(0, 2) = objPoint.z;
            row1.at<double>(0, 3) = 1.0f;

            row1.at<double>(0, 8) = -1 * imgPoint.x * objPoint.x;
            row1.at<double>(0, 9) = -1 * imgPoint.x * objPoint.y;
            row1.at<double>(0, 10) = -1 * imgPoint.x * objPoint.z;
            row1.at<double>(0, 11) = -1 * imgPoint.x;
            
            row2.at<double>(0, 4) = objPoint.x;
            row2.at<double>(0, 5) = objPoint.y;
            row2.at<double>(0, 6) = objPoint.z;
            row2.at<double>(0, 7) = 1.0f;

            row2.at<double>(0, 8) = -1 * imgPoint.y * objPoint.x;
            row2.at<double>(0, 9) = -1 * imgPoint.y * objPoint.y;
            row2.at<double>(0, 10) = -1 * imgPoint.y * objPoint.z;
            row2.at<double>(0, 11) = -1 * imgPoint.y;
            
            P.push_back(row1);
            P.push_back(row2);
        }
    }

    Mat endP;

    mulTransposed(P, endP, true);

    cout<<"P: "<<endP<<endl;

    Mat S, U, V;

    SVD::compute(endP, S, U, V);

    cout<<"S: "<<S<<endl;

    cout<<"U: "<<U<<endl;
    
    cout<<"V: "<<V<<endl;
}

int main()
{
    VideoCapture input(0);

    int width = 5;
    int height = 8;
    int boardNum = 8;
    int boardDim = width * height;
    Size boardSize = Size(width, height);

    if(!input.isOpened())
    {
        return -1;
    }

    vector<vector<Point2f>> imagePoints; 
    vector<vector<Point3f>> objectPoints;
    Mat intrinsicMat = Mat::eye(3, 3, CV_64F);
    Mat distortionCoeffs = Mat::zeros(8, 1, CV_64F);

    vector<Point2f> corners;

    int successes = 0;

    Mat frame;
    Mat grayImg;
    // Need at lease 8 boardNum samples
    while(successes < boardNum)
    {
        input >> frame;

        cvtColor(frame, grayImg, COLOR_BGR2GRAY);

        bool found = findChessboardCorners(
            frame, 
            boardSize, 
            corners, 
            CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FILTER_QUADS
        );

        drawChessboardCorners(frame, boardSize, corners, found);

        imshow("frame", frame);

        if(corners.size() == boardDim)
        {
            imagePoints.push_back(corners);

            vector<Point3f> temp;
            for(int i = 0; i < corners.size(); i++)
            {
                temp.push_back(Point3f(i / width, i % width, 0.0f));
            }

            objectPoints.push_back(temp);
        
            successes++;
        }

        cout<<successes<<" "<<corners.size()<<" "<<boardDim<<endl;

        if(waitKey(15) >= 0) break;
    }

    vector<Mat> rvecs, tvecs;

    cout<<"Start Calibrate"<<endl;
    myCalibrateCamera(objectPoints, imagePoints, frame.size(), intrinsicMat, distortionCoeffs, rvecs, tvecs, CALIB_FIX_ASPECT_RATIO); 
   
    /*cout<<"Start Store"<<endl;
    FileStorage fs("Intrinsics.xml", FileStorage::WRITE);

    fs << "Intrinsics" << intrinsicMat;

    fs.release();

    fs = FileStorage("Distortion.xml", FileStorage::WRITE);

    fs << "Distortion" << distortionCoeffs;

    fs.release();

    Mat map1;
    Mat map2;

    cout<<"Start initUndistortMap"<<endl;
    initUndistortRectifyMap(intrinsicMat, distortionCoeffs, Mat(), intrinsicMat, frame.size(), CV_32FC1, map1, map2);

    while(true)
    {
        Mat undistorted;

        imshow("Calibration", frame);

        cout<<"Start remap"<<endl;
        remap(frame, undistorted, map1, map2, INTER_LINEAR);

        imshow("Undistorted", undistorted);
        
        input >> frame;

        if(waitKey(15) >= 0) break;
    }*/

    return 0;
}
