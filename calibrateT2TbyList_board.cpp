//**************************************
//*New calibration code for increase the accuracy
//*by using marker board from the support camera.
//*Input:	intrinsic matrix of support camera
//*			image list from supprot camera
//*			"marker to target" matrix for first target camera
//*			"marker to target" matrix for second target camera
//*Output:	"target camera to target camera" transformation matrix
//**************************************

#include <iostream>
#include <fstream>
#include <sstream>
#include <dirent.h>
#include <err.h>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "aruco/aruco.h"
#include "aruco/cvdrawingutils.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/registration/transformation_estimation_svd.h>
using namespace cv;
using namespace std;

//*******************************************
//Take 3 by 3 rotation and 1 by 3 translation marix,
//Output homogeneous transformation matrix
//*******************************************
template<typename T>
void RT2homogeneous( cv::Mat& H, cv::Mat& R, cv::Mat& t)
{

	H.at<T>(0, 0) = R.at<T>(0, 0);	H.at<T>(0, 1) = R.at<T>(0, 1);	H.at<T>(0, 2) = R.at<T>(0, 2);	H.at<T>(0, 3) = t.at<T>( 0);
	H.at<T>(1, 0) = R.at<T>(1, 0);	H.at<T>(1, 1) = R.at<T>(1, 1);	H.at<T>(1, 2) = R.at<T>(1, 2);	H.at<T>(1, 3) = t.at<T>( 1);
	H.at<T>(2, 0) = R.at<T>(2, 0);	H.at<T>(2, 1) = R.at<T>(2, 1);	H.at<T>(2, 2) = R.at<T>(2, 2);	H.at<T>(2, 3) = t.at<T>( 2);
	H.at<T>(3, 0) = 0			;	H.at<T>(3, 1) = 0			;	H.at<T>(3, 2) = 0			;	H.at<T>(3, 3) = 1			;

};

void draw3dCamera(cv::Mat &Image, aruco::Marker &m, const aruco::CameraParameters &CP, const cv::Scalar &color, char* text = NULL, int position = 0 ) {

    float size = m.ssize * 1;

	//5 points, camera and ul, ur, bl, br (project plane)
    Mat objectPoints(5, 3, CV_32FC1);

	//original
    objectPoints.at< float >(0, 0) = 0;
    objectPoints.at< float >(0, 1) = 0;
    objectPoints.at< float >(0, 2) = -size * 2;

	//ul
    objectPoints.at< float >(1, 0) = -size;
    objectPoints.at< float >(1, 1) = -size;
    objectPoints.at< float >(1, 2) = 0;

	//ur
    objectPoints.at< float >(2, 0) = size;
    objectPoints.at< float >(2, 1) = -size;
    objectPoints.at< float >(2, 2) = 0;

	//bl
    objectPoints.at< float >(3, 0) = -size;
    objectPoints.at< float >(3, 1) = size;
    objectPoints.at< float >(3, 2) = 0;

	//br
    objectPoints.at< float >(4, 0) = size;
    objectPoints.at< float >(4, 1) = size;
    objectPoints.at< float >(4, 2) = 0;

    vector< Point2f > imagePoints;
    cv::projectPoints(objectPoints, m.Rvec, m.Tvec, CP.CameraMatrix, CP.Distorsion, imagePoints);
	//for (int i = 0; i < 4; i++)
		//std::cout<<imagePoints[i]<<endl;
    // draw lines of project plane
    cv::line(Image, imagePoints[1], imagePoints[2], color, 5, CV_AA);
    cv::line(Image, imagePoints[2], imagePoints[4], color, 5, CV_AA);
    cv::line(Image, imagePoints[4], imagePoints[3], color, 5, CV_AA);
    cv::line(Image, imagePoints[3], imagePoints[1], color, 5, CV_AA);

    // draw lines of camera
    cv::line(Image, imagePoints[0], imagePoints[2], color, 5, CV_AA);
    cv::line(Image, imagePoints[0], imagePoints[3], color, 5, CV_AA);
    cv::line(Image, imagePoints[0], imagePoints[4], color, 5, CV_AA);
    cv::line(Image, imagePoints[0], imagePoints[1], color, 5, CV_AA);

    //draw text
    if (text == NULL)
	    return;

    cv::Point2f offset (50, 50);
    putText(Image, text, imagePoints[position] + offset, FONT_HERSHEY_SIMPLEX, 2, color, 4);

    //cv::line(Image, imagePoints[0], imagePoints[1], Scalar(0, 0, 255, 255), 1, CV_AA);
    //cv::line(Image, imagePoints[0], imagePoints[2], Scalar(0, 255, 0, 255), 1, CV_AA);
    //cv::line(Image, imagePoints[0], imagePoints[3], Scalar(255, 0, 0, 255), 1, CV_AA);
    //putText(Image, "x", imagePoints[1], FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 255, 255), 2);
    //putText(Image, "y", imagePoints[2], FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 0, 255), 2);
    //putText(Image, "z", imagePoints[3], FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 0, 0, 255), 2);
}

float thetaErr( const cv::Mat& r1, const cv::Mat& r2)
{
	//calculate the theta
	//theta = arccos((trace(R')-1)/2)
	//R' * R2 = R1 -> R2.inv * R' * R2 = R2.inv * R1 -> R' = R2.inv * R1
	cv::Mat R1, R2, Rp;
	cv::Rodrigues(r1, R1);
	cv::Rodrigues(r2, R2);
	Rp = R2.inv() * R1;
	float tr = cv::trace(Rp)[0];

	return acos(( tr -1)/2);
};

static bool readStringList( const string& filename, vector<string>& l )
{
	l.resize(0);
	FileStorage fs(filename, FileStorage::READ);
	if( !fs.isOpened() )
		return false;
	FileNode n = fs.getFirstTopLevelNode();
	if( n.type() != FileNode::SEQ )
		return false;
	FileNodeIterator it = n.begin(), it_end = n.end();
	for( ; it != it_end; ++it )
		l.push_back((string)*it);
	return true;
}

template<typename _Tp, int _rows, int _cols, int _options, int _maxRows, int _maxCols>
void eigen2cv( const Eigen::Matrix<_Tp, _rows, _cols, _options, _maxRows, _maxCols>& src, Mat& dst)
{
    if( !(src.Flags & Eigen::RowMajorBit) )
    {
        Mat _src(src.cols(), src.rows(), DataType<_Tp>::type,
              (void*)src.data(), src.stride()*sizeof(_Tp));
        transpose(_src, dst);
    }
    else
    {
        Mat _src(src.rows(), src.cols(), DataType<_Tp>::type,
                 (void*)src.data(), src.stride()*sizeof(_Tp));
        _src.copyTo(dst);
    }
}

template<typename _Tp>
void cv2eigen( const Mat& src,
               Eigen::Matrix<_Tp, Eigen::Dynamic, Eigen::Dynamic>& dst )
{
    dst.resize(src.rows, src.cols);
    if( !(dst.Flags & Eigen::RowMajorBit) )
    {
        Mat _dst(src.cols, src.rows, DataType<_Tp>::type,
             dst.data(), (size_t)(dst.stride()*sizeof(_Tp)));
        if( src.type() == _dst.type() )
            transpose(src, _dst);
        else if( src.cols == src.rows )
        {
            src.convertTo(_dst, _dst.type());
            transpose(_dst, _dst);
        }
        else
            Mat(src.t()).convertTo(_dst, _dst.type());
        CV_DbgAssert(_dst.data == (uchar*)dst.data());
    }
    else
    {
        Mat _dst(src.rows, src.cols, DataType<_Tp>::type,
                 dst.data(), (size_t)(dst.stride()*sizeof(_Tp)));
        src.convertTo(_dst, _dst.type());
        CV_DbgAssert(_dst.data == (uchar*)dst.data());
    }
}

cv::Mat estimateTransformation(std::vector<cv::Mat>& M2S1List, std::vector<cv::Mat>& M2S2List, cv::Mat& M2T1, cv::Mat& M2T2, std::vector<cv::Point3f>& boardPattern)
{
	//original equation:
	//T2T = M2T_2 * M2S_2^{-1} * M2S_1 * M2T_1^{-1}

	//M2S2*Point = T_{M2M}*M2S1*Point<-this is wrong
	//Point = M2M*M2S2.inv*M2S1*Point
	//OK lets just use PCL funtion to solve it
	//so what we need to do is simply some init and warpping

	cv:Mat M2S1, M2S2;

	//init a pcl cloud from boardPattern
	pcl::PointCloud<pcl::PointXYZ> pat;
	for (int i = 0; i < boardPattern.size(); i++)
	{
		pcl::PointXYZ point;
		point.x = boardPattern[i].x;
		point.y = boardPattern[i].y;
		point.z = boardPattern[i].z;

		pat.push_back(point);
	}

	//calculate Pointset1, by M2S2*boardPattern
	pcl::PointCloud<pcl::PointXYZ> tgt;
	for (int i = 0; i < M2S2List.size(); i++)
	{
		//M2S2 = M2S2List[i];
		//Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > m2s2(M2S2.ptr<float>(), M2S2.rows, M2S2.cols);
		pcl::PointCloud<pcl::PointXYZ> tgtsingle;
		//pcl::transformPointCloud (pat, tgtsingle, m2s2);
		tgt += pat;

		//cout<<"M2S2: \n"<<M2S2<<endl;
		//cout<<"m2s2: \n"<<m2s2<<endl;
	}
	cout<<"size of tgt is: "<<tgt.size()<<endl;
	//cout<<"tgt is: "<<tgt<<endl;


	//calculate Pointset2, by M2S2^{-1}*M2S1*boardPattern
	pcl::PointCloud<pcl::PointXYZ> src;
	for (int i = 0; i < M2S1List.size(); i++)
	{
		M2S1 = M2S1List[i];
		M2S2 = M2S2List[i];
		Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > m2s2(M2S2.ptr<float>(), M2S2.rows, M2S2.cols);
		Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > m2s1(M2S1.ptr<float>(), M2S1.rows, M2S1.cols);

		pcl::PointCloud<pcl::PointXYZ> srcsingle;
		pcl::transformPointCloud (pat, srcsingle, m2s1);

		m2s2 = m2s2.inverse();
		pcl::transformPointCloud (srcsingle, srcsingle, m2s2);
		m2s2 = m2s2.inverse();
		src += srcsingle;
		//cout<<"M2S1: \n"<<M2S1<<endl;
		//cout<<"m2s1: \n"<<m2s1<<endl;
	}
	cout<<"size of src is: "<<src.size()<<endl;
	//cout<<"src is: "<<src<<endl;

	//estimate M2M from Pointset1 and Pointset2
	Eigen::Matrix4f trans;
	pcl::registration::TransformationEstimationSVD<pcl::PointXYZ, pcl::PointXYZ> estimater;
	estimater.estimateRigidTransformation (src, tgt, trans);
	//estimater.estimateRigidTransformation (tgt, src, trans);

	Mat M2M;
	eigen2cv(trans,M2M);
	//cout<<"trans: \n"<<trans<<endl;
	//cout<<"M2M: \n"<<M2M<<endl;
	//cout<<"M2M.inv: \n"<<M2M.inv()<<endl;


	//cout<<"M2S2.inv()*M2S1: \n"<<M2S2.inv()*M2S1<<endl;
	//C2T = C2TList[0];
	//C2S = C2SList[0];
	//M2S = M2SList[0];
	//M2T = C2T * C2S.inv() * M2S;
	//cout<<"M2T: \n"<<M2T<<endl;

	//T_{1}2T_{2} = M2T2 * M2M * M2T1^{-1}

	//return M2T1 * M2M * M2T2.inv();
	return M2T2 * M2M.inv() * M2T1.inv();
}

int main (int argc, char* argv[])
{
	//!<check the argc
	if (argc < 5)
	{
		std::cout<<"usage: calculateT2T [supportCamera.yml] [support.yml] [board1.yml] [board2.yml] [sceneNum = support.size()]" <<std::endl;
		//std::cout<<"Make sure you have board.yml under the same path, or you may get Segmentation fault." <<std::endl;
		return 0;
	}

	//!<define file reader
	cv::FileStorage fs;

	//!<load support camera's intrinsic matrix
	cv::Mat supportDistortion;
	cv::Mat supportIntrinsic;
	fs.open(argv[1], cv::FileStorage::READ);
	fs["distortion_coefficients"] >> supportDistortion;
	fs["camera_matrix"] >> supportIntrinsic;
	fs.release();

	//!<load M2T1 and M2T2 matrix
	cv::Mat M2T1, M2T2;
	fs.open(argv[3], cv::FileStorage::READ);
	fs["M2T"] >> M2T1;
	fs.release();

	fs.open(argv[4], cv::FileStorage::READ);
	fs["M2T"] >> M2T2;
	fs.release();

	//!<init chessboard pattern
	float const squareSize = 2.54*2/3;
	cv::Size patternSize = cv::Size(6,6);
	std::vector<cv::Point3f> boardPattern;
	for( int i = 0; i < patternSize.height; i++ )
		for( int j = 0; j < patternSize.width; j++ )
			{
				if (i == j == patternSize.width-1)
				{boardPattern.push_back(cv::Point3f(float(j*(squareSize+2)), float(i*(squareSize+2)), 0));break;}
				boardPattern.push_back(cv::Point3f(float(j*squareSize), float(i*squareSize), 0));
			}

	//!<load images taken by support camera
	//cv::Mat supportImg;
	//supportImg = cv::imread(argv[4]);//, CV_LOAD_IMAGE_GRAYSCALE);
	std::vector<std::string> supportFileList;
	readStringList( argv[2], supportFileList );

	std::vector<cv::Mat> supportImgList;
	for (int i = 0; i < supportFileList.size(); i++)
	{
		cv::Mat supportImg;
		supportImg = cv::imread(supportFileList[i]);
		supportImgList.push_back(supportImg);
	}

	if (argc == 6)
	{
		int sceneNum = atoi(argv[5]);
		std::cout<<"use " <<sceneNum<<" images to estimate." <<std::endl;
		if (sceneNum < supportFileList.size())
			supportImgList.resize(sceneNum);
	}


	//!<detect chessboard in support camera's image
	//std::vector<std::vector<cv::Point2f> > supportCornerList;
	std::vector<cv::Mat> M2SrList1, M2StList1, M2SList1;
	std::vector<cv::Mat> M2SrList2, M2StList2, M2SList2;

	aruco::BoardConfiguration TheBoardConfig1;
	aruco::BoardConfiguration TheBoardConfig2;
	aruco::BoardDetector TheBoardDetector;
	aruco::Board TheBoardDetected1;
	aruco::Board TheBoardDetected2;
	TheBoardConfig1.readFromFile(argv[3]);
	TheBoardConfig2.readFromFile(argv[4]);
	aruco::CameraParameters CamParam(supportIntrinsic, supportDistortion, supportImgList[0].size());
	aruco::MarkerDetector MDetector;
	std::vector< aruco::Marker > Markers;
	float const MarkerSize = 2.54 * 2/3;
	MDetector.setMinMaxSize	( 0.02, 0.5);
	MDetector.setThresholdParams(7, 7);
	MDetector.setThresholdParamRange(2, 0);
	MDetector.setCornerRefinementMethod(aruco::MarkerDetector::SUBPIX);

	//!<init marker pattern
	std::vector<cv::Point3f> markerPattern;
	for (int i = 0; i < TheBoardConfig1.size(); i++)
	{
		for (int j = 0; j < TheBoardConfig1[i].size(); j++)
			{
				cv::Point3f tmp = TheBoardConfig1.at(i)[j];
				tmp.x = tmp.x/200*2.54*2/3;
				tmp.y = tmp.y/200*2.54*2/3;
				tmp.z = tmp.z/200*2.54*2/3;
				markerPattern.push_back(tmp);
			}
	}
	for (int i = 0; i < supportImgList.size(); i++)
	{
		cv::Mat supportImg = supportImgList[i];

		//!<detect border in support camera's image
		//MDetector.detect(supportImg, Markers, CamParam, MarkerSize);
		MDetector.detect(supportImg, Markers);
		float probDetect = TheBoardDetector.detect(Markers, TheBoardConfig1, TheBoardDetected1, CamParam, MarkerSize);
		//aruco::CvDrawingUtils::draw3dAxis(supportImg, TheBoardDetected1, CamParam);
		float probDetect1 = TheBoardDetector.detect(Markers, TheBoardConfig2, TheBoardDetected2, CamParam, MarkerSize);
		//aruco::CvDrawingUtils::draw3dAxis(supportImg, TheBoardDetected2, CamParam);

		if (Markers.size() <= 0)
			{std::cout<<"unable to detect marker in support camera." <<std::endl; return 0;}

		cv::Mat M2S(4,4,CV_32FC(1)), M2St, M2Sr;
		cv::Rodrigues(TheBoardDetected1.Rvec, M2Sr);
		M2St = TheBoardDetected1.Tvec;
		RT2homogeneous<float>(M2S, M2Sr, M2St);
		cv::Rodrigues(M2Sr, M2Sr);
		std::cout<<"M2S1, (r, t):  " << M2Sr <<" " << M2St <<std::endl;
		M2SList1.push_back(M2S);

		cv::Mat M2S2(4,4,CV_32FC(1)), M2St2, M2Sr2;
		cv::Rodrigues(TheBoardDetected2.Rvec, M2Sr2);
		M2St2 = TheBoardDetected2.Tvec;
		RT2homogeneous<float>(M2S2, M2Sr2, M2St2);
		cv::Rodrigues(M2Sr2, M2Sr2);
		std::cout<<"M2S2, (r, t):  " << M2Sr2 <<" " << M2St2 <<std::endl;
		M2SList2.push_back(M2S2);
	}

	//!<calculate T2T matrix
	//we want T2T, so
	//T2T = M2T_2 * M2S_2^{-1} * M2S_1 * M2T_1^{-1}

	//use multi images to find the T2T
	//estimate the T2T by 3D points with information from PnP
	cv::Mat T2T(4,4,CV_32FC(1)), T2Tt, T2Tr;
	T2T = estimateTransformation(M2SList1, M2SList2, M2T1, M2T2, boardPattern);

	cv::Rodrigues(T2T(cv::Range(0, 3),cv::Range(0, 3)), T2Tr);
	T2Tt = T2T(cv::Range(0, 3),cv::Range(3, 4));
	std::cout<<"T2Tr: " << T2Tr <<"\n T2Tt: " << T2Tt <<std::endl;
	std::cout<< "T2T: "<<T2T <<std::endl;

	/*cv::Mat rgt, tgt, T2Tgt(4,4,CV_32FC(1));
	fs.open("R.xml", cv::FileStorage::READ);
	fs["R"] >> rgt;
	fs.release();
	fs.open("T.xml", cv::FileStorage::READ);
	fs["T"] >> tgt;
	fs.release();
	rgt.convertTo(rgt, CV_32F);
	tgt.convertTo(tgt, CV_32F);
	RT2homogeneous<float>(T2Tgt, rgt, tgt);
	cv::Rodrigues(rgt, rgt);
	std::cout<<"rgt: " << rgt <<"\n tgt: " << tgt <<std::endl;

	float rErr, tErr;
	rErr = cv::norm(T2Tr, rgt);
	tErr = cv::norm(T2Tt, tgt);
	float rErrP, tErrP;
	rErrP = rErr / cv::norm(rgt);
	tErrP = tErr / cv::norm(tgt);
	std::cout<<"rErr: " << rErr <<"\n tErr: " << tErr <<std::endl;
	std::cout<<"rErrP: " << rErrP <<"\n tErrP: " << tErrP <<std::endl;

	float theErr = thetaErr(T2Tr, rgt);
	std::cout<<"theErr: " << theErr <<std::endl;*/

	fs.open("T2Tresult.yml", cv::FileStorage::WRITE);
	fs << "T2T" << T2T;
	fs << "T2Tr" << T2Tr;
	fs << "T2Tt" << T2Tt;
	//fs << "rErr" << rErr;
	//fs << "tErr" << tErr;
	//fs << "rErrP" << rErrP;
	//fs << "tErrP" << tErrP;
	//fs << "theErr" << theErr;
	fs.release();

	//!<visulize result on image
	for (int i = 0; i < supportImgList.size(); i++)
	{
		cv::Mat M2S1 = M2SList1[i];
		cv::Mat M2S2 = M2SList2[i];

		cv::Mat T2S1, T2Sr1, T2St1;
		T2S1 = M2S1 * M2T1.inv();
		cv::Rodrigues(T2S1(cv::Range(0, 3),cv::Range(0, 3)), T2Sr1);
		T2St1 = T2S1(cv::Range(0, 3),cv::Range(3, 4));

		cv::Mat T2S2, T2Sr2, T2St2;
		//T2S2 = M2S2 * M2T2.inv();
		T2S2 = T2S1 * T2T.inv();
		cv::Rodrigues(T2S2(cv::Range(0, 3),cv::Range(0, 3)), T2Sr2);
		T2St2 = T2S2(cv::Range(0, 3),cv::Range(3, 4));
		//cout << "point1" << endl;

		cv::Mat pM2S, pM2St, pM2Sr;
		pM2S = T2S2 * M2T2;
		cv::Rodrigues(pM2S(cv::Range(0, 3),cv::Range(0, 3)), pM2Sr);
		pM2St = pM2S(cv::Range(0, 3),cv::Range(3, 4));

		cv::Mat supportImg = supportImgList[i];
		//MDetector.detect(supportImg, Markers, CamParam, MarkerSize);
		aruco::Marker mak = Markers[0];
		mak.id = 125;
		mak.ssize = MarkerSize;
		mak.Tvec = T2St1;
		mak.Rvec = T2Sr1;
		//aruco::CvDrawingUtils::draw3dAxis(supportImg, mak, CamParam);
		char text1[] = "Base camera";
		draw3dCamera(supportImg, mak, CamParam, cv::Scalar(0, 0, 0, 255), text1);
		//cout << "point2" << endl;

		mak.Tvec = T2St2;
		mak.Rvec = T2Sr2;
		//aruco::CvDrawingUtils::draw3dAxis(supportImg, mak, CamParam);
		char text2[] = "Our method";
		draw3dCamera(supportImg, mak, CamParam, cv::Scalar(0, 0, 255, 255), text2);


		/*cv::Mat T2S2gt, T2Sr2gt, T2St2gt;
		T2S2gt = T2S1 * T2Tgt.inv();
		cv::Rodrigues(T2S2gt(cv::Range(0, 3),cv::Range(0, 3)), T2Sr2gt);
		T2St2gt = T2S2gt(cv::Range(0, 3),cv::Range(3, 4));

		mak.Tvec = T2St2gt;
		mak.Rvec = T2Sr2gt;
		//aruco::CvDrawingUtils::draw3dAxis(supportImg, mak, CamParam);
		
		char text3[] = "Stereo calibration";
		//draw3dCamera(supportImg, mak, CamParam, cv::Scalar(255, 0, 0, 255), text3);
		
		{cv::Mat T2S2mir, T2Sr2mir, T2St2mir, T2Tmir;
		T2Tmir = (Mat_<float>(4,4)<<0.18150696, -0.031477101, -0.98288578, -1.6321335,
  -0.21129914, 0.97489429, -0.070241235, 1.341095,
  0.96042073, 0.22043218, 0.17029901, 20.143436,
  0, 0, 0, 1);
		T2S2mir = T2S1 * T2Tmir.inv();
		cv::Rodrigues(T2S2mir(cv::Range(0, 3),cv::Range(0, 3)), T2Sr2mir);
		T2St2mir = T2S2mir(cv::Range(0, 3),cv::Range(3, 4));

		mak.Tvec = T2St2mir;
		mak.Rvec = T2Sr2mir;
		//aruco::CvDrawingUtils::draw3dAxis(supportImg, mak, CamParam);
		char text4[] = "Mirror based";
		draw3dCamera(supportImg, mak, CamParam, cv::Scalar(0, 255, 0, 255), text4);}

		{cv::Mat T2S2mir, T2Sr2mir, T2St2mir, T2Tmir;
		T2Tmir = (Mat_<float>(4,4)<<0.11461379, -0.022947229, 0.99314511, 11.049391,
  0.01836702, 0.99961126, 0.020976849, 0.074738503,
  -0.99324042, 0.015836846, 0.11499074, -6.5499372,
  0, 0, 0, 1);
		T2S2mir = T2S1 * T2Tmir;//.inv();
		cv::Rodrigues(T2S2mir(cv::Range(0, 3),cv::Range(0, 3)), T2Sr2mir);
		T2St2mir = T2S2mir(cv::Range(0, 3),cv::Range(3, 4));

		mak.Tvec = T2St2mir;
		mak.Rvec = T2Sr2mir;
		//aruco::CvDrawingUtils::draw3dAxis(supportImg, mak, CamParam);
		char text5[] = "Mirror based2";
		draw3dCamera(supportImg, mak, CamParam, cv::Scalar(255, 255, 0, 255), text5);}*/

		//draw marker points
		std::vector<cv::Point2f> pmarkerCorners;
		cv::projectPoints(markerPattern, pM2Sr, pM2St,
					supportIntrinsic, supportDistortion, pmarkerCorners);
		cv::drawChessboardCorners(supportImg, patternSize, Mat(pmarkerCorners), false);

		// for each marker, draw info and its boundaries in the image
		//TheBoardDetected2.Rvec = T2Sr2gt;
		//TheBoardDetected2.Tvec = T2St2gt;
		for (unsigned int j = 0; j < TheBoardDetected2.size(); j++)
		{
			//cout << Markers[j] << endl;
			//Markers[j].draw(supportImg, cv::Scalar(0, 0, 255), 2);
			//TheBoardDetected2.draw(supportImg, cv::Scalar(0, 0, 255), 2);
		}
		// draw a 3d cube in each marker if there is 3d info
		if (CamParam.isValid() && MarkerSize != -1)
			for (unsigned int i = 0; i < Markers.size(); i++)
		{
			//aruco::CvDrawingUtils::draw3dCube(supportImg, Markers[i], CamParam);
			//aruco::CvDrawingUtils::draw3dAxis(supportImg, Markers[i], CamParam);
		}

		cv::imshow("supportImg", supportImg);
		char filename[50];
		sprintf( filename, "./supportImg%02d.jpg", i );
		imwrite( filename, supportImg );
		cv::waitKey(0);
	}

}
