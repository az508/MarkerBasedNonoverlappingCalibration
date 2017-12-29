//**************************************
//*New calibration code for increase the accuracy
//*by using multi images from the support camera.
//*Input:	intrinsic matrix of target camera
//*			intrinsic matrix of support camera
//*			image list from target camera
//*			image list from supprot camera
//*Output:	"marker to target camera" transformation matrix
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
//#include <opencv2/core/eigen.hpp>

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

cv::Mat estimateTransformation(std::vector<cv::Mat>& C2SList, std::vector<cv::Mat>& M2SList, std::vector<cv::Mat>& C2TList, std::vector<cv::Point3f>& boardPattern)
{
	//C2T*Point = M2T*M2S^{-1}*C2S*Point
	//maybe better to use PCL function to slove it?
	//OK lets just use PCL funtion to solve it
	//so what we need to do is simply some init and warpping
	
	cv:Mat C2T, C2S, M2S;
	
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
	
	//calculate Pointset1, by C2T*boardPattern
	pcl::PointCloud<pcl::PointXYZ> tgt;
	for (int i = 0; i < C2TList.size(); i++)
	{
		C2T = C2TList[i];
		Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > c2t(C2T.ptr<float>(), C2T.rows, C2T.cols);
		pcl::PointCloud<pcl::PointXYZ> tgtsingle;
		pcl::transformPointCloud (pat, tgtsingle, c2t);
		tgt += tgtsingle;
		
		//cout<<"C2T: \n"<<C2T<<endl;
		//cout<<"c2t: \n"<<c2t<<endl; 
	}
	cout<<"size of tgt is: "<<tgt.size()<<endl;

	
	//calculate Pointset2, by M2S^{-1}*C2S*boardPattern
	pcl::PointCloud<pcl::PointXYZ> src;
	for (int i = 0; i < C2SList.size(); i++)
	{
		C2S = C2SList[i];
		M2S = M2SList[i];
		Mat C2M = M2S.inv() * C2S;
		
		Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > c2m(C2M.ptr<float>(), C2M.rows, C2M.cols);
		
		pcl::PointCloud<pcl::PointXYZ> srcsingle;
		pcl::transformPointCloud (pat, srcsingle, c2m);
		src += srcsingle;
		//cout<<"C2M: \n"<<C2M<<endl;
		//cout<<"c2m: \n"<<c2m<<endl;
	}
	cout<<"size of src is: "<<src.size()<<endl;
	
	//estimate M2T from Pointset1 and Pointset2
	Eigen::Matrix4f trans;
	pcl::registration::TransformationEstimationSVD<pcl::PointXYZ, pcl::PointXYZ> estimater;
	estimater.estimateRigidTransformation (src, tgt, trans);
	
	Mat M2T;
	eigen2cv(trans,M2T);
	//cout<<"trans: \n"<<trans<<endl; 
	//cout<<"M2T: \n"<<M2T<<endl;
	
	
	C2T = C2TList[0];
	C2S = C2SList[0];
	M2S = M2SList[0];
	//M2T = C2T * C2S.inv() * M2S;
	//cout<<"M2T: \n"<<M2T<<endl;
	
	return M2T;
}

int main (int argc, char* argv[])
{
	//!<check the argc
	if (argc < 5)
	{
		std::cout<<"usage: calculateM2T [targetCamera.yml] [supportCamera.yml] [targetImages.yml] [supportImages.yml]" <<std::endl;
		std::cout<<"Make sure you have board.yml under the same path, or you may get Segmentation fault." <<std::endl;
		return 0;
	}
	
	//!<define file reader
	cv::FileStorage fs;
	
	//!<load target camera's intrinsic matrix
	cv::Mat targetDistortion;
	cv::Mat targetIntrinsic;
	fs.open(argv[1], cv::FileStorage::READ);
	fs["distortion_coefficients"] >> targetDistortion;
	fs["camera_matrix"] >> targetIntrinsic;
	fs.release();
	
	//!<load support camera's intrinsic matrix
	cv::Mat supportDistortion;
	cv::Mat supportIntrinsic;
	fs.open(argv[2], cv::FileStorage::READ);
	fs["distortion_coefficients"] >> supportDistortion;
	fs["camera_matrix"] >> supportIntrinsic;
	fs.release();
	
	//!<load images taken by target camera
	//cv::Mat targetImg;
	//targetImg = cv::imread(argv[3]);
	std::vector<std::string> targetFileList;
	readStringList( argv[3], targetFileList );
	
	std::vector<cv::Mat> targetImgList;
	for (int i = 0; i < targetFileList.size(); i++)
	{
		cv::Mat targetImg;
		targetImg = cv::imread(targetFileList[i]);
		targetImgList.push_back(targetImg);
	}
	
	//!<init chessboard pattern
	float const squareSize = 2.54;
	cv::Size patternSize = cv::Size(9,6);
	std::vector<cv::Point3f> boardPattern;
	for( int i = 0; i < patternSize.height; i++ )
		for( int j = 0; j < patternSize.width; j++ )
			{boardPattern.push_back(cv::Point3f(float(j*squareSize), float(i*squareSize), 0));}
			
	//!<detect chessboard in target camera's image
	std::vector<std::vector<cv::Point2f> > targetCornerList;
	std::vector<cv::Mat> C2TrList, C2TtList, C2TList;
	bool found;
	for (int i = 0; i < targetImgList.size(); i++)
	{
		cv::Mat targetImg = targetImgList[i];
		std::vector<cv::Point2f> targetCorners;
		found = cv::findChessboardCorners( targetImg, patternSize, targetCorners, cv::CALIB_CB_ADAPTIVE_THRESH | /*CALIB_CB_FAST_CHECK |*/ cv::CALIB_CB_NORMALIZE_IMAGE);
		if (!found)
			{std::cout<<"unable to find chessboard in target camera." <<std::endl; return 0;}	
		targetCornerList.push_back(targetCorners);
		
		//cv::drawChessboardCorners(targetImg, patternSize, Mat(targetCorners), found);
		//cv::imshow("targetImg", targetImg);
		//cv::waitKey(10);

		cv::Mat C2Tr, C2Tt, C2T(4,4,CV_32FC(1));
		found = solvePnP(boardPattern, targetCorners, targetIntrinsic, targetDistortion, C2Tr, C2Tt);
		if (!found)
			{std::cout<<"unable to slove PnP in target camera." <<std::endl; return 0;}
		std::cout<< C2Tr <<" " << C2Tt <<std::endl;
		C2Tt.convertTo(C2Tt, CV_32F);
		C2Tr.convertTo(C2Tr, CV_32F);
		cv::Rodrigues(C2Tr, C2Tr);
		RT2homogeneous<float>(C2T, C2Tr, C2Tt);
		std::cout<< C2T <<std::endl;
		
		C2TList.push_back(C2T);
	}
	
	//!<load images taken by support camera
	//cv::Mat supportImg;
	//supportImg = cv::imread(argv[4]);//, CV_LOAD_IMAGE_GRAYSCALE);
	std::vector<std::string> supportFileList;
	readStringList( argv[4], supportFileList );
	
	std::vector<cv::Mat> supportImgList;
	for (int i = 0; i < supportFileList.size(); i++)
	{
		cv::Mat supportImg;
		supportImg = cv::imread(supportFileList[i]);
		supportImgList.push_back(supportImg);
	}
	
	//!<detect chessboard in support camera's image
	std::vector<std::vector<cv::Point2f> > supportCornerList;
	std::vector<cv::Mat> C2SrList, C2StList, C2SList;
	std::vector<cv::Mat> M2SrList, M2StList, M2SList;
	
	aruco::BoardConfiguration TheBoardConfig;
	aruco::BoardDetector TheBoardDetector;
	aruco::Board TheBoardDetected;
	TheBoardConfig.readFromFile("board.yml");
	aruco::CameraParameters CamParam(supportIntrinsic, supportDistortion, supportImgList[0].size());
	aruco::MarkerDetector MDetector;
	std::vector< aruco::Marker > Markers;
	float const MarkerSize = 2.54 * 2/3;
	MDetector.setMinMaxSize	( 0.02, 0.5);
	MDetector.setThresholdParams(7, 7);
	MDetector.setThresholdParamRange(2, 0);
	MDetector.setCornerRefinementMethod(aruco::MarkerDetector::SUBPIX);
	for (int i = 0; i < supportImgList.size(); i++)
	{
		cv::Mat supportImg = supportImgList[i];
		std::vector<cv::Point2f> supportCorners;
		found = cv::findChessboardCorners( supportImg, patternSize, supportCorners, cv::CALIB_CB_ADAPTIVE_THRESH | /*CALIB_CB_FAST_CHECK |*/ cv::CALIB_CB_NORMALIZE_IMAGE);
		if (!found)
			{std::cout<<"unable to find chessboard in support camera." <<std::endl; return 0;}
		cv::drawChessboardCorners(supportImg, patternSize, Mat(supportCorners), found);
		//cv::imshow("supportImg", supportImg);
		//cv::waitKey(0);
		
		cv::Mat C2Sr, C2St, C2S(4,4,CV_32FC(1));
		found = solvePnP(boardPattern, supportCorners, supportIntrinsic, supportDistortion, C2Sr, C2St);
		if (!found)
			{std::cout<<"unable to slove PnP in support camera." <<std::endl; return 0;}
		std::cout<< C2Sr <<" " << C2St <<std::endl;
		C2St.convertTo(C2St, CV_32F);
		C2Sr.convertTo(C2Sr, CV_32F);
		cv::Rodrigues(C2Sr, C2Sr);
		RT2homogeneous<float>(C2S, C2Sr, C2St);
		std::cout<< C2S <<std::endl;
		C2SList.push_back(C2S);
			
		//!<detect border in support camera's image
		//MDetector.detect(supportImg, Markers, CamParam, MarkerSize);
		MDetector.detect(supportImg, Markers);
		float probDetect = TheBoardDetector.detect(Markers, TheBoardConfig, TheBoardDetected, CamParam, MarkerSize);
		aruco::CvDrawingUtils::draw3dAxis(supportImg, TheBoardDetected, CamParam);
			
		if (Markers.size() <= 0)
			{std::cout<<"unable to detect marker in support camera." <<std::endl; return 0;}
		
		cv::Mat M2S(4,4,CV_32FC(1)), M2St, M2Sr;
		cv::Rodrigues(TheBoardDetected.Rvec, M2Sr);
		M2St = TheBoardDetected.Tvec;
		RT2homogeneous<float>(M2S, M2Sr, M2St);
		std::cout<< M2Sr <<" " << M2St <<std::endl;
		M2SList.push_back(M2S);
	}
	
	//!<estimate M2T
	//to find M2T we use the following equation
	//M2T = C2T*C2S^{-1}*M2S
	cv::Mat M2T, M2Tr, M2Tt;
	//M2T = C2T * C2S.inv() * M2S;
	//std::cout<<M2T<<std::endl;
	
	//use multi images to find the M2S
	//estimate the M2T by 3D points get from PnP
	M2T = estimateTransformation(C2SList, M2SList, C2TList, boardPattern);
	std::cout<<M2T<<std::endl;
	
	//decompose 4x4 matrix in to R&t vectors
	cv::Rodrigues(M2T(cv::Range(0, 3),cv::Range(0, 3)), M2Tr);
	M2Tt = M2T(cv::Range(0, 3),cv::Range(3, 4));
	
	//write result to file
	fs.open("marker2target.yml", cv::FileStorage::WRITE);
	fs << "M2T" << M2T;
	fs << "M2Tr" << M2Tr;
	fs << "M2Tt" << M2Tt;
	
	{	cv::Mat M2S = M2SList[0];
		cv::Mat C2S = C2SList[0];
	fs << "C2M" << M2S.inv()*C2S;
	}
	fs.release();
	
	fs.open("board.yml", cv::FileStorage::APPEND);
	fs << "M2T" << M2T;
	fs << "M2Tr" << M2Tr;
	fs << "M2Tt" << M2Tt;
	fs.release();
	
	//!<visulize result on image
	for (int i = 0; i < supportImgList.size(); i++)
	{
		cv::Mat M2S = M2SList[i];
		cv::Mat T2S, T2Sr, T2St;
		T2S = M2S * M2T.inv();
		cv::Rodrigues(T2S(cv::Range(0, 3),cv::Range(0, 3)), T2Sr);
		T2St = T2S(cv::Range(0, 3),cv::Range(3, 4));
		
		cv::Mat supportImg = supportImgList[i];
		MDetector.detect(supportImg, Markers, CamParam, MarkerSize);
		aruco::Marker mak = Markers[0];
		mak.Tvec = T2St;
		mak.Rvec = T2Sr;
		mak.id = 125;
		mak.ssize = MarkerSize;
		aruco::CvDrawingUtils::draw3dAxis(supportImg, mak, CamParam);
		//Markers.push_back(mak);
		
		// for each marker, draw info and its boundaries in the image
		for (unsigned int j = 0; j < Markers.size(); j++) 
		{
			//cout << Markers[j] << endl;
			Markers[j].draw(supportImg, cv::Scalar(0, 0, 255), 2);
		}
		// draw a 3d cube in each marker if there is 3d info
		if (CamParam.isValid() && MarkerSize != -1)
			for (unsigned int i = 0; i < Markers.size(); i++) 
		{
			aruco::CvDrawingUtils::draw3dCube(supportImg, Markers[i], CamParam);
			//aruco::CvDrawingUtils::draw3dAxis(supportImg, Markers[i], CamParam);
		}
			
		cv::imshow("supportImg", supportImg);
		char filename[50];
		sprintf( filename, "./supportImg%02d.jpg", i );
		imwrite( filename, supportImg );
		cv::waitKey(0);
	}
	
	//!<estimate reprojection error for estimated M2T
	float sumerr = 0;
	
	//!<project to target camera
	//transform from support to target camera
	//use P_target = M2T*M2S^{-1}*C2S*P_support
	for (int i = 0; i < targetImgList.size(); i++)
	{
		cv::Mat M2S = M2SList[i];
		cv::Mat C2S = C2SList[i];
		cv::Mat pC2T, pC2Tr, pC2Tt;
		pC2T = M2T*M2S.inv()*C2S;
		cv::Rodrigues(pC2T(cv::Range(0, 3),cv::Range(0, 3)), pC2Tr);
		pC2Tt = pC2T(cv::Range(0, 3),cv::Range(3, 4));
		
		std::vector<cv::Point2f> ptargetCorners;
		std::vector<cv::Point2f> targetCorners = targetCornerList[i];
		for (int j = 0; j < boardPattern.size(); j++)
		{
			//projection
			cv::projectPoints(boardPattern, pC2Tr, pC2Tt,
					targetIntrinsic, targetDistortion, ptargetCorners);
			
			//!<check distance to img taken by target camera
			//float err = cv::norm(cv::Mat(targetCorners), cv::Mat(ptargetCorners), cv::NORM_L2);
			float err = cv::norm(cv::Mat(targetCorners[j]), cv::Mat(ptargetCorners[j]), cv::NORM_L1);
			sumerr += err;
			//std::cout<<"reprojection error is "<< err <<std::endl;
		}
		cv::Mat targetImg = targetImgList[i];
		cv::drawChessboardCorners(targetImg, patternSize, Mat(ptargetCorners), true);
		char filename[50];
		sprintf( filename, "./projectImg%02d.jpg", i );
		imwrite( filename, targetImg );
		cv::imshow("projectImg", targetImg);
		cv::waitKey(0);
	}
	
	float avgerr = sumerr / targetImgList.size() / boardPattern.size();
	//float avgerr = sumerr / boardPattern.size();
	std::cout<<"avg reprojection error is "<< avgerr <<std::endl;
	
}
