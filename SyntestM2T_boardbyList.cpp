//**************************************
//**Synthetic test for M2T
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

#include <cstdlib>

using namespace cv;
using namespace std;

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

//*******************************************
//Take 3 by 3 rotation and 1 by 3 translation marix,
//Output homogeneous transformation matrix
//*******************************************
template<typename T>
void RT2homogeneous( cv::Mat& H, cv::Mat& R, cv::Mat& t)
{
	
	H.at<T>(0, 0) = R.at<T>(0, 0);	H.at<T>(0, 1) = R.at<T>(0, 1);	H.at<T>(0, 2) = R.at<T>(0, 2);	H.at<T>(0, 3) = t.at<T>(0);
	H.at<T>(1, 0) = R.at<T>(1, 0);	H.at<T>(1, 1) = R.at<T>(1, 1);	H.at<T>(1, 2) = R.at<T>(1, 2);	H.at<T>(1, 3) = t.at<T>(1);
	H.at<T>(2, 0) = R.at<T>(2, 0);	H.at<T>(2, 1) = R.at<T>(2, 1);	H.at<T>(2, 2) = R.at<T>(2, 2);	H.at<T>(2, 3) = t.at<T>(2);
	H.at<T>(3, 0) = 0			;	H.at<T>(3, 1) = 0			;	H.at<T>(3, 2) = 0			;	H.at<T>(3, 3) = 1			;

};

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

void addnoise (std::vector<cv::Point2f>& target, float stddev)
{
	cv::theRNG().state = cv::getTickCount();
	//std::cout<<"before noise: \n" <<target <<std::endl;
	std::vector<cv::Point2f> noise(target);
	randn(noise, 0, stddev);
	//std::cout<<"noise: \n" <<noise<<std::endl;
	for (int i = 0; i < target.size(); i++)
	{
		//target[i].x = target[i].x;
		target[i] = target[i] + noise[i];
	}
	
	//std::cout<<"after noise: \n" <<target<<std::endl;
};

int main (int argc, char* argv[])
{	
	int sceneNum = atoi(argv[1]);
	FILE* listFile = fopen("synctest_BoardList.txt", "w");
	if (listFile == NULL)
	{
	 printf("Failed to create file in current folder.  Please check permissions.\n");
	 return -1;
	}
	//fprintf(listFile,  "%d scenes used in this test. \n", sceneNum);
	//fprintf(listFile,  "noisedev, avgerr, M2TrErr, M2TtErr, M2TrErrPercent, Exr, Eyr, Ezr, M2TtErrPercent, Ext, Eyt, Ezt, \n");
	
	//!<define file reader
	cv::FileStorage fs;
	
	//!<build target camera's intrinsic matrix
	cv::Mat targetDistortion;
	cv::Mat targetIntrinsic= (Mat_<float>(3,3)<< 1100,0,800,
										0,1100,600,
										0,0,1);

	
	//!<build support camera's intrinsic matrix
	cv::Mat supportDistortion;
	cv::Mat supportIntrinsic = (Mat_<float>(3,3)<<	1100,0,800,
											0,1100,600,
											0,0,1);
	
	//!<init chessboard pattern
	float const squareSize = 2.54;
	cv::Size patternSize = cv::Size(9,6);
	std::vector<cv::Point3f> boardPattern;
	for( int i = 0; i < patternSize.height; i++ )
		for( int j = 0; j < patternSize.width; j++ )
			{boardPattern.push_back(cv::Point3f(float(j*squareSize), float(i*squareSize), 0));}
			
	//!<init marker pattern
	float const markerSize = 2.54 * 2/3;
	cv::Size markerBoardSize = cv::Size(6,6);
	std::vector<cv::Point3f> markerBoardPattern;
	for( int i = 0; i < markerBoardSize.height; i++ )
		for( int j = 0; j < markerBoardSize.width; j++ )
			{markerBoardPattern.push_back(cv::Point3f(float(j*markerSize), float(i*markerSize), 0));}
			
			
	//!<init ground truth
	std::vector<cv::Mat> C2TrgtList, C2TtgtList;
	std::vector<cv::Mat> C2SrgtList, C2StgtList;
	std::vector<cv::Mat> M2TrgtList, M2TtgtList;
	std::vector<cv::Mat> M2SrgtList, M2StgtList;
	
	for (int i = 0; i < sceneNum; i++)
	{
		cv::Mat C2Trgt, C2Ttgt;
		cv::Mat C2Srgt, C2Stgt;
		cv::Mat M2Trgt, M2Ttgt;
		cv::Mat M2Srgt, M2Stgt;
		//init from real data
		char filename[50];
		sprintf( filename, "./groundTruth/marker2target%0d.yml", i );
		
		fs.open(filename, cv::FileStorage::READ);
		fs["C2Tr"] >> C2Trgt;
		fs["C2Tt"] >> C2Ttgt;
		fs["C2Sr"] >> C2Srgt;
		fs["C2St"] >> C2Stgt;
		fs["M2Tr"] >> M2Trgt;
		fs["M2Tt"] >> M2Ttgt;
		fs["M2Sr"] >> M2Srgt;
		fs["M2St"] >> M2Stgt;
		fs.release();
		
		
		if (i == 0)
		{
			//calculate M2T from other value
			//M2T = C2T*C2S^{-1}*M2S
			cv::Mat M2T(4,4,CV_32FC(1));
			cv::Mat C2T(4,4,CV_32FC(1));
			cv::Mat C2S(4,4,CV_32FC(1));
			cv::Mat M2S(4,4,CV_32FC(1));
			//M2Ttgt.convertTo(M2Ttgt, CV_32F);
			//M2Trgt.convertTo(M2Trgt, CV_32F);
			//cv::Rodrigues(M2Trgt, M2Trgt);
			//RT2homogeneous<float>(M2T, M2Trgt, M2Ttgt);
			//cv::Rodrigues(M2Trgt, M2Trgt);
			
			cv::Rodrigues(C2Trgt, C2Trgt);
			RT2homogeneous<float>(C2T, C2Trgt, C2Ttgt);
			cv::Rodrigues(C2Trgt, C2Trgt);
			
			cv::Rodrigues(C2Srgt, C2Srgt);
			RT2homogeneous<float>(C2S, C2Srgt, C2Stgt);
			cv::Rodrigues(C2Srgt, C2Srgt);
			
			cv::Rodrigues(M2Srgt, M2Srgt);
			RT2homogeneous<float>(M2S, M2Srgt, M2Stgt);
			cv::Rodrigues(M2Srgt, M2Srgt);
			
			M2T = C2T * C2S.inv() * M2S;
			cv::Rodrigues(M2T(cv::Range(0, 3),cv::Range(0, 3)), M2Trgt);
			M2Ttgt = M2T(cv::Range(0, 3),cv::Range(3, 4));
		}
		
		if (i != 0)
		{
			//set fixed value
			C2Trgt = C2TrgtList[0];
			C2Ttgt = C2TtgtList[0];
			M2Trgt = M2TrgtList[0];
			M2Ttgt = M2TtgtList[0];
			
			//calculate M2S from other value
			//M2S = C2S * C2T.inv * M2T
			cv::Mat M2T(4,4,CV_32FC(1));
			cv::Mat C2T(4,4,CV_32FC(1));
			cv::Mat C2S(4,4,CV_32FC(1));
			cv::Mat M2S(4,4,CV_32FC(1));
			//M2Ttgt.convertTo(M2Ttgt, CV_32F);
			//M2Trgt.convertTo(M2Trgt, CV_32F);
			cv::Rodrigues(M2Trgt, M2Trgt);
			RT2homogeneous<float>(M2T, M2Trgt, M2Ttgt);
			cv::Rodrigues(M2Trgt, M2Trgt);
			
			cv::Rodrigues(C2Trgt, C2Trgt);
			RT2homogeneous<float>(C2T, C2Trgt, C2Ttgt);
			cv::Rodrigues(C2Trgt, C2Trgt);
			
			cv::Rodrigues(C2Srgt, C2Srgt);
			RT2homogeneous<float>(C2S, C2Srgt, C2Stgt);
			cv::Rodrigues(C2Srgt, C2Srgt);
			
			M2S = C2S * C2T.inv() * M2T;
			cv::Rodrigues(M2S(cv::Range(0, 3),cv::Range(0, 3)), M2Srgt);
			M2Stgt = M2S(cv::Range(0, 3),cv::Range(3, 4));
		}
		
		C2TrgtList.push_back(C2Trgt);
		C2TtgtList.push_back(C2Ttgt);
		C2SrgtList.push_back(C2Srgt);
		C2StgtList.push_back(C2Stgt);
		M2TrgtList.push_back(M2Trgt);
		M2TtgtList.push_back(M2Ttgt);
		M2SrgtList.push_back(M2Srgt);
		M2StgtList.push_back(M2Stgt);
	}

	//init noise stddev
	float noisedev = 0;
	for (noisedev = 0; noisedev<5; noisedev+=0.1)
	{		
		//repeat 10 times to get the avg
		float _M2TrErr = 0, _M2TtErr = 0;
		float _M2TrErrP = 0, _M2TtErrP = 0;
		float _avgerr = 0;
		float _thetaErr = 0;
		
		float _M2SrErr = 0, _M2StErr = 0;
		float _M2SrErrP = 0, _M2StErrP = 0;
		for (int repeat = 0; repeat < 10; repeat++ )
		{			
			std::vector<std::vector<cv::Point2f> > targetCornerList;
			std::vector<cv::Mat> C2TrList, C2TtList, C2TList;
			
			std::vector<std::vector<cv::Point2f> > supportCornerList;
			std::vector<cv::Mat> C2SrList, C2StList, C2SList;
			std::vector<cv::Mat> M2SrList, M2StList, M2SList;
			
			for (int filecnt = 0; filecnt<sceneNum; filecnt++)
			{
				//load gt for each pose
				cv::Mat C2Trgt = C2TrgtList[filecnt], 	C2Ttgt = C2TtgtList[filecnt];
				cv::Mat C2Srgt = C2SrgtList[filecnt],	 C2Stgt = C2StgtList[filecnt];
				cv::Mat M2Trgt = M2TrgtList[filecnt], 	M2Ttgt = M2TtgtList[filecnt];
				cv::Mat M2Srgt = M2SrgtList[filecnt], 	M2Stgt = M2StgtList[filecnt];
				
				//targetImg = cv::imread(argv[3]);//,  CV_LOAD_IMAGE_GRAYSCALE);
			//std::cout<<"checkpoint." <<std::endl;
				//!<project chessboard in target camera's image
				bool found;
				std::vector<cv::Point2f> targetCorners;
				cv::projectPoints(boardPattern, C2Trgt, C2Ttgt, targetIntrinsic, targetDistortion, targetCorners);
				//!<add noise there if needed
				addnoise(targetCorners, noisedev);
				targetCornerList.push_back(targetCorners);
			//std::cout<<"checkpoint." <<std::endl;	
					
				cv::Mat C2Tr, C2Tt, C2T(4,4,CV_32FC(1));
				found = solvePnP(boardPattern, targetCorners, targetIntrinsic, targetDistortion, C2Tr, C2Tt);
				if (!found)
					{std::cout<<"unable to slove PnP in target camera." <<std::endl; return 0;}
				//std::cout<< C2Tr <<" " << C2Tt <<std::endl;
				//std::cout<<"C2Tr: "<< C2Tr.t() <<std::endl;
				//std::cout<<"C2Tt: "<< C2Tt.t() <<std::endl;
				//std::cout<<"Ground truth C2Tr: "<< C2Trgt.t() <<std::endl;
				//std::cout<<"Ground truth C2Tt: "<< C2Ttgt.t() <<std::endl;
					
				C2Tt.convertTo(C2Tt, CV_32F);
				C2Tr.convertTo(C2Tr, CV_32F);
				cv::Rodrigues(C2Tr, C2Tr);
				RT2homogeneous<float>(C2T, C2Tr, C2Tt);
				C2TList.push_back(C2T);
				//std::cout<< C2T <<std::endl;

				//supportImg = cv::imread(argv[4]);//, CV_LOAD_IMAGE_GRAYSCALE);
			//std::cout<<"checkpoint." <<std::endl;
				//!<detect chessboard in support camera's image
				std::vector<cv::Point2f> supportCorners;
				cv::projectPoints(boardPattern, C2Srgt, C2Stgt, supportIntrinsic, supportDistortion, supportCorners);
				//!<add noise there if needed
				addnoise(supportCorners, noisedev);
				supportCornerList.push_back(supportCorners);
			//std::cout<<"checkpoint." <<std::endl;
				
				cv::Mat C2Sr, C2St, C2S(4,4,CV_32FC(1));
				found = solvePnP(boardPattern, supportCorners, supportIntrinsic, supportDistortion, C2Sr, C2St);
				if (!found)
					{std::cout<<"unable to slove PnP in support camera." <<std::endl; return 0;}
				//std::cout<< C2Sr <<" " << C2St <<std::endl;
				//std::cout<<"C2Sr: "<< C2Sr.t() <<std::endl;
				//std::cout<<"C2St: "<< C2St.t() <<std::endl;
				//std::cout<<"Ground truth C2Sr: "<< C2Srgt.t() <<std::endl;
				//std::cout<<"Ground truth C2St: "<< C2Stgt.t() <<std::endl;
					
				C2St.convertTo(C2St, CV_32F);
				C2Sr.convertTo(C2Sr, CV_32F);
				cv::Rodrigues(C2Sr, C2Sr);
				RT2homogeneous<float>(C2S, C2Sr, C2St);
				C2SList.push_back(C2S);
				//std::cout<< C2S <<std::endl;

					
			//std::cout<<"checkpoint." <<std::endl;	
				//!<project markerboard in support camera's image
				std::vector<cv::Point2f> markerCorners;
				cv::Mat M2Sr, M2St, M2S(4,4,CV_32FC(1));
				cv::projectPoints(markerBoardPattern, M2Srgt, M2Stgt, supportIntrinsic, supportDistortion, markerCorners);
				//!<add noise there if needed
				addnoise(markerCorners, noisedev);
			//std::cout<<"checkpoint." <<std::endl;
					
				found = solvePnP(markerBoardPattern, markerCorners, supportIntrinsic, supportDistortion, M2Sr, M2St);
				if (!found)
					{std::cout<<"unable to slove PnP in support camera." <<std::endl; return 0;}
				//std::cout<< C2Sr <<" " << C2St <<std::endl;
				//std::cout<<"M2Sr: "<< M2Sr.t() <<std::endl;
				//std::cout<<"M2St: "<< M2St.t() <<std::endl;
				//std::cout<<"Ground truth M2Sr: "<< M2Srgt.t() <<std::endl;
				//std::cout<<"Ground truth M2St: "<< M2Stgt.t() <<std::endl;
					
				M2St.convertTo(M2St, CV_32F);
				M2Sr.convertTo(M2Sr, CV_32F);
				cv::Rodrigues(M2Sr, M2Sr);
				RT2homogeneous<float>(M2S, M2Sr, M2St);
				M2SList.push_back(M2S);
				cv::Rodrigues(M2Sr, M2Sr);
					
				float M2SrErr, M2StErr;
				M2SrErr = cv::norm(M2Sr, M2SrgtList[filecnt]);
				M2StErr = cv::norm(M2St, M2StgtList[filecnt]);
				_M2SrErr += M2SrErr;
				_M2StErr += M2StErr;
					
				float M2SrErrP, M2StErrP;
				M2SrErrP = M2SrErr/cv::norm(M2SrgtList[filecnt]);
				M2StErrP = M2StErr/cv::norm(M2StgtList[filecnt]);
				_M2SrErrP += M2SrErrP;
				_M2StErrP += M2StErrP;
			}
			
			
			//!<estimate M2T
			//to find M2T we use the following equation
			//M2T = C2T*C2S^{-1}*M2S
			cv::Mat M2T, M2Tr, M2Tt;
			//M2T = C2T * C2S.inv() * M2S;
			M2T = estimateTransformation(C2SList, M2SList, C2TList, boardPattern);
			//std::cout<<M2T<<std::endl;
			
			//decompose 4x4 matrix in to R&t vectors
			cv::Rodrigues(M2T(cv::Range(0, 3),cv::Range(0, 3)), M2Tr);
			M2Tt = M2T(cv::Range(0, 3),cv::Range(3, 4));
			//std::cout<<"M2Tr: "<< M2Tr.t() <<std::endl;
			//std::cout<<"M2Tt: "<< M2Tt.t() <<std::endl;
			//std::cout<<"Ground truth M2Tr: "<< M2Trgt.t() <<std::endl;
			//std::cout<<"Ground truth M2Tt: "<< M2Ttgt.t() <<std::endl;
			
			
			//!<estimate reprojection error for estimated M2T
			float sumerr = 0;
			//!<project to target camera
			//transform from support to target camera
			//use P_target = M2T*M2S^{-1}*C2S*P_support
			for (int i = 0; i < targetCornerList.size(); i++)
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
				//cv::Mat targetImg = targetImgList[i];
				//cv::drawChessboardCorners(targetImg, patternSize, Mat(ptargetCorners), true);
				//char filename[50];
				//sprintf( filename, "./projectImg%02d.jpg", i );
				//imwrite( filename, targetImg );
				//cv::imshow("projectImg", targetImg);
				//cv::waitKey(0);
			}
			
			float avgerr = sumerr / targetCornerList.size() / boardPattern.size();
			//float avgerr = sumerr / boardPattern.size();
			std::cout<<"avg reprojection error is "<< avgerr <<std::endl;
			_avgerr += avgerr;
			
			//write result to file
			/*fs.open("synctest.yml", cv::FileStorage::WRITE);
			fs << "M2T" << M2T;
			fs << "M2Tr" << M2Tr;
			fs << "M2Tt" << M2Tt;
			fs << "avgerr" << avgerr;
			fs.release();*/
			
			//try to calculate the "percentage error"
			float M2TrErr, M2TtErr;
			M2TrErr = cv::norm(M2Tr, M2TrgtList[0]);
			M2TtErr = cv::norm(M2Tt, M2TtgtList[0]);
			_M2TrErr += M2TrErr;
			_M2TtErr += M2TtErr;
			
			
			//releative error in translation vector
			//releative error in rotation vector
			float M2TrErrP, M2TtErrP;
			M2TrErrP = M2TrErr/cv::norm(M2TrgtList[0]);
			M2TtErrP = M2TtErr/cv::norm(M2TtgtList[0]);
			
			_M2TrErrP += M2TrErrP;
			_M2TtErrP += M2TtErrP;
			
			float theErr = thetaErr(M2Tr, M2TrgtList[0]);
			_thetaErr += theErr;
		}
		_avgerr = _avgerr/10;
		std::cout<<"_avgerr is "<< _avgerr <<std::endl;
		
		_thetaErr = _thetaErr / 10;
		std::cout<<"_thetaErr is "<< _thetaErr <<std::endl;
		
		_M2TrErr = _M2TrErr/10;
		_M2TtErr = _M2TtErr/10;
		std::cout<<"_M2TrErr is "<< _M2TrErr <<std::endl;
		std::cout<<"_M2TtErr is "<< _M2TtErr <<std::endl;
		
		_M2TrErrP = _M2TrErrP/10;
		_M2TtErrP = _M2TtErrP/10;
		std::cout<<"percentage error _M2Tr is "<< _M2TrErrP<<std::endl;
		std::cout<<"percentage error _M2Tt is "<< _M2TtErrP<<std::endl;
		
		
		_M2SrErr = _M2SrErr/(10*sceneNum);
		_M2StErr = _M2StErr/(10*sceneNum);
		_M2SrErrP = _M2SrErrP/(10*sceneNum);
		_M2StErrP = _M2StErrP/(10*sceneNum);
		/*float M2TrErrPx, M2TrErrPy, M2TrErrPz;
		float M2TtErrPx, M2TtErrPy, M2TtErrPz;
		M2TrErrPx = abs(M2Tr.at<float>(0) - (M2TrgtList[0]).at<float>(0) );
		M2TrErrPy = abs(M2Tr.at<float>(1) - (M2TrgtList[0]).at<float>(1) );
		M2TrErrPz = abs(M2Tr.at<float>(2) - (M2TrgtList[0]).at<float>(2) );
		M2TtErrPx = abs(M2Tt.at<float>(0) - (M2TtgtList[0]).at<float>(0) );
		M2TtErrPy = abs(M2Tt.at<float>(1) - (M2TtgtList[0]).at<float>(1) );
		M2TtErrPz = abs(M2Tt.at<float>(2) - (M2TtgtList[0]).at<float>(2) );*/
		//fprintf(listFile,  "%f %f %f %f %f %f %f %f %f %f %f %f\n", noisedev, avgerr, M2TrErr, M2TtErr, M2TrErrP, M2TrErrPx, M2TrErrPy, M2TrErrPz, M2TtErrP, M2TtErrPx, M2TtErrPy, M2TtErrPz);
		fprintf(listFile,  "%f %f %f %f %f %f %f %f %f %f %f\n", noisedev, _avgerr, _M2TrErr, _M2TtErr, _M2TrErrP, _M2TtErrP, _M2SrErr, _M2StErr, _M2SrErrP, _M2StErrP, _thetaErr);

		std::cout<<"at noise: "<< noisedev <<"pixel.\n" <<std::endl;
	}
	fclose(listFile);
}
