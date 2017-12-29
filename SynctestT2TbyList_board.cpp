//**************************************
//*Synthetic test for T2T
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
#include <math.h> 
using namespace cv;
using namespace std;

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
	
	//cout<< cv::trace(Rp) <<"  "<<tr<<endl;
	
	return acos(( tr -1)/2);
};

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
	//cout<<"size of tgt is: "<<tgt.size()<<endl;
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
		src += srcsingle;
		//cout<<"M2S1: \n"<<M2S1<<endl;
		//cout<<"m2s1: \n"<<m2s1<<endl;
	}
	//cout<<"size of src is: "<<src.size()<<endl;
	//cout<<"src is: "<<src<<endl;
	
	//estimate M2M from Pointset1 and Pointset2
	Eigen::Matrix4f trans;
	pcl::registration::TransformationEstimationSVD<pcl::PointXYZ, pcl::PointXYZ> estimater;
	estimater.estimateRigidTransformation (src, tgt, trans);
	//estimater.estimateRigidTransformation (tgt, src, trans);
	
	Mat M2M, M2Mr, M2Mt;
	eigen2cv(trans,M2M);
	//cout<<"trans: \n"<<trans.inverse()<<endl; 
	//cout<<"M2M: \n"<<M2M<<endl;
	//cout<<"M2M.inv: \n"<<M2M.inv()<<endl;
	M2M = M2M.inv();
	
			//cv::Rodrigues(M2M(cv::Range(0, 3),cv::Range(0, 3)), M2Mr);
			//M2Mt = M2M(cv::Range(0, 3),cv::Range(3, 4));
			//std::cout<<"M2Mr: "<< M2Mr.t()<<std::endl;
			//std::cout<<"M2Mt: "<< M2Mt.t() <<std::endl;
	

	//cout<<"M2S2.inv()*M2S1: \n"<<M2S2.inv()*M2S1<<endl;
	//C2T = C2TList[0];
	//C2S = C2SList[0];
	//M2S = M2SList[0];
	//M2T = C2T * C2S.inv() * M2S;
	//cout<<"M2T: \n"<<M2T<<endl;
	
	//T_{1}2T_{2} = M2T2 * M2M * M2T1^{-1}
	
	//return M2T1 * M2M * M2T2.inv();
	return M2T2 * M2M * M2T1.inv();
}

int main (int argc, char* argv[])
{
	//!<check the argc
	if (argc < 2)
	{
		std::cout<<"usage: synctestT2T [sceneNum]" <<std::endl;
		return 0;
	}
	
	int sceneNum = atoi(argv[1]);
	FILE* listFile = fopen("synctest_T2TList.txt", "w");
	if (listFile == NULL)
	{
	 printf("Failed to create file in current folder.  Please check permissions.\n");
	 return -1;
	}
	
	//int sceneNum = atoi(argv[1]);
	
	//!<define file reader
	cv::FileStorage fs;
	
	//!<build support camera's intrinsic matrix
	cv::Mat supportDistortion;
	cv::Mat supportIntrinsic = (Mat_<float>(3,3)<<	1100,0,800,
													0,1100,600,
													0,0,1);
	
	//!<build M2T1 and M2T2 matrix
	cv::Mat M2T1, M2T2;
	//fs.open(argv[3], cv::FileStorage::READ);
	//fs["M2T"] >> M2T1;
	//fs.release();

	//fs.open(argv[4], cv::FileStorage::READ);
	//fs["M2T"] >> M2T2;
	//fs.release();
	
	//!<init ground truth
	std::vector<cv::Mat> M2Srgt1List, M2Stgt1List;
	std::vector<cv::Mat> M2Srgt2List, M2Stgt2List;
	std::vector<cv::Mat> M2MrgtList, M2MtgtList;
	std::vector<cv::Mat> T2TrgtList, T2TtgtList;
	for (int i = 0; i < sceneNum; i++)
	{
		cv::Mat M2Mrgt, M2Mtgt;
		cv::Mat T2Trgt, T2Ttgt;
		cv::Mat M2Srgt1, M2Stgt1;
		cv::Mat M2Srgt2, M2Stgt2;
		
		//init from real data
		char filename[50];
		sprintf( filename, "./T2TGroundTruth/T2Tresult%0d.yml", i );
		
		fs.open(filename, cv::FileStorage::READ);
		//fs["T2Tr"] >> T2Trgt;
		//fs["T2Tt"] >> T2Ttgt;
		fs["M2Sr1"] >> M2Srgt1;
		fs["M2St1"] >> M2Stgt1;
		fs["M2Sr2"] >> M2Srgt2;
		fs["M2St2"] >> M2Stgt2;
		
		cv::Mat M2M(4,4,CV_32FC(1));
		cv::Mat T2T(4,4,CV_32FC(1));
		cv::Mat M2S1(4,4,CV_32FC(1));
		cv::Mat M2S2(4,4,CV_32FC(1));
		
		cv::Rodrigues(M2Srgt1, M2Srgt1);
		RT2homogeneous<float>(M2S1, M2Srgt1, M2Stgt1);
		cv::Rodrigues(M2Srgt1, M2Srgt1);
		
		if (i ==0)
		{
			fs["M2T1"] >> M2T1;
			fs["M2T2"] >> M2T2;
			
			cv::Rodrigues(M2Srgt2, M2Srgt2);
			RT2homogeneous<float>(M2S2, M2Srgt2, M2Stgt2);
			cv::Rodrigues(M2Srgt2, M2Srgt2);
			
			M2M = M2S2.inv() * M2S1;
			cv::Rodrigues(M2M(cv::Range(0, 3),cv::Range(0, 3)), M2Mrgt);
			M2Mtgt = M2M(cv::Range(0, 3),cv::Range(3, 4));
			
			//cout<<M2Stgt2.t()<<endl;
			//cout<<M2S2<<endl;
			//clalculate T2T by M2M 
			//T2T = M2T_2 * M2S_2^{-1} * M2S_1 * M2T_1^{-1}
			T2T = M2T2 * M2S2.inv() * M2S1 * M2T1.inv();
			cv::Rodrigues(T2T(cv::Range(0, 3),cv::Range(0, 3)), T2Trgt);
			T2Ttgt = T2T(cv::Range(0, 3),cv::Range(3, 4));
		}

		
		//M2MrgtList.push_back(M2Mrgt);
		//M2MtgtList.push_back(M2Mtgt);

		fs.release();
		
		if (i != 0)
		{
			//M2M is fixed
			M2Mrgt = M2MrgtList[0];
			M2Mtgt = M2MtgtList[0];
			
			cv::Rodrigues(M2Mrgt, M2Mrgt);
			RT2homogeneous<float>(M2M, M2Mrgt, M2Mtgt);
			cv::Rodrigues(M2Mrgt, M2Mrgt);
			
			//T2T is fixed
			T2Trgt = T2TrgtList[0];
			T2Ttgt = T2TtgtList[0];
			
			//use real M2S1
			//already loaded
			
			//Calculate M2S2 (M_2ToS)
			//M_2ToS = M_1ToS * M_1ToM_2.inv
			M2S2 = M2S1 * M2M.inv();
			cv::Rodrigues(M2S2(cv::Range(0, 3),cv::Range(0, 3)), M2Srgt2);
			M2Stgt2 = M2S2(cv::Range(0, 3),cv::Range(3, 4));
			
			//cout<<M2Mtgt.t()<<endl;
			//cout<<M2M<<endl;
		}
		
		M2MrgtList.push_back(M2Mrgt);
		M2MtgtList.push_back(M2Mtgt);
		T2TrgtList.push_back(T2Trgt);
		T2TtgtList.push_back(T2Ttgt);
		
		M2Srgt1List.push_back(M2Srgt1);
		M2Stgt1List.push_back(M2Stgt1);
		M2Srgt2List.push_back(M2Srgt2);
		M2Stgt2List.push_back(M2Stgt2);
	}
	
	/*for (int i  = 0; i < sceneNum; i++)
	{
		//cout<<M2Srgt1List[i]<<endl;
		//cout<<M2Stgt1List[i]<<endl;
		//cout<<M2Srgt2List[i]<<endl;
		//cout<<M2Stgt2List[i]<<endl;
		cout<<M2MrgtList[i].t()<<endl;
		cout<<M2MtgtList[i].t()<<endl;
	}*/
	
	//return 0;
	
	//!<init marker pattern
	float const markerSize = 2.54 * 2/3;
	cv::Size markerBoardSize = cv::Size(6,6);
	std::vector<cv::Point3f> markerBoardPattern;
	for( int i = 0; i < markerBoardSize.height; i++ )
		for( int j = 0; j < markerBoardSize.width; j++ )
			{markerBoardPattern.push_back(cv::Point3f(float(j*markerSize), float(i*markerSize), 0));}
			
	/*aruco::BoardConfiguration TheBoardConfig;
	TheBoardConfig.readFromFile("board.yml");
	markerBoardPattern.clear();
	for (int i = 0 ; i < TheBoardConfig.size(); i++)
	{
		for (int j = 0; j < 4; j++)
		{
			markerBoardPattern.push_back(TheBoardConfig.at(i).at(j));
		}
	}*/
	
	
	
	//init noise stddev
	float noisedev = 0;
	for (noisedev = 0; noisedev<5.0; noisedev+=0.1)
	{
		//repeat 10 times to get the avg
		float _T2TrErr = 0, _T2TtErr = 0;
		float _T2TrErrP = 0, _T2TtErrP = 0;
		float _T2TthetaErr = 0;
		
		for (int repeat = 0; repeat < 10; repeat++ )
		{
			//!<detect chessboard in support camera's image
			//std::vector<std::vector<cv::Point2f> > supportCornerList;
			std::vector<cv::Mat> M2SrList1, M2StList1, M2SList1;
			std::vector<cv::Mat> M2SrList2, M2StList2, M2SList2;
			

			for (int i = 0; i < sceneNum; i++)
			{
				cv::Mat M2Srgt1 = M2Srgt1List[i], M2Stgt1 = M2Stgt1List[i];
				cv::Mat M2Srgt2 = M2Srgt2List[i], M2Stgt2 = M2Stgt2List[i];
				
				//!<detect border in support camera's image
				std::vector<cv::Point2f> markerCorners1;
				cv::projectPoints(markerBoardPattern, M2Srgt1, M2Stgt1, supportIntrinsic, supportDistortion, markerCorners1);
				//!<add noise there if needed
				addnoise(markerCorners1, noisedev);
				
				std::vector<cv::Point2f> markerCorners2;
				cv::projectPoints(markerBoardPattern, M2Srgt2, M2Stgt2, supportIntrinsic, supportDistortion, markerCorners2);
				//!<add noise there if needed
				addnoise(markerCorners2, noisedev);
				
				cv::Mat M2S1(4,4,CV_32FC(1)), M2St1, M2Sr1;
				bool found = solvePnP(markerBoardPattern, markerCorners1, supportIntrinsic, supportDistortion, M2Sr1, M2St1);
				if (!found)
					{std::cout<<"unable to slove PnP in support camera." <<std::endl; return 0;}
					
			//std::cout<<"M2Sr1: "<< M2Sr1.t() <<std::endl;
			//std::cout<<"M2St1: "<< M2St1.t() <<std::endl;
			//std::cout<<"Ground truth M2Sr1: "<< M2Srgt1.t() <<std::endl;
			//std::cout<<"Ground truth M2St1: "<< M2Stgt1.t() <<std::endl;
				M2St1.convertTo(M2St1, CV_32F);
				M2Sr1.convertTo(M2Sr1, CV_32F);
				cv::Rodrigues(M2Sr1, M2Sr1);
				RT2homogeneous<float>(M2S1, M2Sr1, M2St1);
				cv::Rodrigues(M2Sr1, M2Sr1);
				M2SList1.push_back(M2S1);
				
				cv::Mat M2S2(4,4,CV_32FC(1)), M2St2, M2Sr2;
				found = solvePnP(markerBoardPattern, markerCorners2, supportIntrinsic, supportDistortion, M2Sr2, M2St2);
				if (!found)
					{std::cout<<"unable to slove PnP in support camera." <<std::endl; return 0;}
			//std::cout<<"M2Sr2: "<< M2Sr2.t() <<std::endl;
			//std::cout<<"M2St2: "<< M2St2.t() <<std::endl;
			//std::cout<<"Ground truth M2Sr2: "<< M2Srgt2.t() <<std::endl;
			//std::cout<<"Ground truth M2St2: "<< M2Stgt2.t() <<std::endl;
				M2St2.convertTo(M2St2, CV_32F);
				M2Sr2.convertTo(M2Sr2, CV_32F);
				cv::Rodrigues(M2Sr2, M2Sr2);
				RT2homogeneous<float>(M2S2, M2Sr2, M2St2);
				cv::Rodrigues(M2Sr2, M2Sr2);
				M2SList2.push_back(M2S2);
					
				//std::cout<<"M2Sr1: " << M2Sr1 <<"\n M2St1: " << M2St1 <<std::endl;
				//std::cout<<"M2Srgt1: " << M2Srgt1 <<"\n M2Stgt1: " << M2Stgt1 <<std::endl;
			}
			
			//!<calculate T2T matrix
			//we want T2T, so
			//T2T = M2T_2 * M2S_2^{-1} * M2S_1 * M2T_1^{-1}
			
			//use multi images to find the T2T
			//estimate the T2T by 3D points with information from PnP
			cv::Mat T2T(4,4,CV_32FC(1)), T2Tt, T2Tr;
			T2T = estimateTransformation(M2SList1, M2SList2, M2T1, M2T2, markerBoardPattern);
			
			//std::cout<<"Ground truth M2Mr: "<< M2MrgtList[0].t()<<std::endl;
			//std::cout<<"Ground truth M2Mt: "<< M2MtgtList[0].t() <<std::endl;
			
			cv::Rodrigues(T2T(cv::Range(0, 3),cv::Range(0, 3)), T2Tr);
			T2Tt = T2T(cv::Range(0, 3),cv::Range(3, 4));
			//std::cout<<"T2Tr: " << T2Tr.t() <<"\n T2Tt: " << T2Tt.t() <<std::endl;
			//std::cout<< "T2T: "<<T2T <<std::endl;
			
			//std::cout<<"T2TrgtList[0]: " << T2TrgtList[0].t() <<"\n T2TtgtList[0]: " << T2TtgtList[0].t() <<std::endl;
			//std::cout<< "T2T: "<<T2T <<std::endl;
			
			
			//!<estimate error of T2T matrix
			float T2TrErr, T2TtErr;
			T2TrErr = cv::norm(T2Tr, T2TrgtList[0]);
			T2TtErr = cv::norm(T2Tt, T2TtgtList[0]);
			_T2TrErr += T2TrErr;
			_T2TtErr += T2TtErr;
				
			float T2TrErrP, T2TtErrP;
			T2TrErrP = T2TrErr/cv::norm(T2TrgtList[0]);
			T2TtErrP = T2TtErr/cv::norm(T2TtgtList[0]);
			_T2TrErrP += T2TrErrP;
			_T2TtErrP += T2TtErrP;
			
			float T2TthetaErr = thetaErr(T2Tr, T2TrgtList[0]);
			_T2TthetaErr += T2TthetaErr;
		}
		_T2TthetaErr = _T2TthetaErr/10;
		std::cout<<"_T2TthetaErr is "<< _T2TthetaErr <<std::endl;
		
		_T2TrErr = _T2TrErr/10;
		_T2TtErr = _T2TtErr/10;
		std::cout<<"_T2TrErr is "<< _T2TrErr <<std::endl;
		std::cout<<"_T2TtErr is "<< _T2TtErr <<std::endl;
		
		_T2TrErrP = _T2TrErrP/10;
		_T2TtErrP = _T2TtErrP/10;
		std::cout<<"percentage error _T2Tr is "<< _T2TrErrP<<std::endl;
		std::cout<<"percentage error _T2Tt is "<< _T2TtErrP<<std::endl;
		
		fprintf(listFile,  "%f %f %f %f %f %f\n", noisedev, _T2TrErr, _T2TtErr, _T2TrErrP, _T2TtErrP, _T2TthetaErr);

		std::cout<<"at noise: "<< noisedev <<"pixel.\n" <<std::endl;
	}
	fclose(listFile);	
}
