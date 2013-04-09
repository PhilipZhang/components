
#include <iostream>
#include <vector>
#include <fstream>
#include <map>
#include <algorithm>
#include <cmath>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\opencv.hpp>
#include "FeaturePoints.h"
using namespace std;
using namespace cv;

// global variables
vector<string> nameList;			// the name of images
vector<IplImage*> imageList;		// the IplImage* of each image
unsigned itimes = 5;				
unsigned nImgs = 0;					// number of images
const unsigned nMaxCorners = 8000;	// maximum number of corners
unsigned win_size = 64;
unsigned nDescriptors = 128;
map<pair<string, string>, vector<pair<int, int> > > pair_matches;  // the data structure to map the image pair to the feature matches
//CvPoint2D32f **corners = 0;			// several arrays to hold corners of each image
//int *countArr = 0;					// hold the actual count of corners of each image
vector<vector<CvPoint2D32f> > corners;
vector<int> countArr;
vector<int> prevCount;

bool operator ==(const CvPoint2D32f& lhs, const CvPoint2D32f& rhs)
{
	return lhs.x == rhs.x && lhs.y == rhs.y;
}

string genSiftName(const string& imgName)
{
	return imgName.substr(0, imgName.find('.')) + ".sift";
}

void readImageList(const char* filename)
{
	// read the image list into memory...
	fstream ifs(filename, ios_base::in);
	char imgName[64];
	while(ifs.getline(imgName, 64))
	{
		nameList.push_back(imgName);		
		imageList.push_back(cvLoadImage(imgName, CV_LOAD_IMAGE_GRAYSCALE));
	}
	ifs.close();

	nImgs = nameList.size();
	
	/*
	corners = new CvPoint2D32f*[nImgs];
	for(int i = 0; i < nImgs; i++)
		corners[i] = new CvPoint2D32f[nMaxCorners];
	countArr = new int[nImgs];
	for(int i = 0; i < nImgs; i++)
		countArr[i] = nMaxCorners;
		*/
	vector<CvPoint2D32f> tmp(8000, CvPoint2D32f());	
	for(int i = 0; i < nImgs; i++)
	{
		corners.push_back(tmp);
	}
	for(int i = 0; i < nImgs; i++)
		countArr.push_back(nMaxCorners);
}

void findGoodFeatures()
{
	// for each image, we find the good features of it using corners	
	IplImage* tmpImage;
	IplImage* tmpImage2;
	
	for(int i = 0; i < imageList.size(); i++)
	{
		tmpImage = cvCreateImage(cvGetSize(imageList[i]), IPL_DEPTH_32F, 1);
		tmpImage2 = cvCreateImage(cvGetSize(imageList[i]), IPL_DEPTH_32F, 1);
		cvGoodFeaturesToTrack(imageList[i], tmpImage, tmpImage2, &corners[i][0], &countArr[i], 0.01, 5);
		corners[i].resize(countArr[i]);
		/*
		cvFindCornerSubPix(imageList[i], &corners[i][0], countArr[i], cvSize(win_size, win_size), cvSize(-1, -1),
			cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03));
			*/
		cout << nameList[i] << " has found " << countArr[i] << " initial corners." << endl;
	}	
	//tmpImage = cvCreateImage(cvGetSize(imageList[0]), IPL_DEPTH_32F, 1);
	//tmpImage2 = cvCreateImage(cvGetSize(imageList[0]), IPL_DEPTH_32F, 1);
	//cvGoodFeaturesToTrack(imageList[0], tmpImage, tmpImage2, &corners[0][0], &countArr[0], 0.01, 5);
	//cout << nameList[0] << " has found " << countArr[0] << " corners." << endl;
	cvReleaseImage(&tmpImage);
	cvReleaseImage(&tmpImage2);
}

void conductMatching()
{
	// for image i, sequence j,j+1....count-1 will try to match it and generate the pair-matches map
	int iHalf = nameList.size() / 2;
	char *status = new char[nMaxCorners];
	float *errors = new float[nMaxCorners];
	for(int i = 0; i < nImgs; i++)
		prevCount.push_back(0);
	for(int i = 0; i < nImgs - 1; i++)
	{
		for(int j = i + 1; j < nImgs; j++)
		{
			if(prevCount[i] < iHalf)
				prevCount[i] ++;
			else
				prevCount[j] ++;
			cout << "Matching " << nameList[i] << " and " << nameList[j] << "..." << endl;
			vector<pair<int, int> > vec;
			CvPoint2D32f *dest_corners = new CvPoint2D32f[nMaxCorners];
			int prev, next;
			// determine whether to add features in image i or image j, according to their appearance count of precedings
			if(prevCount[j] == 0)
			{
				prev = i; 
				next = j;				
			}
			else
			{
				prev = j;
				next = i;
			}
			cvCalcOpticalFlowPyrLK(imageList[prev], imageList[next], NULL, NULL, &corners[prev][0], dest_corners, countArr[prev],
				cvSize(win_size, win_size), itimes, status, errors, cvTermCriteria(CV_TERMCRIT_ITER, 5, 0.1), 0);
			for(int k = 0; k < countArr[prev]; k++)
			{
				if(status[k] == 1 && errors[k] <= 550)
				{						
					vec.push_back(pair<int, int>(k, corners[next].size()));
					corners[next].push_back(dest_corners[k]);
				}
			}	
			pair_matches[pair<string, string>(nameList[i], nameList[j])] = vec; 		
		}
		
		// use PyrLK to find matches.
		
	}
	delete [] status;
	delete [] errors;
	/*
	for(int i = 0; i < nImgs; i++)
	{
		for(int j = i + 1; j < nImgs; j++)
		{
			cout << "Matching " << nameList[i] << " and " << nameList[j] << "..." << endl;
			// use PyrLK to find matches.
			cvCalcOpticalFlowPyrLK(imageList[i], imageList[j], NULL, NULL, &corners[i][0], dest_corners, nMaxCorners,
			cvSize(64,64), itimes, status, NULL, cvTermCriteria(CV_TERMCRIT_ITER, 5, 0.1), 0);
			vector<pair<int, int> > vec;			
			for(int k = 0; k < min((double)countArr[i], (double)nMaxCorners); k++)
			{
				if(status[k] == 1)
				{
					// if the 2 points match, we look up in the features of the jth image to check whether the second point is in it.
					vector<CvPoint2D32f>::iterator result = find(corners[j].begin(), corners[j].begin() + countArr[j], dest_corners[k]);
					if(result != corners[j].begin() + countArr[j])
					{
						vec.push_back(pair<int, int>(k, result - corners[j].begin()));
					}
					else
					{
						corners[j].push_back(dest_corners[k]);
						countArr[j] ++;
					}
				}
			}
			pair_matches[pair<string, string>(nameList[i], nameList[j])] = vec; 
		}
	}
	*/
	//delete [] dest_corners;
}

void writeToFeatureFiles()
{
	cout << "Writing to feature files..." << endl;
	for(int i = 0; i < nameList.size(); i++)
	{
		cout << genSiftName(nameList[i]) << "..."<< endl;
		fstream ofs(genSiftName(nameList[i]), ios_base::out);
		ofs << corners[i].size() << " " << nDescriptors << endl;
		for(int j = 0; j < corners[i].size(); j++)
		{
			ofs << corners[i][j].x << " " << corners[i][j].y << " " << 1.0 << " " << 1.0 << endl;
			for(int k = 0; k< nDescriptors; k++)
			{
				ofs << 1.0 << " ";
				if((k + 1) % 20 == 0) ofs << endl;
			}
			ofs << endl;				
		}
		ofs.close();
	}
}

void convertToBinary()
{
	FeatureData fd;
	for(int i = 0; i < nameList.size(); i++)
	{
		fd.ReadSIFTA(genSiftName(nameList[i]).c_str());
		fd.saveSIFTB2(genSiftName(nameList[i]).c_str());
	}
}

void writeToMatchFile(const char* filename)
{
	cout << "Wring to match file..." << endl;
	map<pair<string, string>, vector<pair<int, int> > >::iterator it;
	fstream ofs(filename, ios_base::out);
	for(it = pair_matches.begin(); it != pair_matches.end(); it++)
	{
		ofs << it->first.first << " " << it->first.second << " " << it->second.size() << endl;
		for(int i = 0; i < it->second.size(); i++)
			ofs << it->second[i].first << " ";
		ofs << endl;
		for(int i = 0; i < it->second.size(); i++)
			ofs << it->second[i].second << " ";
		ofs << endl;
	}
	ofs.close();
}

void cleanUp()
{
	/*
	for(int i = 0; i < nImgs; i++)
		delete [] corners[i];
	delete [] corners;
	*/
	for(int i = 0; i < imageList.size(); i++)
		cvReleaseImage(&imageList[i]);
	cout << "Process done." << endl;
}

int main(int arg, char** argv)
{
	readImageList("nameList.txt");

	findGoodFeatures();

	conductMatching();
	
	writeToFeatureFiles();	

	writeToMatchFile("match.txt");

	cleanUp();
	
	return 0;
}