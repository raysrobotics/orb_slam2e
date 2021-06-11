/**
* This file is part of ORB-SLAM2e.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef DENSEMAPPING_H
#define DENSEMAPPING_H

#include "KeyFrame.h"
#include "Map.h"
#include "LoopClosing.h"
#include "Tracking.h"
#include "KeyFrameDatabase.h"

#include <mutex>
#include <queue>

#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>

namespace ORB_SLAM2
{

class Tracking;
class LoopClosing;
class Map;

class DenseMapping
{
public:
    typedef pcl::PointXYZRGBA PointT;
    typedef pcl::PointCloud<PointT> PointCloud;

    DenseMapping(double resolution_, const int sensorType_);

    // void SetLoopCloser(LoopClosing* pLoopCloser);

    // void SetTracker(Tracking* pTracker);

    // Main function
    void Run();

    void InsertKeyFrame(KeyFrame* pKF, cv::Mat& color, cv::Mat& depth );

    // Thread Synch
    void RequestStop();
    void RequestReset();
    bool Stop();
    void Release();
    bool isStopped();
    bool stopRequested();
    bool AcceptKeyFrames();
    void SetAcceptKeyFrames(bool flag);
    bool SetNotStop(bool flag);

    void InterruptBA();

    void RequestFinish();
    bool isFinished();

    // int KeyframesInQueue(){
    //     unique_lock<std::mutex> lock(mMutexNewKFs);
    //     return mlNewKeyFrames.size();
    // }

protected:

    bool CheckNewKeyFrames();
    void ProcessNewKeyFrame();
    // void CreateNewMapPoints();
    void CreateDensePoints();
    void CreateGlobalMap();
    void SaveGlobalMap();

    // void MapPointCulling();
    // void SearchInNeighbors();

    // void KeyFrameCulling();

    // cv::Mat ComputeF12(KeyFrame* &pKF1, KeyFrame* &pKF2);

    // cv::Mat SkewSymmetricMatrix(const cv::Mat &v);



    double resolution;
    int meSensorType;


    // bool mbMonocular;

    void ResetIfRequested();
    bool mbResetRequested;
    std::mutex mMutexReset;

    bool CheckFinish();
    void SetFinish();
    bool mbFinishRequested;
    bool mbFinished;
    std::mutex mMutexFinish;

    // Map* mpMap;

    // LoopClosing* mpLoopCloser;
    // Tracking* mpTracker;

    // std::list<KeyFrame*> mlNewKeyFrames;
    std::queue<KeyFrame*, std::list<KeyFrame*>> mqKeyFrames;
    std::queue<cv::Mat, std::list<cv::Mat>> mqColorImgs;
    std::queue<cv::Mat, std::list<cv::Mat>> mqDepthImgs;

    KeyFrame* mpCurrentKeyFrame;
    cv::Mat mmCurrentColorImgs;
    cv::Mat mmCurrentDepthImgs;

    PointCloud::Ptr mLocalDenseMap;
    PointCloud::Ptr mGlobalDenseMap;

    // std::list<MapPoint*> mlpRecentAddedMapPoints;

    std::mutex mMutexNewKFs;

    bool mbAbortBA;

    bool mbStopped;
    bool mbStopRequested;
    bool mbNotStop;
    std::mutex mMutexStop;

    bool mbAcceptKeyFrames;
    std::mutex mMutexAccept;
};

} //namespace ORB_SLAM

#endif // DENSEMAPPING_H
