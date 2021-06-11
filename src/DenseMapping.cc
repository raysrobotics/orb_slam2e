/**
* This file is part of ORB-SLAM2.
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

#include "DenseMapping.h"
#include "LoopClosing.h"
#include "ORBmatcher.h"
#include "Optimizer.h"

#include "Converter.h"

#include<mutex>

namespace ORB_SLAM2
{

DenseMapping::DenseMapping(double resolution_, const int sensorType_):
    resolution(resolution_), meSensorType(sensorType_), 
    mbResetRequested(false), mbFinishRequested(false), mbFinished(true),
    mbAbortBA(false), mbStopped(false), mbStopRequested(false), mbNotStop(false), mbAcceptKeyFrames(true)
{
    // We do not support monocular camera
    if (meSensorType == System::eSensor::MONOCULAR)
        return;

    // Initialize filter parameters

    // Initialize PointCloud::Ptr
    mGlobalDenseMap.reset(new PointCloud);
    mLocalDenseMap.reset(new PointCloud);
}

void DenseMapping::Run()
{
    mbFinished = false;

    while(1)
    {
        // Tracking will see that Dense Mapping is busy
        SetAcceptKeyFrames(false);

        // Check if there are keyframes in the queue
        if(CheckNewKeyFrames())
        {
            ProcessNewKeyFrame();            

            mbAbortBA = false;

            if(!CheckNewKeyFrames() && !stopRequested())
            {
                // we may do some optimization stuff here
                cout << "Creating Dense Map... Point count:"<< mGlobalDenseMap->points.size() << endl;
                // Create dense points
                CreateDensePoints();
                CreateGlobalMap();
            }

            // mpLoopCloser->InsertKeyFrame(mpCurrentKeyFrame);
        }
        else if(Stop())
        {
            // Safe area to stop
            while(isStopped() && !CheckFinish())
            {
                usleep(3000);
            }
            if(CheckFinish())
            {
                SaveGlobalMap();
                break;
            }
        }

        ResetIfRequested();

        // Tracking will see that Dense Mapping is busy
        SetAcceptKeyFrames(true);

        if(CheckFinish())    
        {
            SaveGlobalMap();
            break;
        }

        usleep(3000);
    }

    SetFinish();
}

void DenseMapping::InsertKeyFrame(KeyFrame *pKF, cv::Mat &color, cv::Mat &depth)
{
    unique_lock<mutex> lock(mMutexNewKFs);
    mqKeyFrames.push(pKF);
    mqColorImgs.push(color.clone());
    mqDepthImgs.push(depth.clone());
    mbAbortBA=true;
}

bool DenseMapping::CheckNewKeyFrames()
{
    unique_lock<mutex> lock(mMutexNewKFs);
    // Since mqKeyFrames|mqColorImgs|mqDepthImgs are always handled simuteniously,
    // we only check if mqKeyFrames is empty
    return(!mqKeyFrames.empty());
}

void DenseMapping::ProcessNewKeyFrame()
{
    unique_lock<mutex> lock(mMutexNewKFs);
    
    mpCurrentKeyFrame = mqKeyFrames.front();
    mmCurrentColorImgs = mqColorImgs.front();
    mmCurrentDepthImgs = mqDepthImgs.front();

    mqKeyFrames.pop();
    mqColorImgs.pop();
    mqDepthImgs.pop();
}

void DenseMapping::CreateDensePoints()
{
    const int step = 2;

    PointCloud::Ptr pLocal(new PointCloud());
    for (int m = 0; m < mmCurrentDepthImgs.rows; m += step)
    {
        for (int n = 0; n < mmCurrentDepthImgs.cols; n += step)
        {
            float d = mmCurrentDepthImgs.ptr<float>(m)[n];
            
            // if the point is too near or too far, we ignore it
            if (d < 0.01 || d > 8)
                continue;
            
            PointT p;
            // Transformation from camera frame to point cloud frame
            p.z = d;
            p.x = (n - mpCurrentKeyFrame->cx) * p.z / mpCurrentKeyFrame->fx;
            p.y = (m - mpCurrentKeyFrame->cy) * p.z / mpCurrentKeyFrame->fy;

            p.r = mmCurrentColorImgs.ptr<uchar>(m)[n * 3]; //CV通道顺序BGR
            p.g = mmCurrentColorImgs.ptr<uchar>(m)[n * 3 + 1];
            p.b = mmCurrentColorImgs.ptr<uchar>(m)[n * 3 + 2];

            pLocal->points.push_back(p);
        }
    }

    // From point cloud frame to ROS frame
    Eigen::Matrix3d Rc0w;
    Rc0w = Eigen::AngleAxisd(M_PI_2, Eigen::Vector3d::UnitZ()) *
           Eigen::AngleAxisd(-M_PI_2, Eigen::Vector3d::UnitY()) *
           Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitX());
    Eigen::Isometry3d Tc0w = Eigen::Isometry3d::Identity();
    Tc0w.rotate(Rc0w);

    Eigen::Isometry3d Tcc0 = Converter::toSE3Quat(mpCurrentKeyFrame->GetPose());
    Eigen::Isometry3d Tcw = Tcc0 * Tc0w; 

    // Set output point cloud  
    pcl::transformPointCloud(*pLocal, *mLocalDenseMap, Tcw.inverse().matrix());
}

void DenseMapping::CreateGlobalMap()
{
    // we may use some pcl filters here
    *mGlobalDenseMap += *mLocalDenseMap;
}

void DenseMapping::SaveGlobalMap()
{
    cout << "Saving Global Map..." << endl;
    pcl::io::savePCDFile("./result.pcd", *mGlobalDenseMap);
    cout << "Global Map Saved. Point count:" << mGlobalDenseMap->points.size() << endl;
}

void DenseMapping::RequestStop()
{
    unique_lock<mutex> lock(mMutexStop);
    mbStopRequested = true;
    unique_lock<mutex> lock2(mMutexNewKFs);
    mbAbortBA = true;
}

bool DenseMapping::Stop()
{
    unique_lock<mutex> lock(mMutexStop);
    if(mbStopRequested && !mbNotStop)
    {
        mbStopped = true;
        cout << "Dense Mapping STOP" << endl;
        return true;
    }

    return false;
}

bool DenseMapping::isStopped()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopped;
}

bool DenseMapping::stopRequested()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopRequested;
}

void DenseMapping::Release()
{
    unique_lock<mutex> lock(mMutexStop);
    unique_lock<mutex> lock2(mMutexFinish);
    if(mbFinished)
        return;
    mbStopped = false;
    mbStopRequested = false;
    
    // release the queues
    // stackoverflow.com/questions/709146/how-do-i-clear-the-stdqueue-efficiently
    std::queue<KeyFrame*, std::list<KeyFrame*>>().swap(mqKeyFrames);
    std::queue<cv::Mat, std::list<cv::Mat>>().swap(mqColorImgs);
    std::queue<cv::Mat, std::list<cv::Mat>>().swap(mqDepthImgs);

    cout << "Dense Mapping RELEASE" << endl;
}

bool DenseMapping::AcceptKeyFrames()
{
    unique_lock<mutex> lock(mMutexAccept);
    return mbAcceptKeyFrames;
}

void DenseMapping::SetAcceptKeyFrames(bool flag)
{
    unique_lock<mutex> lock(mMutexAccept);
    mbAcceptKeyFrames=flag;
}

bool DenseMapping::SetNotStop(bool flag)
{
    unique_lock<mutex> lock(mMutexStop);

    if(flag && mbStopped)
        return false;

    mbNotStop = flag;

    return true;
}

void DenseMapping::InterruptBA()
{
    mbAbortBA = true;
}

void DenseMapping::RequestReset()
{
    {
        unique_lock<mutex> lock(mMutexReset);
        mbResetRequested = true;
    }

    while(1)
    {
        {
            unique_lock<mutex> lock2(mMutexReset);
            if(!mbResetRequested)
                break;
        }
        usleep(3000);
    }
}

void DenseMapping::ResetIfRequested()
{
    unique_lock<mutex> lock(mMutexReset);
    if(mbResetRequested)
    {
        // release the queues
        // stackoverflow.com/questions/709146/how-do-i-clear-the-stdqueue-efficiently
        std::queue<KeyFrame*, std::list<KeyFrame*>>().swap(mqKeyFrames);
        std::queue<cv::Mat, std::list<cv::Mat>>().swap(mqColorImgs);
        std::queue<cv::Mat, std::list<cv::Mat>>().swap(mqDepthImgs);

        mGlobalDenseMap.reset(new PointCloud);
        mLocalDenseMap.reset(new PointCloud);

        mbResetRequested=false;
    }
}

void DenseMapping::RequestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool DenseMapping::CheckFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void DenseMapping::SetFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;    
    unique_lock<mutex> lock2(mMutexStop);
    mbStopped = true;
}

bool DenseMapping::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}

} //namespace ORB_SLAM
