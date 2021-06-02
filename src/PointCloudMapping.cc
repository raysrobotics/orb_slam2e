/*
 * <one line to give the program's name and a brief idea of what it does.>
 * Copyright (C) 2016  <copyright holder> <email>
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 * 
 */

#include <thread>
#include <queue>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include <KeyFrame.h>

#include "Converter.h"
#include "PointCloudMapping.h"

#include <pcl/visualization/cloud_viewer.h>
#include <pcl/io/pcd_io.h>

PointCloudMapping::PointCloudMapping(double resolution_, const int sensorType_)
    : shutDownFlag(false), resolution(resolution_), meSensorType(sensorType_)
{
    if (meSensorType == System::eSensor::MONOCULAR)
        return;

    voxel.setLeafSize(resolution, resolution, resolution);
    sor.setMeanK(50);
    sor.setStddevMulThresh(1.0);
    globalMap = boost::make_shared<PointCloud>();
    localMap = boost::make_shared<PointCloud>();

    viewerThread = make_shared<thread>(bind(&PointCloudMapping::viewer, this));
}

void PointCloudMapping::shutdown()
{
    shutDownFlag = true;
    keyFrameUpdated.notify_one();

    viewerThread->join();
    pcl::io::savePCDFile("./result.pcd", *globalMap);
    cout << "globalMap save finished" << endl;
}

void PointCloudMapping::insertKeyFrame(KeyFrame *kf, cv::Mat &color, cv::Mat &depth)
{
    if (meSensorType == System::eSensor::MONOCULAR)
        return; //Does not support

    cout<<"receive a keyframe, id = "<<kf->mnId<<endl;
    // cout<<"depth.rows:"<<depth.rows<<", depth.cols:"<<depth.cols<<endl;
    unique_lock<mutex> lck(keyframeMutex);
    keyframes.push(kf);
    colorImgs.push(color.clone());
    depthImgs.push(depth.clone());
    // //     if(colorImgs.size()>10)
    // //     {
    // //       keyframes.pop( );
    // //       colorImgs.pop( );
    // //       depthImgs.pop( );
    // //     }
    lck.unlock();

    keyFrameUpdated.notify_one();
}

pcl::PointCloud<PointCloudMapping::PointT>::Ptr PointCloudMapping::getGlobalMap(void)
{
    // 	unique_lock<mutex> lck( globalMapMutex );//这里在外面加锁会不会更好一点??这里一定要记得在外面
    return globalMap;
}

pcl::PointCloud<PointCloudMapping::PointT>::Ptr PointCloudMapping::getLocalMap(void)
{
    // 	unique_lock<mutex> lck( globalMapMutex );//这里在外面加锁会不会更好一点??这里一定要记得在外面
    return localMap;
}

pcl::PointCloud<PointCloudMapping::PointT>::Ptr PointCloudMapping::generatePointCloud(KeyFrame *kf, cv::Mat &color, cv::Mat &depth)
{
    const int step = 2;
    if (meSensorType == System::eSensor::STEREO) //stereo
    {
        cv::Mat leftGrayImgs = color; //Kitti是单通道图
        if (leftGrayImgs.channels() != 1)
            cv::cvtColor(leftGrayImgs, leftGrayImgs, CV_BGR2GRAY);
        else
            cv::cvtColor(leftGrayImgs, color, CV_GRAY2RGB);

        //1.根据双目图像对点云进行恢复
        // 神奇的参数
        cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(0, 96, 9, 8 * 9 * 9, 32 * 9 * 9, 1, 63, 10, 100, 32);
        cv::Mat disparity_sgbm;
        sgbm->compute(leftGrayImgs, depth, disparity_sgbm); //这里输入三通图有问题
        disparity_sgbm.convertTo(depth, CV_32F, 1.0 / 16.0f);

        for (int v = 0; v < leftGrayImgs.rows; v += step)
            for (int u = 0; u < leftGrayImgs.cols; u += step)
                if (depth.at<float>(v, u) <= 0.1 || depth.at<float>(v, u) >= 96.0)
                    depth.at<float>(v, u) = FLT_MAX;
                else
                    depth.at<float>(v, u) = kf->mbf / (depth.at<float>(v, u));
    }

    PointCloud::Ptr tmp(new PointCloud());
    // point cloud is null ptr
    for (int m = 0; m < depth.rows; m += step)
    {
        for (int n = 0; n < depth.cols; n += step)
        {
            float d = depth.ptr<float>(m)[n];
            if (d < 0.01 || d > 8)
                continue;
            PointT p;
            p.z = d;
            p.x = (n - kf->cx) * p.z / kf->fx;
            p.y = (m - kf->cy) * p.z / kf->fy;

            p.r = color.ptr<uchar>(m)[n * 3]; //CV通道顺序BGR
            p.g = color.ptr<uchar>(m)[n * 3 + 1];
            p.b = color.ptr<uchar>(m)[n * 3 + 2];

            tmp->points.push_back(p);
        }
    }
    // cout<<"depth.rows:"<<depth.rows<<", depth.cols:"<<depth.cols<<endl;
    
    Eigen::Matrix3d Rc0w;
    Rc0w = Eigen::AngleAxisd(M_PI_2, Eigen::Vector3d::UnitZ()) *
           Eigen::AngleAxisd(-M_PI_2, Eigen::Vector3d::UnitY()) *
           Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitX());
    Eigen::Isometry3d Tc0w = Eigen::Isometry3d::Identity();
    Tc0w.rotate(Rc0w);

    Eigen::Isometry3d Tcc0 = ORB_SLAM2::Converter::toSE3Quat(kf->GetPose());
    Eigen::Isometry3d Tcw = Tcc0 * Tc0w; //转到ROS坐标系下

    PointCloud::Ptr cloud(new PointCloud);
    pcl::transformPointCloud(*tmp, *cloud, Tcw.inverse().matrix());
    cloud->is_dense = false;

    cout<<"generate point cloud for kf "<<kf->mnId<<", size="<<cloud->points.size()<<endl;
    return cloud;
}

void PointCloudMapping::viewer()
{
    pcl::visualization::CloudViewer viewer("viewer");

    while (!shutDownFlag)
    {
        bool KFIsEmpty = true;
        {
            unique_lock<mutex> lck(keyframeMutex);
            KFIsEmpty = keyframes.empty();
        }
        if (KFIsEmpty)
        { //这里如果队列是空的才等待//不是空的会造成阻塞浪费//不在wait状态的notify无效
            unique_lock<mutex> lck_keyframeUpdated(keyFrameUpdateMutex);
            keyFrameUpdated.wait(lck_keyframeUpdated);
        }

        if (keyframes.empty()) //防止队列是空的,指针无效
            continue;

        // keyframe is updated
        KeyFrame *pKeyframes_temp = nullptr;
        cv::Mat colorImgs_temp;
        cv::Mat depthImgs_temp;
        {
            unique_lock<mutex> lck(keyframeMutex);

            pKeyframes_temp = keyframes.front();
            keyframes.pop();
            colorImgs_temp = colorImgs.front();
            colorImgs.pop();
            depthImgs_temp = depthImgs.front();
            depthImgs.pop();

            //--------------------
            lck.unlock();
        }

        PointCloud::Ptr p = generatePointCloud(pKeyframes_temp, colorImgs_temp, depthImgs_temp);

        {
            unique_lock<mutex> lck(localMapMutex);
            localMap = p;
            //--------------------
            // PointCloud::Ptr tmp(new PointCloud);
            // sor.setInputCloud(localMap);
            // sor.filter(*tmp);
            //--------------------
            lck.unlock();
            localMapUpdated.notify_one();
        }
        {
            unique_lock<mutex> lck(globalMapMutex);
            *globalMap += *localMap;
            lck.unlock();
            globalMapUpdated.notify_one(); //通知getGlobalPointCloud的点
        }

        {
            // //https://blog.csdn.net/cqtju/article/details/80169009
            // PointCloud::Ptr tmp(new PointCloud);
            // sor.setInputCloud(localMap);
            // sor.filter(*tmp);

            // unique_lock<mutex> lck(globalMapMutex);
            // *globalMap += *tmp;

            // voxel.setInputCloud(globalMap);
            // PointCloud::Ptr tmp2(new PointCloud());
            // voxel.filter(*tmp2);

            // globalMap->swap(*tmp2);
            // lck.unlock();
            // globalMapUpdated.notify_one(); //通知getGlobalPointCloud的点
        }

        viewer.showCloud(globalMap);
        cout<<"Local Map Size="<<localMap->points.size()<<endl;
        cout<<"show global map, size="<<globalMap->points.size()<<endl;
    }
}
