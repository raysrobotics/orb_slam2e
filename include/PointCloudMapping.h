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

#ifndef POINTCLOUDMAPPING_H
#define POINTCLOUDMAPPING_H

#include "System.h"

#include <pcl/common/transforms.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <condition_variable>
#include <queue>
#include <atomic> 

using namespace ORB_SLAM2;

class PointCloudMapping
{
public:
    typedef pcl::PointXYZRGBA PointT;
    typedef pcl::PointCloud<PointT> PointCloud;
   
    PointCloudMapping( double resolution_, const int sensorType_);
    
    // 插入一个keyframe，会更新一次地图
    void insertKeyFrame( KeyFrame* kf, cv::Mat& color, cv::Mat& depth );
    void shutdown();
    void viewer();
    PointCloud::Ptr getGlobalMap(void);
    PointCloud::Ptr getLocalMap(void);
    
    mutex                globalMapMutex;
    mutex                localMapMutex;
    condition_variable   globalMapUpdated;
    condition_variable   localMapUpdated;
protected:
    PointCloud::Ptr generatePointCloud(KeyFrame* kf, cv::Mat& color, cv::Mat& depth);
    
    PointCloud::Ptr globalMap;
    PointCloud::Ptr localMap;
    shared_ptr<thread>  viewerThread;   
    
    atomic_bool shutDownFlag;
   
    condition_variable  keyFrameUpdated;
    mutex               keyFrameUpdateMutex;
    
    //data to generate point clouds
    queue<KeyFrame*, list<KeyFrame*>> keyframes;
    queue<cv::Mat,   list<cv::Mat>>   colorImgs;
    queue<cv::Mat,   list<cv::Mat>>   depthImgs;
    mutex     keyframeMutex;
    
    double resolution = 0.04;
    pcl::VoxelGrid<PointT> voxel;
    pcl::StatisticalOutlierRemoval<PointT> sor;   //创建滤波器对象
    
    int meSensorType;
};

#endif // POINTCLOUDMAPPING_H
