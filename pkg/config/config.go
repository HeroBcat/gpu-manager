/*
 * Tencent is pleased to support the open source community by making TKEStack available.
 *
 * Copyright (C) 2012-2019 Tencent. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use
 * this file except in compliance with the License. You may obtain a copy of the
 * License at
 *
 * https://opensource.org/licenses/Apache-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OF ANY KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations under the License.
 */

package config

import (
	"time"

	"tkestack.io/gpu-manager/pkg/types"
)

// Config contains the necessary options for the plugin.
type Config struct { // 以下参数同 cmd/manager/options/options.go 中传入的参数
	Driver                   string            // GPU 驱动，默认值 nvidia，显然可以扩展支持其他类型的 GPU
	ExtraConfigPath          string            // 额外的设置
	QueryPort                int               // 统计信息服务的查询接口
	QueryAddr                string            // 统计信息服务的监听接口
	KubeConfig               string            // k8s 授权配置文件
	SamplePeriod             time.Duration     // gpu-manager 会查询 GPU 设备的使用情况，这个参数用来设定采样周期
	Hostname                 string            // gpu-manager 在运行时，只关注自己节点上的 pod，这主要靠 hostname 来辨别
	NodeLabels               map[string]string // 给节点自动打标签
	VirtualManagerPath       string            // gpu-manager 会为所有需要虚拟 gpu 资源的 pod 创建唯一的文件夹，文件夹的路径就在这个地址下
	DevicePluginPath         string            // 默认的 device plugin 的目录地址s
	VolumeConfigPath         string            // volume 动态链接库和可执行文件的位置，也就是 GPU-manager 需要拦截调用的一些库
	EnableShare              bool              // 是否打开共享模式，将一个物理 gpu 分成多个虚拟 gpu
	AllocationCheckPeriod    time.Duration     // 检查分配了虚拟 gpu 资源的 pod 的状态，及时回收资源
	CheckpointPath           string            // 会产生 checkpoint 来当缓存用
	ContainerRuntimeEndpoint string
	CgroupDriver             string
	RequestTimeout           time.Duration

	VCudaRequestsQueue chan *types.VCudaRequest
}

// ExtraConfig contains extra options other than Config
type ExtraConfig struct {
	Devices []string `json:"devices,omitempty"`
}
