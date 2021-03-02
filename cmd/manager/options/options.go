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

package options

import (
	"time"

	"github.com/spf13/pflag"
)

const (
	DefaultDriver                   = "nvidia"
	DefaultQueryPort                = 5678
	DefaultSamplePeriod             = 1
	DefaultVirtualManagerPath       = "/etc/gpu-manager/vm"
	DefaultAllocationCheckPeriod    = 30
	DefaultCheckpointPath           = "/etc/gpu-manager/checkpoint"
	DefaultContainerRuntimeEndpoint = "/var/run/dockershim.sock"
	DefaultCgroupDriver             = "cgroupfs"
)

// Options contains plugin information
type Options struct {
	Driver                   string
	ExtraPath                string
	VolumeConfigPath         string
	QueryPort                int
	QueryAddr                string
	KubeConfigFile           string
	SamplePeriod             int
	NodeLabels               string
	HostnameOverride         string
	VirtualManagerPath       string
	DevicePluginPath         string
	EnableShare              bool
	AllocationCheckPeriod    int
	CheckpointPath           string
	ContainerRuntimeEndpoint string
	CgroupDriver             string
	RequestTimeout           time.Duration
}

// NewOptions gives a default options template.
func NewOptions() *Options {
	return &Options{
		Driver:                   DefaultDriver,
		QueryPort:                DefaultQueryPort,
		QueryAddr:                "localhost",
		SamplePeriod:             DefaultSamplePeriod,
		VirtualManagerPath:       DefaultVirtualManagerPath,
		AllocationCheckPeriod:    DefaultAllocationCheckPeriod,
		CheckpointPath:           DefaultCheckpointPath,
		ContainerRuntimeEndpoint: DefaultContainerRuntimeEndpoint,
		CgroupDriver:             DefaultCgroupDriver,
		RequestTimeout:           time.Second * 5,
	}
}

// AddFlags add some commandline flags.
func (opt *Options) AddFlags(fs *pflag.FlagSet) {
	// GPU 驱动，默认值 nvidia，显然可以扩展支持其他类型的 GPU
	fs.StringVar(&opt.Driver, "driver", opt.Driver, "The driver name for manager")
	// 额外的设置
	fs.StringVar(&opt.ExtraPath, "extra-config", opt.ExtraPath, "The extra config file location")
	// volume 动态链接库和可执行文件的位置，也就是 GPU-manager 需要拦截调用的一些库
	fs.StringVar(&opt.VolumeConfigPath, "volume-config", opt.VolumeConfigPath, "The volume config file location")
	// 统计信息服务的查询接口
	fs.IntVar(&opt.QueryPort, "query-port", opt.QueryPort, "port for query statistics information")
	// 统计信息服务的监听接口
	fs.StringVar(&opt.QueryAddr, "query-addr", opt.QueryAddr, "address for query statistics information")
	// k8s 授权配置文件
	fs.StringVar(&opt.KubeConfigFile, "kubeconfig", opt.KubeConfigFile, "Path to kubeconfig file with authorization information (the master location is set by the master flag).")
	// gpu-manager 会查询 GPU 设备的使用情况，这个参数用来设定采样周期
	fs.IntVar(&opt.SamplePeriod, "sample-period", opt.SamplePeriod, "Sample period for each card, unit second")
	// 给节点自动打标签
	fs.StringVar(&opt.NodeLabels, "node-labels", opt.NodeLabels, "automated label for this node, if empty, node will be only labeled by gpu model")
	// gpu-manager 在运行时，只关注自己节点上的 pod，这主要靠 hostname 来辨别
	fs.StringVar(&opt.HostnameOverride, "hostname-override", opt.HostnameOverride, "If non-empty, will use this string as identification instead of the actual hostname.")
	// gpu-manager 会为所有需要虚拟 gpu 资源的 pod 创建唯一的文件夹，文件夹的路径就在这个地址下
	fs.StringVar(&opt.VirtualManagerPath, "virtual-manager-path", opt.VirtualManagerPath, "configuration path for virtual manager store files")
	// 默认的 device plugin 的目录地址
	fs.StringVar(&opt.DevicePluginPath, "device-plugin-path", opt.DevicePluginPath, "the path for kubelet receive device plugin registration")
	// 会产生 checkpoint 来当缓存用
	fs.StringVar(&opt.CheckpointPath, "checkpoint-path", opt.CheckpointPath, "configuration path for checkpoint store file")
	// 是否打开共享模式，将一个物理 gpu 分成多个虚拟 gpu
	fs.BoolVar(&opt.EnableShare, "share-mode", opt.EnableShare, "enable share mode allocation")
	// 检查分配了虚拟 gpu 资源的 pod 的状态，及时回收资源
	fs.IntVar(&opt.AllocationCheckPeriod, "allocation-check-period", opt.AllocationCheckPeriod, "allocation check period, unit second")
	fs.StringVar(&opt.ContainerRuntimeEndpoint, "container-runtime-endpoint", opt.ContainerRuntimeEndpoint, "container runtime endpoint")
	fs.StringVar(&opt.CgroupDriver, "cgroup-driver", opt.CgroupDriver, "Driver that the kubelet uses to manipulate cgroups on the host.  "+
		"Possible values: 'cgroupfs', 'systemd'")
	fs.DurationVar(&opt.RequestTimeout, "runtime-request-timeout", opt.RequestTimeout,
		"request timeout for communicating with container runtime endpoint")
}
