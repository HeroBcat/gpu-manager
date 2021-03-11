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

package nvidia

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	nveval "tkestack.io/gpu-manager/pkg/algorithm/nvidia"
	"tkestack.io/gpu-manager/pkg/config"
	"tkestack.io/gpu-manager/pkg/device"
	nvtree "tkestack.io/gpu-manager/pkg/device/nvidia"
	"tkestack.io/gpu-manager/pkg/services/allocator"
	"tkestack.io/gpu-manager/pkg/services/allocator/cache"
	"tkestack.io/gpu-manager/pkg/services/allocator/checkpoint"
	"tkestack.io/gpu-manager/pkg/services/watchdog"
	"tkestack.io/gpu-manager/pkg/types"
	"tkestack.io/gpu-manager/pkg/utils"

	"golang.org/x/net/context"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	k8stypes "k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog"
	pluginapi "k8s.io/kubelet/pkg/apis/deviceplugin/v1beta1"
)

const (
	checkpointFileName = "gpumanager_internal_checkpoint"
)

func init() {
	allocator.Register("nvidia", NewNvidiaTopoAllocator)
	allocator.Register("nvidia_test", NewNvidiaTopoAllocatorForTest)
}

// NvidiaTopoAllocator is an allocator for Nvidia GPU
type NvidiaTopoAllocator struct {
	sync.Mutex

	tree         *nvtree.NvidiaTree
	allocatedPod *cache.PodCache

	config            *config.Config
	evaluators        map[string]Evaluator
	extraConfig       map[string]*config.ExtraConfig
	k8sClient         kubernetes.Interface
	unfinishedPod     *v1.Pod
	queue             workqueue.RateLimitingInterface
	stopChan          chan struct{}
	checkpointManager *checkpoint.Manager
}

const (
	ALLOCATE_SUCCESS = iota
	ALLOCATE_FAIL
	PREDICATE_MISSING
)

type allocateResult struct {
	pod     *v1.Pod
	result  int
	message string
	reason  string
	resChan chan struct{}
}

var (
	_           allocator.GPUTopoService = &NvidiaTopoAllocator{}
	waitTimeout                          = 10 * time.Second
)

// NewNvidiaTopoAllocator returns a new NvidiaTopoAllocator
func NewNvidiaTopoAllocator(config *config.Config, tree device.GPUTree, k8sClient kubernetes.Interface) allocator.GPUTopoService {
	_tree, _ := tree.(*nvtree.NvidiaTree)
	cm, err := checkpoint.NewManager(config.CheckpointPath, checkpointFileName)
	if err != nil {
		klog.Fatalf("Failed to create checkpoint manager due to %s", err.Error())
	}
	alloc := &NvidiaTopoAllocator{
		tree:              _tree,
		config:            config,
		evaluators:        make(map[string]Evaluator),
		allocatedPod:      cache.NewAllocateCache(),
		k8sClient:         k8sClient,
		queue:             workqueue.NewRateLimitingQueue(workqueue.DefaultControllerRateLimiter()),
		stopChan:          make(chan struct{}),
		checkpointManager: cm,
	}

	// allocator 调用 loadModule() 来启动 nvidia 的内核模块
	// Load kernel module if it's not loaded
	alloc.loadModule()

	// 初始化评估器
	// 这里的 _tree 就是感知到的 GPU 拓扑结构
	// Initialize evaluator
	alloc.initEvaluator(_tree)

	// 加载启动时传入的额外参数配置文件
	// Read extra config if it's given
	alloc.loadExtraConfig(config.ExtraConfigPath)

	// 开启新的协程来处理分配结果
	// Process allocation results in another goroutine
	go wait.Until(alloc.runProcessResult, time.Second, alloc.stopChan)

	// 恢复 gpu-manager 分配结果
	// 比如在 gpu-manager 重启之后，之前的 gpu 分配结果都丢失了，但节点上还有大量的容器正在占用 gpu
	// 这个方法通过查找节点上存活的容器，通过 docker endpoint 调用 InspectContainer 获取容器中占用的 device id (?怎么看出来的?)
	// 然后标记改设备和容器之间的占用关系 (?怎么看出来的?)
	// Recover
	alloc.recoverInUsed()

	// 创建新的协程来周期性的检查资源分配情况
	// 如果是 Failed 和 Pending 状态的容器，就根据错误信息检查是否应该删除它们
	// 然后如果这些 pod 的控制器是 deployment 类似的，就尝试删除它们，这样控制器会重新创建这些 pod 进行调度，让这些 pod 恢复到正常运行状态
	// Check allocation in another goroutine periodically
	go alloc.checkAllocationPeriodically(alloc.stopChan)

	return alloc
}

// NewNvidiaTopoAllocatorForTest returns a new NvidiaTopoAllocator
// with fake docker client, just for testing.
func NewNvidiaTopoAllocatorForTest(config *config.Config, tree device.GPUTree, k8sClient kubernetes.Interface) allocator.GPUTopoService {
	_tree, _ := tree.(*nvtree.NvidiaTree)
	cm, err := checkpoint.NewManager("/tmp", checkpointFileName)
	if err != nil {
		klog.Fatalf("Failed to create checkpoint manager due to %s", err.Error())
	}
	alloc := &NvidiaTopoAllocator{
		tree:              _tree,
		config:            config,
		evaluators:        make(map[string]Evaluator),
		allocatedPod:      cache.NewAllocateCache(),
		k8sClient:         k8sClient,
		stopChan:          make(chan struct{}),
		queue:             workqueue.NewRateLimitingQueue(workqueue.DefaultControllerRateLimiter()),
		checkpointManager: cm,
	}

	// Initialize evaluator
	alloc.initEvaluator(_tree)

	// Check allocation in another goroutine periodically
	go alloc.checkAllocationPeriodically(alloc.stopChan)

	return alloc
}

func (ta *NvidiaTopoAllocator) runProcessResult() {
	for ta.processNextResult() {
	}
}

// #lizard forgives
func (ta *NvidiaTopoAllocator) recoverInUsed() {
	// Read and unmarshal data from checkpoint to allocatedPod before recover from docker
	ta.readCheckpoint()

	// Recover device tree by reading checkpoint file
	for uid, containerToInfo := range ta.allocatedPod.PodGPUMapping {
		for cName, cache := range containerToInfo {
			for _, dev := range cache.Devices {
				if utils.IsValidGPUPath(dev) {
					klog.V(2).Infof("Nvidia GPU %q is in use by container: %q", dev, cName)
					klog.V(2).Infof("Uid: %s, Name: %s, util: %d, memory: %d", uid, cName, cache.Cores, cache.Memory)

					id, _ := utils.GetGPUMinorID(dev)
					ta.tree.MarkOccupied(&nvtree.NvidiaNode{
						Meta: nvtree.DeviceMeta{
							MinorID: id,
						},
					}, cache.Cores, cache.Memory)
				}
			}
		}
	}

	ta.recycle()
	ta.writeCheckpoint()
	ta.checkAllocation()
}

func (ta *NvidiaTopoAllocator) checkAllocation() {
	klog.V(4).Infof("Checking allocation of pods on this node")
	pods, err := getPodsOnNode(ta.k8sClient, ta.config.Hostname, "")
	if err != nil {
		klog.Infof("Failed to get pods on node due to %v", err)
		return
	}

	for i, p := range pods {
		if !utils.IsGPURequiredPod(&p) {
			continue
		}
		switch p.Status.Phase {
		case v1.PodFailed, v1.PodPending:
			if utils.ShouldDelete(&pods[i]) {
				_ = ta.deletePodWithOwnerRef(&p)
			}
		case v1.PodRunning:
			annotaionMap, err := ta.getReadyAnnotations(&pods[i], true)
			if err != nil {
				klog.Infof("failed to get ready annotations for pod %s", p.UID)
				continue
			}
			pass := true
			for key, val := range annotaionMap {
				if v, ok := p.Annotations[key]; !ok || v != val {
					pass = false
					break
				}
			}
			if !pass {
				ar := &allocateResult{
					pod:     &pods[i],
					result:  ALLOCATE_SUCCESS,
					resChan: make(chan struct{}),
				}
				ta.queue.AddRateLimited(ar)
				<-ar.resChan
			}
		default:
			continue
		}
	}
}

func (ta *NvidiaTopoAllocator) checkAllocationPeriodically(quit chan struct{}) {
	ticker := time.NewTicker(ta.config.AllocationCheckPeriod)
	for {
		select {
		case <-ticker.C:
			ta.checkAllocation()
		case <-quit:
			ticker.Stop()
			return
		}
	}
}

func (ta *NvidiaTopoAllocator) loadExtraConfig(path string) {
	if path != "" {
		klog.V(2).Infof("Load extra config from %s", path)

		f, err := os.Open(path)
		if err != nil {
			klog.Fatalf("Can not load extra config at %s, err %s", path, err)
		}

		defer f.Close()

		cfg := make(map[string]*config.ExtraConfig)
		if err := json.NewDecoder(f).Decode(&cfg); err != nil {
			klog.Fatalf("Can not unmarshal extra config, err %s", err)
		}

		ta.extraConfig = cfg
	}
}

func (ta *NvidiaTopoAllocator) initEvaluator(tree *nvtree.NvidiaTree) {
	ta.evaluators["link"] = nveval.NewLinkMode(tree)
	ta.evaluators["fragment"] = nveval.NewFragmentMode(tree)
	ta.evaluators["share"] = nveval.NewShareMode(tree)
}

func (ta *NvidiaTopoAllocator) loadModule() {
	if _, err := os.Stat(types.NvidiaCtlDevice); err != nil {
		if out, err := exec.Command("modprobe", "-va", "nvidia-uvm", "nvidia").CombinedOutput(); err != nil {
			klog.V(3).Infof("Running modprobe nvidia-uvm nvidia failed with message: %s, error: %v", out, err)
		}
	}

	if _, err := os.Stat(types.NvidiaUVMDevice); err != nil {
		if out, err := exec.Command("modprobe", "-va", "nvidia-uvm", "nvidia").CombinedOutput(); err != nil {
			klog.V(3).Infof("Running modprobe nvidia-uvm nvidia failed with message: %s, error: %v", out, err)
		}
	}
}

func (ta *NvidiaTopoAllocator) capacity() (devs []*pluginapi.Device) {
	var (
		gpuDevices, memoryDevices []*pluginapi.Device
		totalMemory               int64
	)

	nodes := ta.tree.Leaves()
	for i := range nodes {
		totalMemory += int64(nodes[i].Meta.TotalMemory)
	}

	totalCores := len(nodes) * nvtree.HundredCore
	gpuDevices = make([]*pluginapi.Device, totalCores)
	for i := 0; i < totalCores; i++ {
		gpuDevices[i] = &pluginapi.Device{
			ID:     fmt.Sprintf("%s-%d", types.VCoreAnnotation, i),
			Health: pluginapi.Healthy,
		}
	}

	totalMemoryBlocks := totalMemory / types.MemoryBlockSize
	memoryDevices = make([]*pluginapi.Device, totalMemoryBlocks)
	for i := int64(0); i < totalMemoryBlocks; i++ {
		memoryDevices[i] = &pluginapi.Device{
			ID:     fmt.Sprintf("%s-%d-%d", types.VMemoryAnnotation, types.MemoryBlockSize, i),
			Health: pluginapi.Healthy,
		}
	}

	devs = append(devs, gpuDevices...)
	devs = append(devs, memoryDevices...)

	return
}

// #lizard forgives
func (ta *NvidiaTopoAllocator) allocateOne(pod *v1.Pod, container *v1.Container, req *pluginapi.ContainerAllocateRequest) (*pluginapi.ContainerAllocateResponse, error) {
	var (
		nodes                       []*nvtree.NvidiaNode
		needCores, needMemoryBlocks int64
		predicateMissed             bool
		allocated                   bool
	)

	// 是否是 gpu 预选 pod
	predicateMissed = !utils.IsGPUPredicatedPod(pod)
	// 单节点的总内存
	singleNodeMemory := int64(ta.tree.Leaves()[0].Meta.TotalMemory)
	// 根据有多少个 deviceID 来计算需要多少 core 和 memory
	for _, v := range req.DevicesIDs {
		if strings.HasPrefix(v, types.VCoreAnnotation) {
			// 请求 core
			needCores++
		} else if strings.HasPrefix(v, types.VMemoryAnnotation) {
			// 请求 memory
			needMemoryBlocks++
		}
	}

	if needCores == 0 && needMemoryBlocks == 0 {
		klog.Warningf("Zero request")
		return nil, nil
	}

	// 回收资源
	ta.recycle()

	needMemory := needMemoryBlocks * types.MemoryBlockSize
	ta.tree.Update()
	shareMode := false

	podCache := ta.allocatedPod.GetCache(string(pod.UID))
	containerCache := &cache.Info{}
	if podCache != nil {
		if c, ok := podCache[container.Name]; ok {
			allocated = true
			containerCache = c
			klog.V(2).Infof("container %s of pod %s has already been allocated, the allcation will be skip", container.Name, pod.UID)
		}
	}
	klog.V(2).Infof("Tree graph: %s", ta.tree.PrintGraph())

	if allocated {
		klog.V(2).Infof("container %s of pod %s has already been allocated, get devices from cached instead", container.Name, pod.UID)
		for _, d := range containerCache.Devices {
			node := ta.tree.Query(d)
			if node != nil {
				nodes = append(nodes, node)
			}
		}
	} else {
		klog.V(2).Infof("Try allocate for %s(%s), vcore %d, vmemory %d", pod.UID, container.Name, needCores, needMemory)

		switch {
		case needCores > nvtree.HundredCore: // 需要核心数大于 100，即超过一个物理GPU，使用 link 评估器选出 GPU 节点
			eval, ok := ta.evaluators["link"]
			if !ok {
				return nil, fmt.Errorf("can not find evaluator link")
			}
			if needCores%nvtree.HundredCore > 0 {
				return nil, fmt.Errorf("cores are greater than %d, must be multiple of %d", nvtree.HundredCore, nvtree.HundredCore)
			}
			nodes = eval.Evaluate(needCores, 0)
		case needCores == nvtree.HundredCore: // 正好是 100 核心，使用 fragment 评估器
			eval, ok := ta.evaluators["fragment"]
			if !ok {
				return nil, fmt.Errorf("can not find evaluator fragment")
			}
			nodes = eval.Evaluate(needCores, 0)
		default: // 小于 100 核心，即共享 GPU，使用 share 评估器
			// EnableShare 是启动时指定的参数，代表是否允许多个容器共享一个 GPU
			if !ta.config.EnableShare {
				return nil, fmt.Errorf("share mode is not enabled")
			}
			if needCores == 0 || needMemory == 0 {
				return nil, fmt.Errorf("that cores or memory is zero is not permitted in share mode")
			}

			// evaluate in share mode
			shareMode = true
			eval, ok := ta.evaluators["share"]
			if !ok {
				return nil, fmt.Errorf("can not find evaluator share")
			}
			nodes = eval.Evaluate(needCores, needMemory)
			if len(nodes) == 0 {
				if shareMode && needMemory > singleNodeMemory {
					return nil, fmt.Errorf("request memory %d is larger than %d", needMemory, singleNodeMemory)
				}

				return nil, fmt.Errorf("no free node")
			}

			if !predicateMissed {
				// get predicate node by annotation
				// 通过预选阶段的 pod 会根据容器的 idx 在 Annotations 为该容器写上配置信息
				// 这说明 gpu-admission 会为容器分配 gpu 设备
				containerIndex, err := utils.GetContainerIndexByName(pod, container.Name)
				if err != nil {
					return nil, err
				}
				var devStr string
				if idxStr, ok := pod.ObjectMeta.Annotations[types.PredicateGPUIndexPrefix+strconv.Itoa(containerIndex)]; ok {
					if _, err := strconv.Atoi(idxStr); err != nil {
						return nil, fmt.Errorf("predicate idx %s invalid for pod %s ", idxStr, pod.UID)
					}
					devStr = types.NvidiaDevicePrefix + idxStr
					if !utils.IsValidGPUPath(devStr) {
						return nil, fmt.Errorf("predicate idx %s invalid", devStr)
					}
				} else {
					return nil, fmt.Errorf("failed to find predicate idx for pod %s", pod.UID)
				}

				predicateNode := ta.tree.Query(devStr)
				if predicateNode == nil {
					return nil, fmt.Errorf("failed to get predicate node %s", devStr)
				}

				// 最后还要检查一下 gpu-manager 分配的 gpu 设备 和 gpu-admission 中是否一致，不一致会返回分配失败
				// check if we choose the same node as scheduler
				if predicateNode.MinorName() != nodes[0].MinorName() {
					return nil, fmt.Errorf("Nvidia node mismatch for pod %s(%s), pick up:%s  predicate: %s",
						pod.Name, container.Name, nodes[0].MinorName(), predicateNode.MinorName())
				}
			}
		}
	}

	if len(nodes) == 0 {
		if shareMode && needMemory > singleNodeMemory {
			return nil, fmt.Errorf("request memory %d is larger than %d", needMemory, singleNodeMemory)
		}

		return nil, fmt.Errorf("no free node")
	}

	// 现在我们已经知道要为当前请求的容器分配哪个 gpu 设备，以及分配的资源数量
	// 这样就可以构建 ContainerAllocateResponse
	ctntResp := &pluginapi.ContainerAllocateResponse{
		Envs:        make(map[string]string),
		Mounts:      make([]*pluginapi.Mount, 0),
		Devices:     make([]*pluginapi.DeviceSpec, 0),
		Annotations: make(map[string]string),
	}

	allocatedDevices := sets.NewString()
	deviceList := make([]string, 0)
	for _, n := range nodes {
		name := n.MinorName()
		klog.V(2).Infof("Allocate %s for %s(%s), Meta (%d:%d)", name, pod.UID, container.Name, n.Meta.ID, n.Meta.MinorID)

		ctntResp.Annotations[types.VCoreAnnotation] = fmt.Sprintf("%d", needCores)
		ctntResp.Annotations[types.VMemoryAnnotation] = fmt.Sprintf("%d", needMemory)

		ctntResp.Devices = append(ctntResp.Devices, &pluginapi.DeviceSpec{
			ContainerPath: name,
			HostPath:      name,
			Permissions:   "rwm",
		})
		deviceList = append(deviceList, n.Meta.UUID)

		if !allocated {
			// 在 gpu tree 中标记设备已占用
			ta.tree.MarkOccupied(n, needCores, needMemory)
		}
		allocatedDevices.Insert(name)
	}

	// 更改响应的 Annotations:
	ctntResp.Annotations[types.VDeviceAnnotation] = vDeviceAnnotationStr(nodes)
	if !allocated {
		ta.allocatedPod.Insert(string(pod.UID), container.Name, &cache.Info{
			Devices: allocatedDevices.UnsortedList(),
			Cores:   needCores,
			Memory:  needMemory,
		})
	}

	// 检查 pod 的所有容器是否都完成了分配，并把新的分配信息写入 checkpoint
	// check if all containers of pod has been allocated; set unfinishedPod if not
	unfinished := false
	for _, c := range pod.Spec.Containers {
		if !utils.IsGPURequiredContainer(&c) {
			continue
		}
		podCache := ta.allocatedPod.GetCache(string(pod.UID))
		if podCache != nil {
			if _, ok := podCache[c.Name]; !ok {
				unfinished = true
				break
			}
		}
	}
	if unfinished {
		ta.unfinishedPod = pod
	} else {
		ta.unfinishedPod = nil
	}
	ta.writeCheckpoint()

	// 在 response 中为容器添加 /dev/nvidiactl 和 /dev/nvidia-uvm
	// Append control device
	ctntResp.Devices = append(ctntResp.Devices, &pluginapi.DeviceSpec{
		ContainerPath: types.NvidiaCtlDevice,
		HostPath:      types.NvidiaCtlDevice,
		Permissions:   "rwm",
	})

	ctntResp.Devices = append(ctntResp.Devices, &pluginapi.DeviceSpec{
		ContainerPath: types.NvidiaUVMDevice,
		HostPath:      types.NvidiaUVMDevice,
		Permissions:   "rwm",
	})

	// 如果配置了 extraConfig，还要将里面要默认添加的设备加进去
	// Append default device
	if cfg, found := ta.extraConfig["default"]; found {
		for _, dev := range cfg.Devices {
			ctntResp.Devices = append(ctntResp.Devices, &pluginapi.DeviceSpec{
				ContainerPath: dev,
				HostPath:      dev,
				Permissions:   "rwm",
			})
		}
	}

	// 至此，response 中的设备信息已经处理结束

	// 接下来处理容器中的环境变量
	// gpu-manager 需要通过修改 LD_LIBRARY_PATH 来劫持程序对 cuda 的调用
	// LD_LIBRARY_PATH
	ctntResp.Envs["LD_LIBRARY_PATH"] = "/usr/local/nvidia/lib64"
	for _, env := range container.Env {
		if env.Name == "compat32" && strings.ToLower(env.Value) == "true" {
			ctntResp.Envs["LD_LIBRARY_PATH"] = "/usr/local/nvidia/lib"
		}
	}

	// 然后通过 NVIDIA_VISIBLE_DEVICES 来让挂载的设备可见
	// NVIDIA_VISIBLE_DEVICES
	ctntResp.Envs["NVIDIA_VISIBLE_DEVICES"] = strings.Join(deviceList, ",")

	// 根据是否处于 shareMode，也就是单个 GPU 是否被共享来挂载不同的 host 目录
	if shareMode {
		// nvidia 是被劫持的库，用于 shareMode 的情况下
		ctntResp.Mounts = append(ctntResp.Mounts, &pluginapi.Mount{
			ContainerPath: "/usr/local/nvidia",
			HostPath:      types.DriverLibraryPath,
			ReadOnly:      true,
		})
	} else {
		// 非 shareMode 用原始的库即可
		ctntResp.Mounts = append(ctntResp.Mounts, &pluginapi.Mount{
			ContainerPath: "/usr/local/nvidia",
			HostPath:      types.DriverOriginLibraryPath,
			ReadOnly:      true,
		})
	}

	// 将 host 上的 /etc/gpu-manager/vm/{podUID} 挂载到容器中
	// 这个是为了容器内可以通过 vcuda.sock 和 vm 通信
	ctntResp.Mounts = append(ctntResp.Mounts, &pluginapi.Mount{
		ContainerPath: types.VCUDA_MOUNTPOINT,
		HostPath:      filepath.Join(ta.config.VirtualManagerPath, string(pod.UID)),
		ReadOnly:      true,
	})

	// 如果当前请求的容器所属 pod 没有经过 gpu-admission，还会被放到一个处理队列中
	// 这个队列会在 pkg/service/allocator/nvidia/allocator.go 的 processResult 中处理 (补打/删除标签之类的操作)
	if predicateMissed {
		ar := &allocateResult{
			pod:     pod,
			result:  PREDICATE_MISSING,
			resChan: make(chan struct{}),
		}
		ta.queue.AddRateLimited(ar)
		<-ar.resChan
	}

	// 至此，kubelet 调用 Allocate 方法就结束了
	return ctntResp, nil
}

func (ta *NvidiaTopoAllocator) requestForVCuda(podUID string) error {
	// Request for a independent directory for vcuda
	vcudaEvent := &types.VCudaRequest{
		PodUID: podUID,
		Done:   make(chan error, 1),
	}
	ta.config.VCudaRequestsQueue <- vcudaEvent
	return <-vcudaEvent.Done
}

func (ta *NvidiaTopoAllocator) recycle() {

	activePods := watchdog.GetActivePods()

	lastActivePodUids := sets.NewString()
	activePodUids := sets.NewString()
	for _, uid := range ta.allocatedPod.Pods() {
		lastActivePodUids.Insert(uid)
	}
	for uid := range activePods {
		activePodUids.Insert(uid)
	}

	// difference 出来的就是已经运行结束的 pod，可以回收分配的 gpu 资源
	podsToBeRemoved := lastActivePodUids.Difference(activePodUids)

	klog.V(5).Infof("Pods to be removed: %v", podsToBeRemoved.List())

	// 释放资源
	ta.freeGPU(podsToBeRemoved.List())
}

func (ta *NvidiaTopoAllocator) freeGPU(podUids []string) {
	for _, uid := range podUids {
		for contName, info := range ta.allocatedPod.GetCache(uid) {
			klog.V(2).Infof("Free %s(%s)", uid, contName)

			for _, devName := range info.Devices {
				id, _ := utils.GetGPUMinorID(devName)
				ta.tree.MarkFree(&nvtree.NvidiaNode{
					Meta: nvtree.DeviceMeta{
						MinorID: id,
					},
				}, info.Cores, info.Memory)
			}
		}
		ta.allocatedPod.Delete(uid)
	}
	ta.writeCheckpoint()
}

// ----------------------------------------
// gpu-manager 做的工作：
// - 为容器挂载 cuda 相关的库，包括 vcuda-control 这个项目的拦截库
// - 通过覆盖容器的 LD_LIBRARY_PATH 来将 cuda 调用指向 libcuda-control.so，这个库里面对显存和计算 api 做了拦截
// - 为容器挂载 vcuda.sock，在容器调用特定的 cuda api 时，会触发 grpc 调用，通过 vcuda.sock 和 virtual manager 通信， vm 下发容器配置。这样拦截库就知道自己应该怎么限制容器了
// ----------------------------------------

// (pluginapi.)AllocateRequest (请求)里包含了每个容器需要的设备数组
// - Allocate 是在 pod 创建时被调用，因为任何容器分配失败都会造成 pod 启动失败
// - Allocate 允许 kubelet 在 pod 环境中引入更多的 artifacts，这部分工作用 device plugin 主导。对于 gpu-manager 来说就是覆盖容器中的 LD_LIBRARY_PATH，挂载 cuda 库文件等
// - Allocate 允许 device plugin 在设备上运行特定的操作

// (pluginapi.)AllocateResponse 为每个容器返回 ContainerAllocateResponse，包括容器的环境变量，容器的挂载，容器的设备信息，容器的 annotations 信息
// 即在容器中挂载设备需要：
// - 设备相对于容器的地址
// - 设备在宿主机上的地址
// - 设备的 Cgroups 信息

// #lizard forgives
// Allocate tries to allocate GPU node for each request
func (ta *NvidiaTopoAllocator) Allocate(_ context.Context, reqs *pluginapi.AllocateRequest) (*pluginapi.AllocateResponse, error) {
	ta.Lock()
	defer ta.Unlock()

	var (
		reqCount           uint
		candidatePod       *v1.Pod
		candidateContainer *v1.Container
		found              bool
	)
	if len(reqs.ContainerRequests) < 1 {
		return nil, fmt.Errorf("empty container request")
	}

	// k8s send allocate request for one container at a time
	// 取 Allocate 中的第一个 ContainerRequest，通过上一句的注释，k8s 一次只为一个容器发送分配请求
	req := reqs.ContainerRequests[0]
	resps := &pluginapi.AllocateResponse{}
	reqCount = uint(len(req.DevicesIDs))

	klog.V(4).Infof("Request GPU device: %s", strings.Join(req.DevicesIDs, ","))

	// 因为 k8s 一次请求只针对一个容器，所以这里的 unfinishedPod 指的是还有部分容器尚未分配的 pod
	if ta.unfinishedPod != nil {
		// 候选 pod
		candidatePod = ta.unfinishedPod
		// 从已分配的 pod 中查找
		cache := ta.allocatedPod.GetCache(string(candidatePod.UID))
		if cache == nil {
			msg := fmt.Sprintf("failed to find pod %s in cache", candidatePod.UID)
			klog.Infof(msg)
			return nil, fmt.Errorf(msg)
		}
		for i, c := range candidatePod.Spec.Containers {
			if _, ok := cache[c.Name]; ok {
				continue
			}

			if !utils.IsGPURequiredContainer(&c) {
				continue
			}

			if reqCount != utils.GetGPUResourceOfContainer(&candidatePod.Spec.Containers[i], types.VCoreAnnotation) {
				msg := fmt.Sprintf("allocation request mismatch for pod %s, reqs %v", candidatePod.UID, reqs)
				klog.Infof(msg)
				return nil, fmt.Errorf(msg)
			}
			// 候选的容器 (应该就是待分配资源的容器)
			candidateContainer = &candidatePod.Spec.Containers[i]
			found = true
			break

			// 上面的代码遍历了这个 pod 的容器列表，然后和缓存中的容器对比
			// 如果没有分配并需要 gpu 资源，且容器请求的资源量和当前的分配请求一致
			// 则认定这个容器就是接下来要为之分配的候选人
		}
	} else {
		// 获取候选的 pod，候选的 pod 是当前节点上需要 GPU，但没有分配，且应该不会删除的 pod
		pods, err := getCandidatePods(ta.k8sClient, ta.config.Hostname)
		if err != nil {
			msg := fmt.Sprintf("Failed to find candidate pods due to %v", err)
			klog.Infof(msg)
			return nil, fmt.Errorf(msg)
		}

		for _, pod := range pods {
			if found {
				break
			}
			for i, c := range pod.Spec.Containers {
				if !utils.IsGPURequiredContainer(&c) {
					continue
				}
				podCache := ta.allocatedPod.GetCache(string(pod.UID))
				if podCache != nil {
					if _, ok := podCache[c.Name]; ok {
						klog.Infof("container %s of pod %s has been allocate, continue to next", c.Name, pod.UID)
						continue
					}
				}
				if utils.GetGPUResourceOfContainer(&pod.Spec.Containers[i], types.VCoreAnnotation) == reqCount {
					klog.Infof("Found candidate Pod %s(%s) with device count %d", pod.UID, c.Name, reqCount)
					candidatePod = pod
					candidateContainer = &pod.Spec.Containers[i]
					found = true
					break
				}
			}
		}
	}

	if found {
		// get vmemory info from container spec
		// 拿到容器的 vmemory 信息，vmemoery 是根据数量划分的，1 vmemory = 256 MB memory = 1 deviceID
		// 所以这里请求多少个 vmemory 就会有多少个 deviceID
		vmemory := utils.GetGPUResourceOfContainer(candidateContainer, types.VMemoryAnnotation)
		for i := 0; i < int(vmemory); i++ {
			req.DevicesIDs = append(req.DevicesIDs, types.VMemoryAnnotation)
		}

		// 调用 allocateOne 为单个容器进行真正的分配工作
		resp, err := ta.allocateOne(candidatePod, candidateContainer, req)
		if err != nil {
			klog.Errorf(err.Error())
			return nil, err
		}
		resps.ContainerResponses = append(resps.ContainerResponses, resp)
	} else {
		msg := fmt.Sprintf("candidate pod not found for request %v, allocation failed", reqs)
		klog.Infof(msg)
		return nil, fmt.Errorf(msg)
	}

	return resps, nil
}

//ListAndWatch is not implement
func (ta *NvidiaTopoAllocator) ListAndWatch(e *pluginapi.Empty, s pluginapi.DevicePlugin_ListAndWatchServer) error {
	return fmt.Errorf("not implement")
}

//ListAndWatchWithResourceName send devices for request resource back to server
func (ta *NvidiaTopoAllocator) ListAndWatchWithResourceName(resourceName string, e *pluginapi.Empty, s pluginapi.DevicePlugin_ListAndWatchServer) error {
	devs := make([]*pluginapi.Device, 0)
	for _, dev := range ta.capacity() {
		if strings.HasPrefix(dev.ID, resourceName) {
			devs = append(devs, dev)
		}
	}

	s.Send(&pluginapi.ListAndWatchResponse{Devices: devs})

	// We don't send unhealthy state
	for {
		time.Sleep(time.Second)
	}

	klog.V(2).Infof("ListAndWatch %s exit", resourceName)

	return nil
}

//GetDevicePluginOptions returns empty DevicePluginOptions
func (ta *NvidiaTopoAllocator) GetDevicePluginOptions(ctx context.Context, e *pluginapi.Empty) (*pluginapi.DevicePluginOptions, error) {
	return &pluginapi.DevicePluginOptions{PreStartRequired: true}, nil
}

//PreStartContainer find the podUID by comparing request deviceids with deviceplugin
//checkpoint data, then checks the validation of allocation of the pod.
//Update pod annotation if check success, otherwise evict the pod.
func (ta *NvidiaTopoAllocator) PreStartContainer(ctx context.Context, req *pluginapi.PreStartContainerRequest) (*pluginapi.PreStartContainerResponse, error) {
	ta.Lock()
	defer ta.Unlock()
	klog.V(2).Infof("get preStartContainer call from k8s, req: %+v", req)
	var (
		podUID        string
		containerName string
		vcore         int64
		vmemory       int64
		//devices       []string
	)

	// try to get podUID, containerName, vcore and vmemory from kubelet deviceplugin checkpoint file
	cp, err := utils.GetCheckpointData(ta.config.DevicePluginPath)
	if err != nil {
		msg := fmt.Sprintf("%s, failed to read from checkpoint file due to %v",
			types.PreStartContainerCheckErrMsg, err)
		klog.Infof(msg)
		return nil, fmt.Errorf(msg)
	}
	for _, entry := range cp.PodDeviceEntries {
		if entry.ResourceName == types.VCoreAnnotation &&
			utils.IsStringSliceEqual(req.DevicesIDs, entry.DeviceIDs) {
			podUID = entry.PodUID
			containerName = entry.ContainerName
			vcore = int64(len(entry.DeviceIDs))
			break
		}
	}

	for _, entry := range cp.PodDeviceEntries {
		if entry.PodUID == podUID &&
			entry.ContainerName == containerName &&
			entry.ResourceName == types.VMemoryAnnotation {
			vmemory = int64(len(entry.DeviceIDs))
			break
		}
	}

	if podUID == "" || containerName == "" {
		msg := fmt.Sprintf("%s, failed to get pod from deviceplugin checkpoint for PreStartContainer request %v",
			types.PreStartContainerCheckErrMsg, req)
		klog.Infof(msg)
		return nil, fmt.Errorf(msg)
	}
	pod, ok := watchdog.GetActivePods()[podUID]
	if !ok {
		msg := fmt.Sprintf("%s, failed to get pod %s in watchdog", types.PreStartContainerCheckErrMsg, podUID)
		klog.Infof(msg)
		return nil, fmt.Errorf(msg)
	}

	err = ta.preStartContainerCheck(podUID, containerName, vcore, vmemory)
	if err != nil {
		klog.Infof(err.Error())
		ta.queue.AddRateLimited(&allocateResult{
			pod:     pod,
			result:  ALLOCATE_FAIL,
			message: err.Error(),
			reason:  types.PreStartContainerCheckErrType,
		})
		return nil, err
	}

	// allocation check ok, request for VCuda to setup vGPU environment
	err = ta.requestForVCuda(podUID)
	if err != nil {
		msg := fmt.Sprintf("failed to setup VCuda for pod %s(%s) due to %v", podUID, containerName, err)
		klog.Infof(msg)
		return nil, fmt.Errorf(msg)
	}

	// prestart check pass, update pod annotation
	ar := &allocateResult{
		pod:     pod,
		result:  ALLOCATE_SUCCESS,
		resChan: make(chan struct{}),
	}
	ta.queue.AddRateLimited(ar)
	<-ar.resChan

	return &pluginapi.PreStartContainerResponse{}, nil
}

func (ta *NvidiaTopoAllocator) preStartContainerCheck(podUID string, containerName string, vcore int64, vmemory int64) error {
	cache := ta.allocatedPod.GetCache(podUID)
	if cache == nil {
		msg := fmt.Sprintf("%s, failed to get pod %s from allocatedPod cache",
			types.PreStartContainerCheckErrMsg, podUID)
		klog.Infof(msg)
		return fmt.Errorf(msg)
	}

	if c, ok := cache[containerName]; !ok {
		msg := fmt.Sprintf("%s, failed to get container %s of pod %s from allocatedPod cache",
			types.PreStartContainerCheckErrMsg, containerName, podUID)
		klog.Infof(msg)
		return fmt.Errorf(msg)
	} else if c.Memory != vmemory*types.MemoryBlockSize || c.Cores != vcore {
		// request and cache mismatch, evict the pod
		msg := fmt.Sprintf("%s, pod %s container %s requset mismatch from cache. req: vcore %d vmemory %d; cache: vcore %d vmemory %d",
			types.PreStartContainerCheckErrMsg, podUID, containerName, vcore, vmemory*types.MemoryBlockSize, c.Cores, c.Memory)
		klog.Infof(msg)
		return fmt.Errorf(msg)
	} else {
		devices := c.Devices
		if (vcore < nvtree.HundredCore && len(devices) != 1) ||
			(vcore >= nvtree.HundredCore && len(devices) != int(vcore/nvtree.HundredCore)) {
			msg := fmt.Sprintf("allocated devices mismatch, request for %d vcore, allocate %v", vcore, devices)
			klog.Infof(msg)
			return fmt.Errorf(msg)
		}
	}
	return nil
}

func (ta *NvidiaTopoAllocator) processNextResult() bool {
	// Wait until there is a new item in the working queue
	key, quit := ta.queue.Get()
	if quit {
		return false
	}
	// Tell the queue that we are done with processing this key. This unblocks the key for other workers
	// This allows safe parallel processing because two pods with the same key are never processed in
	// parallel.
	defer ta.queue.Done(key)

	result, ok := key.(*allocateResult)
	if !ok {
		klog.Infof("Failed to process result: %v, unable to translate to allocateResult", key)
		return true
	}
	// Invoke the method containing the business logic
	err := ta.processResult(result)
	// Handle the error if something went wrong during the execution of the business logic
	if err != nil {
		ta.queue.AddRateLimited(key)
		return true
	}

	ta.queue.Forget(key)
	return true
}

func (ta *NvidiaTopoAllocator) processResult(ar *allocateResult) error {
	switch ar.result {
	case ALLOCATE_SUCCESS:
		annotationMap, err := ta.getReadyAnnotations(ar.pod, true)
		if err != nil {
			msg := fmt.Sprintf("failed to get ready annotation of pod %s due to %s", ar.pod.UID, err.Error())
			klog.Infof(msg)
			return fmt.Errorf(msg)
		}
		err = patchPodWithAnnotations(ta.k8sClient, ar.pod, annotationMap)
		if err != nil {
			msg := fmt.Sprintf("add annotation for pod %s failed due to %s", ar.pod.UID, err.Error())
			klog.Infof(msg)
			return fmt.Errorf(msg)
		}
		close(ar.resChan)
	case ALLOCATE_FAIL:
		// free GPU devices that are already allocated to this pod
		ta.freeGPU([]string{string(ar.pod.UID)})

		ar.pod.Status = v1.PodStatus{
			Phase:   v1.PodFailed,
			Message: ar.message,
			Reason:  ar.reason,
		}
		ar.pod.Annotations = nil
		err := ta.updatePodStatus(ar.pod)
		if err != nil {
			msg := fmt.Sprintf("failed to set status of pod %s to PodFailed due to %s", ar.pod.UID, err.Error())
			klog.Infof(msg)
			return fmt.Errorf(msg)
		}
	case PREDICATE_MISSING:
		annotationMap, err := ta.getReadyAnnotations(ar.pod, false)
		err = patchPodWithAnnotations(ta.k8sClient, ar.pod, annotationMap)
		if err != nil {
			msg := fmt.Sprintf("add annotation for pod %s failed due to %s", ar.pod.UID, err.Error())
			klog.Infof(msg)
			return fmt.Errorf(msg)
		}
		close(ar.resChan)
	default:
		klog.Infof("unknown allocation result %d for pod %s", ar.result, ar.pod.UID)
	}
	return nil
}

func (ta *NvidiaTopoAllocator) getReadyAnnotations(pod *v1.Pod, assigned bool) (annotationMap map[string]string, err error) {
	// ta.Lock()
	// defer ta.Unlock()
	cache := ta.allocatedPod.GetCache(string(pod.UID))
	if cache == nil {
		msg := fmt.Sprintf("failed to get pod %s from allocatedPod cache", pod.UID)
		klog.Infof(msg)
		return nil, fmt.Errorf(msg)
	}

	annotationMap = make(map[string]string)
	for i, c := range pod.Spec.Containers {
		if !utils.IsGPURequiredContainer(&c) {
			continue
		}
		var devices []string
		containerCache, ok := cache[c.Name]
		if !ok {
			msg := fmt.Sprintf("failed to get container %s of pod %s from allocatedPod cache", c.Name, pod.UID)
			klog.Infof(msg)
			err = fmt.Errorf(msg)
			continue
		}

		devices = make([]string, len(containerCache.Devices))
		copy(devices, containerCache.Devices)
		for j, dev := range devices {
			strs := strings.Split(dev, types.NvidiaDevicePrefix)
			devices[j] = strs[len(strs)-1]
		}
		predicateIndexStr := strings.Join(devices, ",")
		annotationMap[types.PredicateGPUIndexPrefix+strconv.Itoa(i)] = predicateIndexStr
	}
	annotationMap[types.GPUAssigned] = strconv.FormatBool(assigned)

	return annotationMap, nil
}

func (ta *NvidiaTopoAllocator) updatePodStatus(pod *v1.Pod) error {
	klog.V(4).Infof("Try to update status of pod %s", pod.UID)

	err := wait.PollImmediate(time.Second, waitTimeout, func() (bool, error) {
		_, err := ta.k8sClient.CoreV1().Pods(pod.Namespace).UpdateStatus(pod)
		if err == nil {
			return true, nil
		}
		if utils.ShouldRetry(err) {
			klog.Infof("update status of pod %s failed due to %v, try again", pod.UID, err)
			newPod, err := ta.k8sClient.CoreV1().Pods(pod.Namespace).Get(pod.Name, metav1.GetOptions{})
			if err != nil {
				return false, err
			}
			newPod.Status = pod.Status
			pod = newPod
			return false, nil
		}
		klog.V(4).Infof("Failed to update status of pod %s due to %v", pod.UID, err)
		return false, err
	})
	if err != nil {
		klog.Errorf("failed to update status of pod %s due to %v", pod.UID, err)
		return err
	}

	return nil
}

// delete pod if it is controlled by workloads like deployment, ignore naked pod
func (ta *NvidiaTopoAllocator) deletePodWithOwnerRef(pod *v1.Pod) error {
	// free GPU devices that are already allocated to this pod
	ta.freeGPU([]string{string(pod.UID)})

	if len(pod.OwnerReferences) > 0 {
		for _, ownerReference := range pod.OwnerReferences {
			// ignore pod if it is owned by another pod
			if ownerReference.Kind == pod.Kind {
				return nil
			}
		}
		// delete the pod
		klog.V(4).Infof("Try to delete pod %s", pod.UID)
		err := wait.PollImmediate(time.Second, waitTimeout, func() (bool, error) {
			err := ta.k8sClient.CoreV1().Pods(pod.Namespace).Delete(pod.Name, &metav1.DeleteOptions{})
			if err == nil {
				return true, nil
			}
			if utils.ShouldRetry(err) {
				return false, nil
			}
			klog.V(4).Infof("Failed to delete pod %s due to %v", pod.UID, err)
			return false, err
		})
		if err != nil {
			klog.Errorf("failed to delete pod %s due to %v", pod.UID, err)
			return err
		}
	}

	return nil
}

func patchPodWithAnnotations(client kubernetes.Interface, pod *v1.Pod, annotationMap map[string]string) error {
	// update annotations by patching to the pod
	type patchMetadata struct {
		Annotations map[string]string `json:"annotations"`
	}
	type patchPod struct {
		Metadata patchMetadata `json:"metadata"`
	}
	payload := patchPod{
		Metadata: patchMetadata{
			Annotations: annotationMap,
		},
	}

	payloadBytes, _ := json.Marshal(payload)
	err := wait.PollImmediate(time.Second, waitTimeout, func() (bool, error) {
		_, err := client.CoreV1().Pods(pod.Namespace).Patch(pod.Name, k8stypes.StrategicMergePatchType, payloadBytes)
		if err == nil {
			return true, nil
		}
		if utils.ShouldRetry(err) {
			return false, nil
		}

		return false, err
	})
	if err != nil {
		msg := fmt.Sprintf("failed to add annotation %v to pod %s due to %s", annotationMap, pod.UID, err.Error())
		klog.Infof(msg)
		return fmt.Errorf(msg)
	}
	return nil
}

func vDeviceAnnotationStr(nodes []*nvtree.NvidiaNode) string {
	str := make([]string, 0)
	for _, node := range nodes {
		str = append(str, node.MinorName())
	}

	return strings.Join(str, ",")
}

func getCandidatePods(client kubernetes.Interface, hostname string) ([]*v1.Pod, error) {
	candidatePods := []*v1.Pod{}
	// 先获取节点上所有的 pod
	allPods, err := getPodsOnNode(client, hostname, string(v1.PodPending))
	if err != nil {
		return candidatePods, err
	}
	for _, pod := range allPods {
		current := pod
		// 从节点上的 pod 中选取需要 GPU 但还没分配 GPU，且应该不会删除的 pod
		if utils.IsGPURequiredPod(&current) && !utils.IsGPUAssignedPod(&current) && !utils.ShouldDelete(&current) {
			candidatePods = append(candidatePods, &current)
		}
	}

	if klog.V(4) {
		for _, pod := range candidatePods {
			klog.Infof("candidate pod %s in ns %s with timestamp %d is found.",
				pod.Name,
				pod.Namespace,
				utils.GetPredicateTimeOfPod(pod))
		}
	}

	// 得到一个候选 pod 列表，对这个列表根据时间排序，即可拿到最先被调度的 pod
	// 这里默认了一个前提，最先调度的 pod 会最先发出分配请求
	// 排序依据的时间有两个选择：预选时间和创建时间
	return OrderPodsdByPredicateTime(candidatePods), nil
}

func getPodsOnNode(client kubernetes.Interface, hostname string, phase string) ([]v1.Pod, error) {
	if len(hostname) == 0 {
		hostname, _ = os.Hostname()
	}
	var (
		selector fields.Selector
		pods     []v1.Pod
	)

	if phase != "" {
		selector = fields.SelectorFromSet(fields.Set{"spec.nodeName": hostname, "status.phase": phase})
	} else {
		selector = fields.SelectorFromSet(fields.Set{"spec.nodeName": hostname})
	}
	var (
		podList *v1.PodList
		err     error
	)

	err = wait.PollImmediate(time.Second, time.Minute, func() (bool, error) {
		podList, err = client.CoreV1().Pods(v1.NamespaceAll).List(metav1.ListOptions{
			FieldSelector: selector.String(),
			LabelSelector: labels.Everything().String(),
		})
		if err != nil {
			return false, err
		}
		return true, nil
	})
	if err != nil {
		return pods, fmt.Errorf("failed to get Pods on node %s because: %v", hostname, err)
	}

	klog.V(9).Infof("all pods on this node: %v", podList.Items)
	for _, pod := range podList.Items {
		pods = append(pods, pod)
	}

	return pods, nil
}

// make the pod ordered by predicate time
func OrderPodsdByPredicateTime(pods []*v1.Pod) []*v1.Pod {
	newPodList := make(PodsOrderedByPredicateTime, 0, len(pods))
	for _, v := range pods {
		newPodList = append(newPodList, v)
	}
	sort.Sort(newPodList)
	return []*v1.Pod(newPodList)
}

type PodsOrderedByPredicateTime []*v1.Pod

func (pods PodsOrderedByPredicateTime) Len() int {
	return len(pods)
}

func (pods PodsOrderedByPredicateTime) Less(i, j int) bool {
	return utils.GetPredicateTimeOfPod(pods[i]) <= utils.GetPredicateTimeOfPod(pods[j])
}

func (pods PodsOrderedByPredicateTime) Swap(i, j int) {
	pods[i], pods[j] = pods[j], pods[i]
}

func (ta *NvidiaTopoAllocator) readCheckpoint() {
	data, err := ta.checkpointManager.Read()
	if err != nil {
		klog.Warningf("Failed to read from checkpoint due to %s", err.Error())
		return
	}
	err = json.Unmarshal(data, ta.allocatedPod)
	if err != nil {
		klog.Warningf("Failed to unmarshal data from checkpoint due to %s", err.Error())
	}
}

func (ta *NvidiaTopoAllocator) writeCheckpoint() {
	data, err := json.Marshal(ta.allocatedPod)
	if err != nil {
		klog.Warningf("Failed to marshal allocatedPod due to %s", err.Error())
		return
	}
	err = ta.checkpointManager.Write(data)
	if err != nil {
		klog.Warningf("Failed to write checkpoint due to %s", err.Error())
	}
}

// 问题：
//
// 问题 A: 为什么要大费周章的通过 grpc，直接挂载容器配置文件可行吗？
// 总结： 这一点在上面的阅读中可以发现，这时候容器本身对自己应该限制多少的 gpu 资源调用并不知道。这个问题得和 B/C/D 问题结合来看。因为做　Allocate 调用时，kubelet 并没有告知此时在为哪个容器请求配置。因此只能根据请求的资源量以及 Pod 的 predicateTime 或 createTime 来判断。这个是无法保证一定准确的，因此此时容器的具体资源配置也无法确定。可能这就是要通过 grpc 而不是挂载容器配置文件的原因吧。
//
// 问题 B: 如果一个 Pod 中有多个 vcore 请求一致，但是 vmemory 不同的容器，这里只通过 vcore 的请求量来判断，可以保证这个分配请求和我们的候选容器能对的上吗？
// 总结：问题 B 是在 unfinisedPod 中查找当前请求的容器。只要能保证 unfinishedPod 是正确的（问题 D 说明不能保证），那么就可以保证容器是对的上的（问题 C 保证了这个结论）。
//
// 问题 C: AllocateRequest 是按照 Pod 中的容器顺序来的?
// 总结：对于这个问题，最好的回答方式是去看 kubelet 的源代码。
//
// for _, container := range pod.Spec.Containers {
//     if err := m.allocateContainerResources(pod, &container, devicesToReuse); err != nil {
//         return err
//     }
//     m.podDevices.removeContainerAllocatedResources(string(pod.UID), container.Name, devicesToReuse)
// }
//
// 这边做 Allocate 的时候，是顺序遍历 Pod 中的容器，因此这个问题的答案是肯定的。
//
// 问题 D: 遍历当前节点上的所有 pod，然后挑出需要 gpu 资源的 pod，根据 predicatedTime 或 createTime 排序。然后再从这些 pod 中， 按顺序挑出符合这次请求的容器，怎么保证挑出来的容器就是这次分配请求的呢？
// 总结：我觉得回答这个问题，需要确定两个大前提，一是 Pod 从创建到发起 Allocate 的过程，都是顺序的。这样就能保证当调用 Allocate 对应的 Pod 永远是尚未分配到资源的第一个。二是在一个 Pod 中，为每个容器 Allocate 时，也是顺序的，这一点在问题 C 中得到确认。
// 但是实际上，第一个前提是不能保证的，在 Pod bind 到节点时，这个是并发执行的。因此可以得出一个结论：在这个阶段无法保证 Allocate 请求和我们的候选容器是对应关系。关于这一点我也提了个 issue：a question about Allocate for a container?。官方也给了回答，因为这个原因 gpu manager 有时候会报 UnexpectedAdmissionError 错误。
// 所以根据问题 4，我们还要使用 gpu-admission 这个项目，来保证该阶段的正确性，具体机制还得等到看 gpu-admission 的时候才能知道了。
// 其实以上四个问题都是因为 kubelet 的 Allocate 请求不会带上正在分配的容器。所以需要一系列的查找方式来确定具体的容器。
