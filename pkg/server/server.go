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

package server

import (
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"net/http/pprof"
	"os"
	"path"
	"path/filepath"
	"strconv"
	"syscall"
	"time"

	displayapi "tkestack.io/gpu-manager/pkg/api/runtime/display"
	"tkestack.io/gpu-manager/pkg/config"
	deviceFactory "tkestack.io/gpu-manager/pkg/device"
	containerRuntime "tkestack.io/gpu-manager/pkg/runtime"
	allocFactory "tkestack.io/gpu-manager/pkg/services/allocator"

	// Register allocator controller
	_ "tkestack.io/gpu-manager/pkg/services/allocator/register"
	"tkestack.io/gpu-manager/pkg/services/display"
	vitrual_manager "tkestack.io/gpu-manager/pkg/services/virtual-manager"
	"tkestack.io/gpu-manager/pkg/services/volume"
	"tkestack.io/gpu-manager/pkg/services/watchdog"
	"tkestack.io/gpu-manager/pkg/types"
	"tkestack.io/gpu-manager/pkg/utils"

	systemd "github.com/coreos/go-systemd/daemon"
	google_protobuf1 "github.com/golang/protobuf/ptypes/empty"
	"github.com/grpc-ecosystem/grpc-gateway/runtime"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"golang.org/x/net/context"
	"google.golang.org/grpc"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/klog"
	pluginapi "k8s.io/kubelet/pkg/apis/deviceplugin/v1beta1"
)

type managerImpl struct {

	// config 中的参数同 cmd/manager/options/options.go 中传入的参数
	config *config.Config

	// 负责在容器调度到节点上，为其分配具体的设备资源
	// allocator 实现了探测节点上的 GPU 拓扑架构
	// 以最佳性能，最少碎片为目的的使用最优方案进行资源分配
	allocator allocFactory.GPUTopoService
	// 将 gpu 的使用情况输出，方便查看 (?哪里用上了?)
	displayer *display.Display
	// 负责 vgpu 分配后的管理工作
	virtualManager *vitrual_manager.VirtualManager

	// 包含 vcore vmemory，这两种资源是以 device plugin 的方式进行注册，所以他们需要启动 grpc server
	bundleServer map[string]ResourceServer
	// 将 gpu display server 注册到 grpc server 中
	srv *grpc.Server
}

//NewManager creates and returns a new managerImpl struct
func NewManager(cfg *config.Config) Manager {
	manager := &managerImpl{
		config:       cfg,
		bundleServer: make(map[string]ResourceServer),
		srv:          grpc.NewServer(),
	}

	return manager
}

// Ready tells the manager whether all bundle servers are truely running
func (m *managerImpl) Ready() bool {
	readyServers := 0

	for _, ins := range m.bundleServer {
		if err := utils.WaitForServer(ins.SocketName()); err == nil {
			readyServers++
			klog.V(2).Infof("Server %s is ready, readyServers: %d", ins.SocketName(), readyServers)
			continue
		}

		return false
	}

	return readyServers > 0 && readyServers == len(m.bundleServer)
}

// #lizard forgives
func (m *managerImpl) Run() error {
	if err := m.validExtraConfig(m.config.ExtraConfigPath); err != nil {
		klog.Errorf("Can not load extra config, err %s", err)

		return err
	}

	if m.config.Driver == "" {
		return fmt.Errorf("you should define a driver")
	}

	if len(m.config.VolumeConfigPath) > 0 {
		volumeManager, err := volume.NewVolumeManager(m.config.VolumeConfigPath, m.config.EnableShare)
		if err != nil {
			klog.Errorf("Can not create volume managerImpl, err %s", err)
			return err
		}

		if err := volumeManager.Run(); err != nil {
			klog.Errorf("Can not start volume managerImpl, err %s", err)
			return err
		}
	}

	sent, err := systemd.SdNotify(true, "READY=1\n")
	if err != nil {
		klog.Errorf("Unable to send systemd daemon successful start message: %v\n", err)
	}

	if !sent {
		klog.Errorf("Unable to set Type=notify in systemd service file?")
	}

	var (
		client    *kubernetes.Clientset
		clientCfg *rest.Config
	)

	clientCfg, err = clientcmd.BuildConfigFromFlags("", m.config.KubeConfig)
	if err != nil {
		return fmt.Errorf("invalid client config: err(%v)", err)
	}

	client, err = kubernetes.NewForConfig(clientCfg)
	if err != nil {
		return fmt.Errorf("can not generate client from config: error(%v)", err)
	}

	containerRuntimeManager, err := containerRuntime.NewContainerRuntimeManager(
		m.config.CgroupDriver, m.config.ContainerRuntimeEndpoint, m.config.RequestTimeout)
	if err != nil {
		klog.Errorf("can't create container runtime manager: %v", err)
		return err
	}
	klog.V(2).Infof("Container runtime manager is running")

	watchdog.NewPodCache(client, m.config.Hostname)
	klog.V(2).Infof("Watchdog is running")

	labeler := watchdog.NewNodeLabeler(client.CoreV1(), m.config.Hostname, m.config.NodeLabels)
	if err := labeler.Run(); err != nil {
		return err
	}

	m.virtualManager = vitrual_manager.NewVirtualManager(m.config, containerRuntimeManager)
	m.virtualManager.Run()

	treeInitFn := deviceFactory.NewFuncForName(m.config.Driver)
	tree := treeInitFn(m.config)

	tree.Init("")
	tree.Update()

	// 初始化资源分配器，这里的 initAllocator 对应的方法是 func NewNvidiaTopoAllocator(config *config.Config, tree device.GPUTree, k8sClient kubernetes.Interface) allocator.GPUTopoService
	initAllocator := allocFactory.NewFuncForName(m.config.Driver)
	if initAllocator == nil {
		return fmt.Errorf("can not find allocator for %s", m.config.Driver)
	}

	m.allocator = initAllocator(m.config, tree, client)
	m.displayer = display.NewDisplay(m.config, tree, containerRuntimeManager)

	klog.V(2).Infof("Starting the GRPC server, driver %s, queryPort %d", m.config.Driver, m.config.QueryPort)
	m.setupGRPCService()
	mux, err := m.setupGRPCGatewayService()
	if err != nil {
		return err
	}
	m.setupMetricsService(mux)

	go func() {
		displayListenHandler := net.JoinHostPort(m.config.QueryAddr, strconv.Itoa(m.config.QueryPort))
		if err := http.ListenAndServe(displayListenHandler, mux); err != nil {
			klog.Fatalf("failed to serve connections: %v", err)
		}
	}()

	return m.runServer()
}

func (m *managerImpl) setupGRPCService() {
	vcoreServer := newVcoreServer(m)
	vmemoryServer := newVmemoryServer(m)

	m.bundleServer[types.VCoreAnnotation] = vcoreServer
	m.bundleServer[types.VMemoryAnnotation] = vmemoryServer

	displayapi.RegisterGPUDisplayServer(m.srv, m)
}

func (m *managerImpl) setupGRPCGatewayService() (*http.ServeMux, error) {
	mux := http.NewServeMux()
	displayMux := runtime.NewServeMux()

	mux.Handle("/", displayMux)
	mux.HandleFunc("/debug/pprof/", pprof.Index)

	go func() {
		if err := displayapi.RegisterGPUDisplayHandlerFromEndpoint(context.Background(), displayMux, types.ManagerSocket, utils.DefaultDialOptions); err != nil {
			klog.Fatalf("Register display service failed, error %s", err)
		}
	}()

	return mux, nil
}

func (m *managerImpl) setupMetricsService(mux *http.ServeMux) {
	r := prometheus.NewRegistry()

	r.MustRegister(m.displayer)

	mux.Handle("/metric", promhttp.HandlerFor(r, promhttp.HandlerOpts{ErrorHandling: promhttp.ContinueOnError}))
}

func (m *managerImpl) runServer() error {
	for name, srv := range m.bundleServer {
		klog.V(2).Infof("Server %s is running", name)
		go srv.Run()
	}

	err := syscall.Unlink(types.ManagerSocket)
	if err != nil && !os.IsNotExist(err) {
		return err
	}

	l, err := net.Listen("unix", types.ManagerSocket)
	if err != nil {
		return err
	}

	klog.V(2).Infof("Server is ready at %s", types.ManagerSocket)

	return m.srv.Serve(l)
}

func (m *managerImpl) Stop() {
	for name, srv := range m.bundleServer {
		klog.V(2).Infof("Server %s is stopping", name)
		srv.Stop()
	}
	m.srv.Stop()
	klog.Fatal("Stop server")
}

func (m *managerImpl) validExtraConfig(path string) error {
	if path != "" {
		if _, err := os.Stat(path); err != nil {
			return err
		}

		file, err := os.Open(path)
		if err != nil {
			return err
		}

		cfg := make(map[string]*config.ExtraConfig)
		if err := json.NewDecoder(file).Decode(&cfg); err != nil {
			return err
		}
	}

	return nil
}

/** device plugin interface */
func (m *managerImpl) Allocate(ctx context.Context, reqs *pluginapi.AllocateRequest) (*pluginapi.AllocateResponse, error) {
	return m.allocator.Allocate(ctx, reqs)
}

func (m *managerImpl) ListAndWatchWithResourceName(resourceName string, e *pluginapi.Empty, s pluginapi.DevicePlugin_ListAndWatchServer) error {
	return m.allocator.ListAndWatchWithResourceName(resourceName, e, s)
}

func (m *managerImpl) GetDevicePluginOptions(ctx context.Context, e *pluginapi.Empty) (*pluginapi.DevicePluginOptions, error) {
	return m.allocator.GetDevicePluginOptions(ctx, e)
}

func (m *managerImpl) PreStartContainer(ctx context.Context, req *pluginapi.PreStartContainerRequest) (*pluginapi.PreStartContainerResponse, error) {
	return m.allocator.PreStartContainer(ctx, req)
}

/** statistics interface */
func (m *managerImpl) PrintGraph(ctx context.Context, req *google_protobuf1.Empty) (*displayapi.GraphResponse, error) {
	return m.displayer.PrintGraph(ctx, req)
}

func (m *managerImpl) PrintUsages(ctx context.Context, req *google_protobuf1.Empty) (*displayapi.UsageResponse, error) {
	return m.displayer.PrintUsages(ctx, req)
}

func (m *managerImpl) Version(ctx context.Context, req *google_protobuf1.Empty) (*displayapi.VersionResponse, error) {
	return m.displayer.Version(ctx, req)
}

// Device Plugin 的注册

// Device Plugin 工作时序图
// [device plugin]										 [kubelet]						[api server]				 [user]
//		| -> 注册 (Unix socket, api server. ResourceName) ->	|								|							|
//		| -> ListAnWatch 发送当前节点上的设备资源 Foo		 ->	|								|							|
//		|													| -> 通过节点状态更新资源状态     ->	|							|
//		|													|								| <- 创建 pod，请求 Foo 资源<-	|
//		|													| <- 根据节点状态进行 pod 调度	 <-	|							|
//		| <- 调用 Allocate 进行资源分配						|								|							|
// [device plugin]										 [kubelet]						[api server]				 [user]

// gpu-manager 的注册方法
// 这里分别注册了 vcuda 和 vmemory
// vcuda 和 vmemory 的 Allocate 方法都指向了同一个方法，写在 service/allocator/nvidia/allocator.go 中 (?怎么看出来的?)
func (m *managerImpl) RegisterToKubelet() error {
	socketFile := filepath.Join(m.config.DevicePluginPath, types.KubeletSocket)
	dialOptions := []grpc.DialOption{grpc.WithInsecure(), grpc.WithDialer(utils.UnixDial), grpc.WithBlock(), grpc.WithTimeout(time.Second * 5)}

	conn, err := grpc.Dial(socketFile, dialOptions...)
	if err != nil {
		return err
	}
	defer conn.Close()

	client := pluginapi.NewRegistrationClient(conn)

	for _, srv := range m.bundleServer {
		req := &pluginapi.RegisterRequest{
			Version:      pluginapi.Version,
			Endpoint:     path.Base(srv.SocketName()),
			ResourceName: srv.ResourceName(),
			Options:      &pluginapi.DevicePluginOptions{PreStartRequired: true},
		}

		klog.V(2).Infof("Register to kubelet with endpoint %s", req.Endpoint)
		_, err = client.Register(context.Background(), req)
		if err != nil {
			return err
		}
	}

	return nil
}

// 至此，gpu-manager 的启动流程结束
// 接下来 gpu-manager 的职责就是等待 kubelet 通过 grpc 调用

//
//										|- vcore
//										|- vmemroy
//					|- server			|- display
//										|- metrics
//
//					|- virtualmanger	|- vcuda
//
//	｜- gpu-manager	|- volume manager
//
// 					|- allocator
//
//										｜- link
//					|- nvidia 的拓扑感知	｜- share
//										｜- fragment
