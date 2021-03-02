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

package volume

import (
	"debug/elf"
	"encoding/json"
	"fmt"
	"os"
	"path"
	"path/filepath"
	"strconv"
	"strings"

	"tkestack.io/gpu-manager/pkg/services/volume/ldcache"
	"tkestack.io/gpu-manager/pkg/types"

	"k8s.io/klog"
)

// VolumeManager manages volumes used by containers running GPU application
type VolumeManager struct {
	Config  []Config `json:"volume,omitempty"`
	cfgPath string

	cudaControlFile string
	cudaSoname      map[string]string
	mlSoName        map[string]string
	share           bool
}

type components map[string][]string

// Config contains volume details in config file
type Config struct {
	Name       string     `json:"name,omitempty"`
	Mode       string     `json:"mode,omitempty"`
	Components components `json:"components,omitempty"`
	BasePath   string     `json:"base,omitempty"`
}

const (
	binDir   = "bin"
	lib32Dir = "lib"
	lib64Dir = "lib64"
)

type volumeDir struct {
	name  string
	files []string
}

// Volume contains directory and file info of volume
type Volume struct {
	Path string
	dirs []volumeDir
}

// VolumeMap stores Volume for each type
type VolumeMap map[string]*Volume

// NewVolumeManager returns a new VolumeManager
func NewVolumeManager(config string, share bool) (*VolumeManager, error) {
	f, err := os.Open(config)
	if err != nil {
		return nil, err
	}

	defer f.Close()

	volumeManager := &VolumeManager{
		cfgPath:    filepath.Dir(config),
		cudaSoname: make(map[string]string),
		mlSoName:   make(map[string]string),
		share:      share,
	}

	if err := json.NewDecoder(f).Decode(volumeManager); err != nil {
		return nil, err
	}

	return volumeManager, nil
}

// Run starts a VolumeManager

// volumeManager 的启动
func (vm *VolumeManager) Run() (err error) {

	// ldcache 是动态连接库的缓存信息
	cache, err := ldcache.Open()
	if err != nil {
		return err
	}

	defer func() {
		if e := cache.Close(); err == nil {
			err = e
		}
	}()

	vols := make(VolumeMap)
	for _, cfg := range vm.Config {
		vol := &Volume{
			Path: path.Join(cfg.BasePath, cfg.Name),
		}

		if cfg.Name == "nvidia" {
			// nvidia 库的位置
			types.DriverLibraryPath = filepath.Join(cfg.BasePath, cfg.Name)
		} else {
			// origin 库的位置
			types.DriverOriginLibraryPath = filepath.Join(cfg.BasePath, cfg.Name)
		}

		for t, c := range cfg.Components {
			switch t {
			case "binaries":
				// 二进制文件使用 which 命令来查找可执行文件的位置
				bins, err := which(c...)
				if err != nil {
					return err
				}

				klog.V(2).Infof("Find binaries: %+v", bins)

				// 将实际位置存起来
				vol.dirs = append(vol.dirs, volumeDir{binDir, bins})
			case "libraries":
				// 库 就从 ldcache 里面去找
				libs32, libs64 := cache.Lookup(c...)
				klog.V(2).Infof("Find 32bit libraries: %+v", libs32)
				klog.V(2).Infof("Find 64bit libraries: %+v", libs64)

				// 将 library 位置存起来
				vol.dirs = append(vol.dirs, volumeDir{lib32Dir, libs32}, volumeDir{lib64Dir, libs64})
			}

			vols[cfg.Name] = vol
		}
	}

	// 上半部分的代码，都是查找指定的动态连接库和可执行文件
	// 这些文件是在 volume.conf 这个配置文件中指定，再通过参数传进来
	// 查找动态链接库时，使用的是 ldcache
	// 查找可执行文件时，使用的是系统的 which 指令
	// 找到之后将其位置记录下来
	// 接着再对找到的库做 mirror 处理，即以下代码

	// 找到了需要的库位置之后，做 mirror 处理
	if err := vm.mirror(vols); err != nil {
		return err
	}

	klog.V(2).Infof("Volume manager is running")

	return nil
}

// #lizard forgives
func (vm *VolumeManager) mirror(vols VolumeMap) error {
	// nvidia 和 origin
	for driver, vol := range vols {
		if exist, _ := vol.exist(); !exist {
			// 这里的 path 是 /etc/gpu-manager/vdriver 内
			if err := os.MkdirAll(vol.Path, 0755); err != nil {
				return err
			}
		}

		for _, d := range vol.dirs {
			vpath := path.Join(vol.Path, d.name)
			// 创建 bin lib lib64
			if err := os.MkdirAll(vpath, 0755); err != nil {
				return err
			}

			// For each file matching the volume components (blacklist excluded), create a hardlink/copy
			// of it inside the volume directory. We also need to create soname symlinks similar to what
			// ldconfig does since our volume will only show up at runtime.
			for _, f := range d.files {
				klog.V(2).Infof("Mirror %s to %s", f, vpath)

				// 对所有上面查找到的库或可执行文件调用 mirrorFiles
				if err := vm.mirrorFiles(driver, vpath, f); err != nil {
					return err
				}

				// 记录下 libcuda.so 的版本号
				if strings.HasPrefix(path.Base(f), "libcuda.so") {
					driverStr := strings.SplitN(strings.TrimPrefix(path.Base(f), "libcuda.so."), ".", 2)
					types.DriverVersionMajor, _ = strconv.Atoi(driverStr[0])
					types.DriverVersionMinor, _ = strconv.Atoi(driverStr[1])
					klog.V(2).Infof("Driver version: %d.%d", types.DriverVersionMajor, types.DriverVersionMinor)
				}

				// 记录下 libcuda-control.so 的位置
				// 这个 libcuda-control 就是 vcuda-control 下面生成的用来拦截 cuda 调用的库
				if strings.HasPrefix(path.Base(f), "libcuda-control.so") {
					vm.cudaControlFile = f
				}
			}
		}
	}

	vCudaFileFn := func(soFile string) error {
		if err := os.Remove(soFile); err != nil {
			if !os.IsNotExist(err) {
				return err
			}
		}
		// clone 返回会先尝试硬链接过去，若失败就直接复制过去
		if err := clone(vm.cudaControlFile, soFile); err != nil {
			return err
		}

		klog.V(2).Infof("Vcuda %s to %s", vm.cudaControlFile, soFile)

		l := strings.TrimRight(soFile, ".0123456789")
		if err := os.Remove(l); err != nil {
			if !os.IsNotExist(err) {
				return err
			}
		}
		// clone 返回会先尝试硬链接过去，若失败就直接复制过去
		if err := clone(vm.cudaControlFile, l); err != nil {
			return err
		}
		klog.V(2).Infof("Vcuda %s to %s", vm.cudaControlFile, l)
		return nil
	}

	// 然后将 cudaControlFile clone 到所有 cudaSoname 和 mlSoName 中库的位置
	// cudaControlFile 就是上面所说的 libcuda-control.so
	if vm.share && len(vm.cudaControlFile) > 0 {
		if len(vm.cudaSoname) > 0 {
			for _, f := range vm.cudaSoname {
				if err := vCudaFileFn(f); err != nil {
					return err
				}
			}
		}

		if len(vm.mlSoName) > 0 {
			for _, f := range vm.mlSoName {
				if err := vCudaFileFn(f); err != nil {
					return err
				}
			}
		}
	}

	return nil
}

// #lizard forgives

// driver 是配置文件中的 "nvidia" 或  "origin"
// vpath 是要 mirror 到的位置，在 /etc/gpu-manager/vdriver 内

func (vm *VolumeManager) mirrorFiles(driver, vpath string, file string) error {
	// 在计算机中，可执行和可连接格式 (ELF，以前成为可扩展链接格式)是可执行文件，目标代码，共享库，核心转储的通用标准文件格式
	obj, err := elf.Open(file)
	if err != nil {
		return fmt.Errorf("%s: %v", file, err)
	}
	defer obj.Close()

	// 黑名单机制，具体用处不详，跟 nvidia 的驱动有关
	ok, err := blacklisted(file, obj)
	if err != nil {
		return fmt.Errorf("%s: %v", file, err)
	}

	if ok {
		return nil
	}

	l := path.Join(vpath, path.Base(file))
	// 不管是否存在，先尝试移除 gpu-manager 里面的文件
	if err := removeFile(l); err != nil {
		return err
	}

	// clone 优先硬链接，其次是复制文件到指定位置
	if err := clone(file, l); err != nil {
		return err
	}

	// 从 elf 中获取当前库的 soname
	soname, err := obj.DynString(elf.DT_SONAME)
	if err != nil {
		return fmt.Errorf("%s: %v", file, err)
	}

	if len(soname) > 0 {
		// 将获取到的 soname 组成完整路径
		l = path.Join(vpath, soname[0])

		// 如果文件和它的 soname 不一致，就创建软链接
		if err := linkIfNotSameName(path.Base(file), l); err != nil && !os.IsExist(err) {
			return err
		}

		// XXX Many applications (wrongly) assume that libcuda.so exists (e.g. with dlopen)
		// Hardcode the libcuda symlink for the time being.
		if strings.Contains(driver, "nvidia") {
			// Remove libcuda symbol link
			if vm.share && driver == "nvidia" && strings.HasPrefix(soname[0], "libcuda.so") {
				os.Remove(l)
				vm.cudaSoname[l] = l
			}

			// Remove libnvidia-ml symbol link
			if vm.share && driver == "nvidia" && strings.HasPrefix(soname[0], "libnvidia-ml.so") {
				os.Remove(l)
				vm.mlSoName[l] = l
			}

			// 以上为何要移除 libcuda.so 和 libnvidia-ml.so 的软链接
			// 因为 GPU 调用会涉及到这两个库，这两个库会链接到真是的库上
			// 移除后替换成拦截的库

			// XXX GLVND requires this symlink for indirect GLX support
			// It won't be needed once we have an indirect GLX vendor neutral library.
			if strings.HasPrefix(soname[0], "libGLX_nvidia") {
				l = strings.Replace(l, "GLX_nvidia", "GLX_indirect", 1)
				if err := linkIfNotSameName(path.Base(file), l); err != nil && !os.IsExist(err) {
					return err
				}
			}
		}
	}

	// 以上代码，先用 blacklisted 排除一些不需要处理的库
	// 然后尝试将库或可执行文件 clone 到 /etc/gpu-manager/vdriver 下
	// /etc/gpu-manager 下有两个文件夹，一个是 nvidia，保存了已被拦截的库，一个是 origin，放的是原始未经处理的库
	// 同时，还将 libcuda.so 和 libnvidia-ml.so 移除，这样就调用不到真实的库了，转而在之后使用拦截的库来替换这几个文件

	return nil
}

func (v *Volume) exist() (bool, error) {
	_, err := os.Stat(v.Path)
	if os.IsNotExist(err) {
		return false, nil
	}

	return true, err
}

func (v *Volume) remove() error {
	return os.RemoveAll(v.Path)
}

func removeFile(file string) error {
	if err := os.Remove(file); err != nil {
		if !os.IsNotExist(err) {
			return err
		}
	}

	return nil
}

func linkIfNotSameName(src, dst string) error {
	if path.Base(src) != path.Base(dst) {
		if err := removeFile(dst); err != nil {
			if !os.IsNotExist(err) {
				return err
			}
		}

		l := strings.TrimRight(dst, ".0123456789")
		if err := removeFile(l); err != nil {
			if !os.IsExist(err) {
				return err
			}
		}

		if err := os.Symlink(src, l); err != nil && !os.IsExist(err) {
			return err
		}

		if err := os.Symlink(src, dst); err != nil && !os.IsExist(err) {
			return err
		}
	}

	return nil
}
