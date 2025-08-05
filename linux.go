package main

type LinuxPlatform struct {
	BasePlatform
}

func (linux *LinuxPlatform) InstallTVM() {
	//TODO
}

func (linux *LinuxPlatform) GenerateConfig() {
	//TODO
}

func (linux *LinuxPlatform) BuildMLC() {
	//TODO
}

func (linux *LinuxPlatform) BuildTVM() {
	//TODO
}

func (linux *LinuxPlatform) BuildAndroid() {
	//TODO
}

func (linux *LinuxPlatform) GetName() string {
	return linux.Name
}

func (linux *LinuxPlatform) GetBuildEnv() string {
	return linux.BuildEnv
}

func (linux *LinuxPlatform) GetCliEnv() string {
	return linux.CliEnv
}
