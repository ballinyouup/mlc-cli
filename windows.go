package main

type WindowsPlatform struct {
	BasePlatform
}

func (windows *WindowsPlatform) InstallTVM() {
	//TODO
}

func (windows *WindowsPlatform) GenerateConfig() {
	//TODO
}

func (windows *WindowsPlatform) BuildMLC() {
	//TODO
}

func (windows *WindowsPlatform) BuildTVM() {
	//TODO
}

func (windows *WindowsPlatform) BuildAndroid() {
	//TODO
}

func (windows *WindowsPlatform) GetName() string {
	return windows.Name
}

func (windows *WindowsPlatform) GetBuildEnv() string {
	return windows.BuildEnv
}

func (windows *WindowsPlatform) GetCliEnv() string {
	return windows.CliEnv
}
