<p align="center"><img src="ultrasound.png" width="70%" /><br><br></p>

-----------------

## Overview
**Ultrasound with AI** 은 [광운대학교 전기공학과 ultrasound 수업](./doc/t265.md) 에서 진행한 텀프로젝트입니다.
> :pushpin: calculator 프로젝트를 보고 싶다면, [calculator repository](https://github.com/heypaprika/calculator)를 참고하세요.

이 프로젝트는 기존에 사용되던 의료 초음파 기술에 AI를 어디에 결합시킬 수 있을까 하는 궁금증으로 출발하였습니다. 기존의 의료 초음파를 사용하기 위한 장비에는 모니터, 파라미터를 조절하는 레버들, 프로브 등이 있어서 크기가 클 수 밖에 없었습니다. 크기가 커서 가정에서는 쉽게 사용할 수 없었는데, 요즘은 스마트폰에 probe를 연결하여 자가진단할 수 있도록 하는 application이 여럿 개발되었습니다. 

이러한 여러 어플리케이션의 개발로 인하여 사용자는 많은 선택지를 가질 수 있게 되었지만 한 가지 문제점이 있었는데, 파라미터 값을 어떻게 설정해야 초음파 이미지의 화면이 뚜렷하게 나오는 지를 잘 알지 못한다는 점입니다.

이러한 부분에 착안하여 저는 probe를 피부에 가져다 대었을 때, 어느 부위를 가져다 댈때에도 뚜렷한 화면을 보여주는 시스템을 만들고 싶었습니다. 그리고 이 repository가 그 시스템의 일부로서 동작할 것입니다.

## Download and Install
* **Download** - git clone URL

* **Install** - (UBUNTU) : sh install.sh


## What’s included in the Project:
| What | Description | link|
| ------- | ------- | ------- |
| **[1](./readme.md)** | file Description | [**link1**](./readme.md) |
| **[2](./readme.md)** | file Description | [**link2**](./readme.md) |
| **[3](./readme.md)** | file Description | [**link3**](./readme.md) |
| **[4](./readme.md)** | file Description | [**link4**](./readme.md) |
| **[5](./readme.md)** | file Description | [**link5**](./readme.md) | |



## File example


```python3
// Create a Pipeline - this serves as a top-level API for streaming and processing frames
rs2::pipeline p;

// Configure and start the pipeline
p.start();

while (true)
{
    // Block program until frames arrive
    rs2::frameset frames = p.wait_for_frames();

    // Try to get a frame of a depth image
    rs2::depth_frame depth = frames.get_depth_frame();

    // Get the depth frame's dimensions
    float width = depth.get_width();
    float height = depth.get_height();

    // Query the distance from the camera to the object in the center of the image
    float dist_to_center = depth.get_distance(width / 2, height / 2);

    // Print the distance
    std::cout << "The camera is facing an object " << dist_to_center << " meters away \r";
}
```
