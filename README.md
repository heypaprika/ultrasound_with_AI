<p align="center"><img src="ultrasound.png" width="70%" /><br><br></p>

-----------------

## Overview
**Ultrasound with AI** 은 [광운대학교 전기공학과 ultrasound 수업](./doc/t265.md) 에서 진행한 텀프로젝트입니다.
> :pushpin: calculator 프로젝트를 보고 싶다면, [calculator repository](https://github.com/heypaprika/calculator)를 참고하세요.

이 프로젝트는 기존에 사용되던 의료 초음파 기술에 AI를 어디에 결합시킬 수 있을까 하는 궁금증으로 출발하였습니다. 기존의 의료 초음파를 사용하기 위한 장비에는 모니터, 파라미터를 조절하는 레버들, 프로브 등이 있어서 크기가 클 수 밖에 없었습니다. 크기가 커서 가정에서는 쉽게 사용할 수 없었는데, 요즘은 스마트폰에 probe를 연결하여 자가진단할 수 있도록 하는 application이 여럿 개발되었습니다. 

이러한 여러 어플리케이션의 개발로 인하여 사용자는 많은 선택지를 가질 수 있게 되었지만 한 가지 문제점이 있는데, 바로 파라미터 값을 어떻게 설정해야 초음파 이미지의 화면이 뚜렷하게 나오는 지를 잘 알지 못한다는 점입니다.

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

하단의 snippet은 해당 모델을 training하는 소스입니다.

```python3
import torch
import torchvision.models as models
from linearRegression import linearRegression as Model
from dataset.ultrasound import Ultrasound_Dataset_for_dump as Dataset_dump
from dataset.ultrasound import Ultrasound_Dataset as Dataset

x_train = torch.randn(100,3,224,224)
y_train = torch.randn(100,5)

#dataloader에 넣어주기.
Train_Dataset = Dataset_dump(x_train, y_train)
Train_Loader = torch.utils.data.DataLoader(Train_Dataset) # + 나머지 option들


inputDim = 1000
outputDim = 5
learningRate = 0.001
epochs = 100

reg_model = Model(inputDim, outputDim)
densenet = models.densenet121(pretrained=True)
cuda = 0
if torch.cuda.is_available():
    reg_model.cuda()
    densenet.cuda()
    cuda = 1

criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(reg_model.parameters(),lr=learningRate)

reg_model.train()
for epoch in range(epochs):
    for ind, (x_train, y_train) in enumerate(Train_Loader):

        optimizer.zero_grad()
        x_feature = densenet(x_train.cuda())
        outputs = reg_model(x_feature)
        loss = criterion(outputs, y_train.cuda())
        # print(loss)
        loss.backward()
        optimizer.step()
    print('epoch {}, loss {}'.format(epoch, loss.item()))

with torch.no_grad():
    reg_model.eval()
    x_feature = densenet(x_train).cuda()
    predicted = reg_model(x_feature)
    print(predicted)

```
