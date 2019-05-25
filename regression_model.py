import torch
import torchvision.models as models
from linearRegression import linearRegression as Model
from dataset.ultrasound import Ultrasound_Dataset_for_dump as Dataset_dump
from dataset.ultrasound import Ultrasound_Dataset as Dataset

# x_train으로 넣어 줄 것 : image
# train에서, DenseNet과 같은 classification reg_model의 feature를 거친다. 1000 정도,,로 출력을 뽑아와서,, 배치로 처리하기.

#dataset가져와서
# if torch.cuda.is_available():
#     x_train = torch.randn(100,3,224,224).cuda()
#     y_train = torch.randn(100,5).cuda()
# else:
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
