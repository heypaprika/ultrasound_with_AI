import torch
import torchvision.models as models
from linearRegression import linearRegression as Model

# x_train으로 넣어 줄 것 : image
# train에서, DenseNet과 같은 classification reg_model의 feature를 거친다. 1000 정도,,로 출력을 뽑아와서,, 배치로 처리하기.

x_train = torch.randn(10,3,224,224)
y_train = torch.randn(10,5)


inputDim = 1000
outputDim = 5
learningRate = 0.01
epochs = 100

reg_model = Model(inputDim, outputDim)
densenet = models.densenet121(pretrained=True)

cuda = 0
if torch.cuda.is_available():
    reg_model.cuda()
    cuda = 1

criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(reg_model.parameters(),lr=learningRate)

reg_model.train()
for epoch in range(epochs):
    optimizer.zero_grad()
    x_feature = densenet(x_train)
    outputs = reg_model(x_feature)
    loss = criterion(outputs, y_train)
    # print(loss)
    loss.backward()
    optimizer.step()
    print('epoch {}, loss {}'.format(epoch, loss.item()))

with torch.no_grad():
    reg_model.eval()
    predicted = reg_model(x_train)
    print(predicted)
