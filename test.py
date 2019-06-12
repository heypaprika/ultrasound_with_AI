import os
import glob
import torch
import torchvision.models as models
from linearRegression import linearRegression as Model
from dataset.ultrasound import Ultrasound_Dataset_test as Dataset
from torchvision import transforms

x_train = glob.glob('./test/*')

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

Test_Dataset = Dataset(x_train, os.path.abspath('./'), transforms=transform)
Test_Loader = torch.utils.data.DataLoader(Test_Dataset, batch_size=14) # + 나머지 option들

inputDim = 1000
outputDim = 5

reg_model = Model(inputDim, outputDim)
reg_model.load_state_dict(torch.load('bestres.pth'))
densenet = models.densenet121(pretrained=True)
cuda = 0

if torch.cuda.is_available():
    reg_model.cuda()
    densenet.cuda()
    cuda = 1


reg_model.eval()
with torch.no_grad():
    for ind, (x_train) in enumerate(Test_Loader):
        x_feature = densenet(x_train.cuda())
        predicted = reg_model(x_feature)
        predicted[:, 0] *= 10000000
        predicted[:, 1] *= 0.0015
        predicted[:, 2] *= 0.00075
        predicted[:, 3] *= 0.01
        predicted[:, 4] *= 0.000025
        print(predicted)
