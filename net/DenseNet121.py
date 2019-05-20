import torch
import torchvision.models as models
densenet = models.densenet121(pretrained=True)

x_input = torch.randn(2,1,224,224)
print(densenet(x_input).shape)
