import torch
import torchvision.models as models
densenet = models.densenet121(pretrained=True)

x_input = torch.randn(1,1,420,580)
print(densenet(x_input).shape)
