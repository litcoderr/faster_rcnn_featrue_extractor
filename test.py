#%%
from rcnn_featrue_extractor import FasterRcnnFeatureExtractor

import torch

input_tensor = torch.randn((2, 3, 224, 224)).cuda()

custom_extractor = FasterRcnnFeatureExtractor.build_pretrained()
# List(torch.Tensor(100, 4), len=batch_size) , List(torch.Tensor(100, 1024), len=batch_size)
boxes, features = custom_extractor(input_tensor)
