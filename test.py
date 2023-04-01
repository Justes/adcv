import torch

#path = '../vit_base_veri.pth'
#path = '../pretrained/vit_base_patch16_224_in21k.pth'
#path = '../2023-04-01_model.pth.tar-60'
path = '../pretrained/jx_vit_base_p16_224-80ecf9dd.pth'
state = torch.load(path, map_location=torch.device('cpu'))
#state_dict = state["state_dict"]
for k, v in state.items():
    print(k, v.shape)