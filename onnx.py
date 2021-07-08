from __future__ import division

import os
import sys
import logging
import torch
import numpy as np

from thop import profile


from models.model_stages_trt import BiSeNet

def main():
    
    print("begin")
    # preparation ################
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    seed = 12345
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Configuration ##############
    use_boundary_2 = False
    use_boundary_4 = False
    use_boundary_8 = True
    use_boundary_16 = False
    use_conv_last = False
    n_classes = 19
    
    # STDC1Seg-50 250.4FPS on NVIDIA GTX 1080Ti
    backbone = 'STDCNet813'
    methodName = 'STDC1-Seg'
    inputSize = 512
    inputScale = 50
    inputDimension = (1, 3, 480, 640)

    # # STDC1Seg-75 126.7FPS on NVIDIA GTX 1080Ti
    # backbone = 'STDCNet813'
    # methodName = 'STDC1-Seg'
    # inputSize = 768
    # inputScale = 75
    # inputDimension = (1, 3, 768, 1536)

    # # STDC2Seg-50 188.6FPS on NVIDIA GTX 1080Ti
    # backbone = 'STDCNet1446'
    # methodName = 'STDC2-Seg'
    # inputSize = 512
    # inputScale = 50
    # inputDimension = (1, 3, 512, 1024)

    # # STDC2Seg-75 97.0FPS on NVIDIA GTX 1080Ti
    # backbone = 'STDCNet1446'
    # methodName = 'STDC2-Seg'
    # inputSize = 768
    # inputScale = 75
    # inputDimension = (1, 3, 768, 1536)
    
    model = BiSeNet(backbone=backbone, n_classes=n_classes, 
    use_boundary_2=use_boundary_2, use_boundary_4=use_boundary_4, 
    use_boundary_8=use_boundary_8, use_boundary_16=use_boundary_16, 
    input_size=inputSize, use_conv_last=use_conv_last)
    

    print('loading parameters...')
    respth = './checkpoints/{}/'.format(methodName)
    save_pth = os.path.join(respth, 'model_maxmIOU{}.pth'.format(inputScale))
    model.load_state_dict(torch.load(save_pth))
    model = model.cuda()
    #####################################################

    # export onnx model
    model.eval()
    _, c, h, w = inputDimension
    dummy_input = torch.randn(1, c, h, w, device='cuda')
    torch.onnx.export(model, dummy_input, "model.onnx", verbose=True, input_names=["input"], output_names=["output"], export_params=True,)

    # calculate FLOPS and params
    
    model = model.cpu()
    flops, params = profile(model, inputs=(torch.randn(inputDimension),), verbose=False)
    print("params = {}MB, FLOPs = {}GB".format(params / 1e6, flops / 1e9))
    logging.info("params = {}MB, FLOPs = {}GB".format(params / 1e6, flops / 1e9))
    


if __name__ == '__main__':
    main() 
