
import torch
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
def load_determined_state_dict(ckpt):
    '''
    Removes module from state dict keys as determined saves model in DataParallel format:
    https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/4
    '''
    new_state_dict = OrderedDict()
    for k, v in ckpt['models_state_dict'][0].items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    return new_state_dict

def visualize_pred(inv_tensor,res,targets_t):
    '''
    '''
    img = Image.fromarray((255.*inv_tensor.cpu().permute((1,2,0)).numpy()).astype(np.uint8))
    draw = ImageDraw.Draw(img)
    # draw ground truth
    print("Num GT Boxes: ",targets_t['boxes'].shape[0])
    for ind,(b,l) in enumerate(zip(targets_t['boxes'],targets_t['labels'])):
        # print(b.detach().numpy(), s.detach().numpy())
        x,y,x2,y2 = b.detach().numpy()
        # print( x,y,x2,y2,l.item())
        draw.rectangle([x,y,x2,y2],fill=None,outline=(0,255,0))
        draw.text([x,y-10],"{}".format(l),fill=None,outline=(0,255,0))

    idx = list(res.keys())[0]
    print("Num Pred Boxes: ",res[idx]['boxes'].shape[0])
    for ind,(b,s,l) in enumerate(zip(res[idx]['boxes'],res[idx]['scores'],res[idx]['labels'])):
        # print(b.detach().numpy(), s.detach().numpy())
        x,y,x2,y2 = b.detach().numpy()
        # print( x,y,x2,y2,s.item(),l.item())
        draw.rectangle([x,y,x2,y2],fill=None,outline=(255,0,0))
        draw.text([x,y-10],"{}".format(l),fill=None,outline=(255,0,0))
    
    return img

def visualize_gt(inv_tensor,targets_t):
    '''
    '''
    img = Image.fromarray((255.*inv_tensor.cpu().permute((1,2,0)).numpy()).astype(np.uint8))
    draw = ImageDraw.Draw(img)
    # draw ground truth
    print("Num GT Boxes: ",targets_t['boxes'].shape[0])
    for ind,(b,l) in enumerate(zip(targets_t['boxes'],targets_t['labels'])):
        # print(b.detach().numpy(), s.detach().numpy())
        x,y,x2,y2 = b.detach().numpy()
        # print( x,y,x2,y2,l.item())
        draw.rectangle([x,y,x2,y2],fill=None,outline=(0,255,0))
        draw.text([x,y-10],"{}".format(l),fill=None,outline=(0,255,0))
    return img

def predict(model,images_t,targets_t):
    '''
    '''
    cpu_device = torch.device('cpu')
    device = torch.device('cuda') if next(model.parameters()).is_cuda else torch.device('cpu')
    images_t = list(image.to(device) for image in images_t)
    outputs = model(images_t)
    # print(x,outputs)
    outputss = []
    for t in outputs:
        outputss.append({k: v.to(cpu_device) for k, v in t.items()})
    # model_time = time.time() - model_time
    # print(targets_t)
    # print(outputss)
    # print("targets_t[image_id]: ",targets_t["image_id"].item())
    res = {int(target["image_id"].item()): output for target, output in zip(targets_t, outputss)}
    for i,t in zip(images_t,targets_t):
        im = visualize_pred(i,res,t)
        plt.imshow(im)
        plt.show()
    return res 