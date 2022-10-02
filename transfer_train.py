from torch import optim
from torchvision.utils import save_image
import numpy as np
import pickle
import time
import random
import os
from net_transfer import *
from libs.Criterion import LossCriterion
from libs.Loader import Dataset
from Matrix import MulLayer

from net import VAE

# from batch_provider import batch_provider

torch.set_default_dtype(torch.float32)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_path = 'VAEmodel_5layers.pkl'
pretrained_model = torch.load(model_path, map_location=DEVICE)
evaluator_path = 'VAEmodel_6layers.pkl'
pretrained_evaluator = torch.load(evaluator_path, map_location=DEVICE)

# img_path = "../../face_images/anime-faces/" ##path of anime-faces
# img_list = pd.read_csv(img_path+'annotations.csv')
content_path = "../dataset/tsushima_yoshiko/"
style_path = "../../face_images/img_align_celeba/"

im_size = 128
zsize = 512
style_layers = ["conv%d_bn" % (i) for i in range(2, 7)]
content_layers = ["conv%d_bn" % (i) for i in range(2, 7)]
style_weight = 0.02
content_weight = 1.0

lr = 1e-4
n_iters = 4
log_interval = 2
loadSize = 128
fineSize = 128
batchSize = 32

encoder = Encoder(zsize, layer_count=5)
encoder.load_pretrained_dict(pretrained_model)
decoder = Decoder(zsize, layer_count=5)
decoder.load_pretrained_dict(pretrained_model)

vae = VAE(zsize, layer_count=5) #vae for generating style image
vae.load_state_dict(torch.load(model_path, map_location=DEVICE))

matrix = MulLayer(1024) ##need modification

lossNetwork = Encoder6(zsize, layer_count=6)
lossNetwork.load_pretrained_dict(pretrained_evaluator)

criterion = LossCriterion(style_layers, content_layers, style_weight, content_weight)
optimizer = optim.Adam(matrix.parameters(), lr)

content_dataset = Dataset(content_path, loadSize, fineSize)
content_loader_ = torch.utils.data.DataLoader(dataset     = content_dataset,
                                              batch_size  = batchSize,
                                              shuffle     = True,
                                              num_workers = 0,
                                              drop_last   = True)
content_loader = iter(content_loader_)

style_dataset = Dataset(style_path, loadSize, fineSize)
style_loader_ = torch.utils.data.DataLoader(dataset     = style_dataset,
                                              batch_size  = batchSize,
                                              shuffle     = True,
                                              num_workers = 0,
                                              drop_last   = True)
style_loader = iter(style_loader_)

animeV = torch.Tensor(batchSize, 3, fineSize, fineSize)
styleV = torch.Tensor(batchSize, 3, fineSize, fineSize)


def adjust_learning_rate(optimizer, iteration):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr / (1+iteration*1e-5)


for iteration in range(1, n_iters+1):
    optimizer.zero_grad()
    try: 
        content, _ = content_loader.next()
    except IOError:
        content, _ = content_loader.next()
    except StopIteration:
        content_loader = iter(content_loader_)
        content, _ = content_loader.next()
    except:
        continue

    try:
        style,_ = style_loader.next()
    except IOError:
        style,_ = style_loader.next()
    except StopIteration:
        style_loader = iter(style_loader_)
        style,_ = style_loader.next()
    except:
        continue

    animeV.resize_(content.size()).copy_(content)
    styleV.resize_(style.size()).copy_(style)
    # styleV, _, _ = vae(animeV)
    sF = encoder(styleV)
    cF = encoder(animeV)

    feature, transmatrix = matrix(cF, sF)
    transfer = decoder(feature)

    sF_loss = lossNetwork(styleV)
    cF_loss = lossNetwork(animeV)
    tF = lossNetwork(transfer)
    loss,styleLoss,contentLoss = criterion(tF,sF_loss,cF_loss)

    # backward & optimization
    loss.backward()
    optimizer.step()
    print('Iteration: [%d/%d] Loss: %.4f contentLoss: %.4f styleLoss: %.4f Learng Rate is %.6f'%
         (n_iters,iteration,loss,contentLoss,styleLoss,optimizer.param_groups[0]['lr']))

    adjust_learning_rate(optimizer, iteration)

    if((iteration) % log_interval == 0):
        transfer = transfer.clamp(0, 1)
        # style = styleV
        concat = torch.cat((content,style,transfer),dim=0)
        save_image(concat,'%s/%d.png'%('./results_gen/', iteration),normalize=True,scale_each=True,nrow=batchSize)

    # if(iteration > 0 and (iteration) % opt.save_interval == 0):
    #     torch.save(matrix.state_dict(), '%s/%s.pth' % ('./results_gen/',opt.layer))
