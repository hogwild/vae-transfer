import torch
from torch import nn
from torch.nn import functional as F

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    def __init__(self, zsize, layer_count=3, channels=3):
        super(Encoder, self).__init__()

        d = 128
        self.d = d
        self.zsize = zsize

        self.layer_count = layer_count

        mul = 1
        inputs = channels
        for i in range(self.layer_count):
            setattr(self, "conv%d" % (i + 1), nn.Conv2d(inputs, d * mul, 4, 2, 1))
            setattr(self, "conv%d_bn" % (i + 1), nn.BatchNorm2d(d * mul))
            inputs = d * mul
            mul *= 2

        self.d_max = inputs

        self.fc1 = nn.Linear(inputs * 4 * 4, zsize)
        self.fc2 = nn.Linear(inputs * 4 * 4, zsize)

    def forward(self, x):
        for i in range(self.layer_count):
            x = F.relu(getattr(self, "conv%d_bn" % (i + 1))(getattr(self, "conv%d" % (i + 1))(x)))
        return x
        # return x.view(x.shape[0], self.d_max * 4 * 4)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)


    def load_pretrained_dict(self, pretrained_dict):
        own_dict = self.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in own_dict}
        # 2. overwrite entries in the existing state dict
        own_dict.update(pretrained_dict) 
        # 3. load the new state dict
        self.load_state_dict(pretrained_dict)
        # own_state = self.state_dict()
        # for name, param in state_dict.items():
        #     if name not in own_state:
        #         continue
        #     if isinstance(param, Parameter):
        #         # backwards compatibility for serialized parameters
        #         param = param.data
        #     own_state[name].copy_(param)


class Decoder(nn.Module):
    def __init__(self, zsize, layer_count=3, channels=3):
        super(Decoder, self).__init__()

        d = 128
        self.d = d
        self.zsize = zsize

        self.layer_count = layer_count

        mul = 1
        inputs = channels
        for i in range(self.layer_count):
            inputs = d * mul
            mul *= 2

        self.d_max = inputs

        self.d1 = nn.Linear(zsize, inputs * 4 * 4)

        mul = inputs // d // 2

        for i in range(1, self.layer_count):
            setattr(self, "deconv%d" % (i + 1), nn.ConvTranspose2d(inputs, d * mul, 4, 2, 1))
            setattr(self, "deconv%d_bn" % (i + 1), nn.BatchNorm2d(d * mul))
            inputs = d * mul
            mul //= 2

        setattr(self, "deconv%d" % (self.layer_count + 1), nn.ConvTranspose2d(inputs, channels, 4, 2, 1))

    def forward(self, x):
        # x = x.view(x.shape[0], self.zsize)
        # x = self.d1(x)
        # x = x.view(x.shape[0], self.d_max, 4, 4)
        #x = self.deconv1_bn(x)
        x = F.leaky_relu(x, 0.2)
        for i in range(1, self.layer_count):
            x = F.leaky_relu(getattr(self, "deconv%d_bn" % (i + 1))(getattr(self, "deconv%d" % (i + 1))(x)), 0.2)

        # x = F.tanh(getattr(self, "deconv%d" % (self.layer_count + 1))(x))
        x = torch.tanh(getattr(self, "deconv%d" % (self.layer_count + 1))(x))
        return x

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    
    def load_pretrained_dict(self, pretrained_dict):
        own_dict = self.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in own_dict}
        # 2. overwrite entries in the existing state dict
        own_dict.update(pretrained_dict) 
        # 3. load the new state dict
        self.load_state_dict(pretrained_dict)
        

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


class Encoder6(nn.Module):
    def __init__(self, zsize, layer_count=3, channels=3):
        super(Encoder6, self).__init__()

        d = 128
        self.d = d
        self.zsize = zsize

        self.layer_count = layer_count

        mul = 1
        inputs = channels
        for i in range(self.layer_count):
            setattr(self, "conv%d" % (i + 1), nn.Conv2d(inputs, d * mul, 4, 2, 1))
            setattr(self, "conv%d_bn" % (i + 1), nn.BatchNorm2d(d * mul))
            inputs = d * mul
            mul *= 2

        self.d_max = inputs

        self.fc1 = nn.Linear(inputs * 2 * 2, zsize)
        self.fc2 = nn.Linear(inputs * 2 * 2, zsize)

    def forward(self, x):
        output = {}
        # out = getattr(self, "conv1")(x)
        for i in range(0, self.layer_count):
            # out = getattr(self, "conv%d" % (i + 1))(out)
            # output["conv%d_bn" % (i + 1)] = F.relu(getattr(self, "conv%d_bn" % (i + 1)))(out)
            # out = output["conv%d_bn" % (i + 1)]

            x = F.relu(getattr(self, "conv%d_bn" % (i + 1))(getattr(self, "conv%d" % (i + 1))(x)))
            output["conv%d_bn" % (i + 1)] = x
        return output
        # return x.view(x.shape[0], self.d_max * 2 * 2)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)


    def load_pretrained_dict(self, pretrained_dict):
        own_dict = self.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in own_dict}
        # 2. overwrite entries in the existing state dict
        own_dict.update(pretrained_dict) 
        # 3. load the new state dict
        self.load_state_dict(pretrained_dict)
        


##loading the pre-trained parameters
if __name__ == "__main__":
    model_path = 'VAEmodel_5layers.pkl'
    pretrained = torch.load(model_path, map_location=DEVICE)
    evaluator_path = 'VAEmodel_6layers.pkl'
    pretrained_evaluator = torch.load(evaluator_path, map_location=DEVICE)
    # print(pretrained.items())
    encoder = Encoder(zsize=512, layer_count=5)
    encoder.load_pretrained_dict(pretrained)
    decoder = Encoder(zsize=512, layer_count=5)
    decoder.load_pretrained_dict(pretrained)
    evaluator = Encoder6(zsize=512, layer_count=6)
    evaluator.load_pretrained_dict(pretrained_evaluator)

