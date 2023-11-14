import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo

manual_seed=321
torch.manual_seed(manual_seed)
torch.cuda.manual_seed(manual_seed)
torch.cuda.manual_seed_all(manual_seed)

from mmseg.models.builder import BACKBONES
from timm.models.layers import trunc_normal_


BatchNorm = nn.BatchNorm2d
# BatchNorm = nn.SyncBatchNorm

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class LoRALayer():
    def __init__(
        self, 
        r: int, 
        lora_alpha: int, 
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights


class ConvLoRA(nn.Module, LoRALayer):
    def __init__(self, conv_module, in_channels, out_channels, kernel_size, r=8, lora_alpha=1, lora_dropout=0., merge_weights=True, **kwargs):
        super(ConvLoRA, self).__init__()
        self.conv = conv_module(in_channels, out_channels, kernel_size, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        assert isinstance(kernel_size, int)
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(
                self.conv.weight.new_zeros((r * kernel_size, in_channels * kernel_size))
            )
            self.lora_B = nn.Parameter(
              self.conv.weight.new_zeros((out_channels//self.conv.groups*kernel_size, r*kernel_size))
            )
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.conv.weight.requires_grad = False
        self.reset_parameters()
        self.merged = False

    def reset_parameters(self):
        self.conv.reset_parameters()
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode=True):
        super(ConvLoRA, self).train(mode)
        if mode:
            if self.merge_weights and self.merged:
                if self.r > 0:
                    # Make sure that the weights are not merged
                    self.conv.weight.data -= (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                if self.r > 0:
                    # Merge the weights and mark it
                    self.conv.weight.data += (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling
                self.merged = True

    def forward(self, x):
        if self.r > 0 and not self.merged: # self.merged=False
            # print('lora_weight', ((self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling).sum())
            return self.conv._conv_forward(
                x, 
                self.conv.weight + (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling,
                self.conv.bias
            )
        return self.conv(x)

class Conv2d(ConvLoRA):
    def __init__(self, *args, **kwargs):
        super(Conv2d, self).__init__(nn.Conv2d, *args, **kwargs)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BatchNorm(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    
class LoRABottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(LoRABottleneck, self).__init__()

        self.inplanes = inplanes
        self.planes = planes

        self.conv1 = Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm(planes)
        self.conv2 = Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BatchNorm(planes)
        self.conv3 = Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        ## Initialize LoRA
        self.r = 8 #4/8
        self.lora_alpha = 1

        self.dropout = nn.Dropout(0.1)
        self.selector = nn.Identity()
        
    def forward(self, x):
        residual = x
        
        # merge the params
        # print('++++++++++++ merge +++++++++++')
        # print('self.conv1.weight', self.conv1.conv.weight.data.sum())
        # print('self.conv2.weight', self.conv2.conv.weight.data.sum())
        # print('self.conv3.weight', self.conv3.conv.weight.data.sum())
        # print('self.lora_up_1', self.conv1.lora_A.sum())
        # print('self.lora_down_1', self.conv1.lora_B.sum())
        # print('self.lora_up_2', self.conv2.lora_A.sum())
        # print('self.lora_down_2', self.conv2.lora_B.sum())
        # print('self.lora_up_3', self.conv3.lora_A.sum())
        # print('self.lora_down_3', self.conv3.lora_B.sum())
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out #(bs, 256, 128, 128)

@BACKBONES.register_module()
class LoRAResNet(nn.Module):
    # def __init__(self, block, layers, num_classes=1000, deep_base=True):
    def __init__(self, layers, block1=Bottleneck, block2=LoRABottleneck, num_classes=1000, deep_base=False, pretrained=None, **kwargs):
        super(LoRAResNet, self).__init__()
        self.deep_base = deep_base
        if not self.deep_base:
            self.inplanes = 64
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu1 = nn.ReLU(inplace=True)
        else:
            self.inplanes = 128
            self.conv1 = conv3x3(3, 64, stride=2)
            self.bn1 = BatchNorm(64)
            self.relu1 = nn.ReLU(inplace=True)
            self.conv2 = conv3x3(64, 64)
            self.bn2 = BatchNorm(64)
            self.relu2 = nn.ReLU(inplace=True)
            self.conv3 = conv3x3(64, 128)
            self.bn3 = BatchNorm(128)
            self.relu3 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block2, 64, layers[0])
        self.layer2 = self._make_layer(block2, 128, layers[1], stride=2) 
        self.layer3 = self._make_layer(block2, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block2, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(16, stride=1)
        # self.mmpool = nn.MaxPool2d(16, stride=1)
        self.fc = nn.Linear(512 * block2.expansion, num_classes)
        self.in_fc_dim = 512 * block2.expansion
        self.out_fc_dim = num_classes
        self.pretrained = pretrained
        self.apply(self._init_weights)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, BatchNorm):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)
                
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    def _make_layer(self, block, planes, blocks, stride=1):
        # layers = [block(self.inplanes, planes, stride)]

        # self.inplanes = planes * block.expansion
        # for _ in range(1, blocks):
        #     layers.append(block(self.inplanes, planes))
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        if self.deep_base:
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x) #(bs, 64, 128, 128)

        outs = []
        features = []
        # print('=================================layer1=================================')
        x = self.layer1(x) #(bs, 256, 128, 128) 
        # print('=================================layer2=================================')
        x = self.layer2(x) #(bs, 512, 64, 64)
        # print('=================================layer3=================================')
        x = self.layer3(x) #(bs, 1024, 32, 32)
        # print('=================================layer4=================================')
        x = self.layer4(x) #(bs, 2024, 16, 16)

        visual_embedding = x
        
        visual_embedding = visual_embedding / visual_embedding.norm(dim=1, keepdim=True) ##ADDED_Norm
    
        features.append(visual_embedding)
        
        ## get embedding:
        # global_embedding = global_embedding / global_embedding.norm(dim=1, keepdim=True) ##ADDED_Norm
        
        global_embedding = self.avgpool(visual_embedding).squeeze()
        # global_embedding = self.mmpool(visual_embedding).squeeze()
        if global_embedding.shape[0] == 2048:
            global_embedding = global_embedding.unsqueeze(0)
        proto_embedding = visual_embedding ##ADDED_Norm, fake proto
        
        outs.append(tuple(features))
        outs.append(global_embedding) 
        outs.append(proto_embedding) 
        return outs

    def init_weights(self, pretrained=None):
        print('==========> Loading parameters from pretrained model ResNet <===========')
        pretrained = pretrained or self.pretrained
        if isinstance(pretrained, str):
            state_dict = torch.load(pretrained, map_location='cpu')
            u, w = self.load_state_dict(state_dict, strict=False)
            print(u, w, 'are misaligned params in resnet') # it should be nothing is misaligned
        print('check:', self.conv1.weight.sum())

@BACKBONES.register_module()
class MyResNet(nn.Module):
    # def __init__(self, block, layers, num_classes=1000, deep_base=True):
    def __init__(self, layers, block=Bottleneck, num_classes=1000, deep_base=False, pretrained=None, **kwargs):
        super(MyResNet, self).__init__()
        self.deep_base = deep_base
        if not self.deep_base:
            self.inplanes = 64
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu1 = nn.ReLU(inplace=True)
        else:
            self.inplanes = 128
            self.conv1 = conv3x3(3, 64, stride=2)
            self.bn1 = BatchNorm(64)
            self.relu1 = nn.ReLU(inplace=True)
            self.conv2 = conv3x3(64, 64)
            self.bn2 = BatchNorm(64)
            self.relu2 = nn.ReLU(inplace=True)
            self.conv3 = conv3x3(64, 128)
            self.bn3 = BatchNorm(128)
            self.relu3 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2) 
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(16, stride=1)
        # self.mmpool = nn.MaxPool2d(16, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.in_fc_dim = 512 * block.expansion
        self.out_fc_dim = num_classes
        self.pretrained = pretrained
        self.apply(self._init_weights)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, BatchNorm):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)
                
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    def _make_layer(self, block, planes, blocks, stride=1):
        # layers = [block(self.inplanes, planes, stride)]

        # self.inplanes = planes * block.expansion
        # for _ in range(1, blocks):
        #     layers.append(block(self.inplanes, planes))
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        if self.deep_base:
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x) #(bs, 64, 128, 128)

        outs = []
        features = []
        # print('=================================layer1=================================')
        x = self.layer1(x) #(bs, 256, 128, 128) 
        # print('=================================layer2=================================')
        x = self.layer2(x) #(bs, 512, 64, 64)
        # print('=================================layer3=================================')
        x = self.layer3(x) #(bs, 1024, 32, 32)
        # print('=================================layer4=================================')
        x = self.layer4(x) #(bs, 2024, 16, 16)

        visual_embedding = x
        global_embedding = self.avgpool(visual_embedding).squeeze()
        if global_embedding.shape[0] == 2048:
            global_embedding = global_embedding.unsqueeze(0)
        
        visual_embedding = visual_embedding / visual_embedding.norm(dim=1, keepdim=True) ##ADDED_Norm
        global_embedding = global_embedding / global_embedding.norm(dim=1, keepdim=True) ##ADDED_Norm
    
        
        # global_embedding = self.mmpool(visual_embedding).squeeze()
        if global_embedding.shape[0] == 2048:
            global_embedding = global_embedding.unsqueeze(0)
            
        features.append(visual_embedding)
        # proto_embedding = visual_embedding ##ADDED_Norm, fake proto
        
        outs.append(tuple(features))
        outs.append(global_embedding) 
        # outs.append(proto_embedding) 
        return outs

    def init_weights(self, pretrained=None):
        print('==========> Loading parameters from pretrained model ResNet <===========')
        pretrained = pretrained or self.pretrained
        if isinstance(pretrained, str):
            state_dict = torch.load(pretrained, map_location='cpu')
            u, w = self.load_state_dict(state_dict, strict=False)
            print(u, w, 'are misaligned params in resnet') # it should be nothing is misaligned
        print('check:', self.conv1.weight.sum())
