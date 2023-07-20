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

        self._init_lora()

    def forward(self, x):
        residual = x

        out = self.relu(self.bn1(
            self.conv1(x)
            + self.dropout(self.lora_up_1(self.selector(self.lora_down_1(x))))
            * self.scale
        )) #(bs, 64, 128, 128)

        out = self.relu(self.bn2(
            self.conv2(out)
            + self.dropout(self.lora_up_2(self.selector(self.lora_down_2(out))))
            * self.scale
        )) #(bs, 64, 128, 128)

        out = self.bn3(
            self.conv3(out)
            + self.dropout(self.lora_up_3(self.selector(self.lora_down_3(out))))
            * self.scale
        ) #(bs, 256, 128, 128)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out #(bs, 256, 128, 128)

    def _init_lora(self):
        self.r = 4 #4/8
        self.lora_down_1 = nn.Conv2d(in_channels=self.inplanes, out_channels=self.r, kernel_size=1, 
                                     stride=1, padding=0,dilation=1,groups=1,bias=False,)
        self.lora_up_1 = nn.Conv2d(in_channels=self.r,out_channels=self.planes,kernel_size=1,stride=1,padding=0,bias=False,)

        self.lora_down_2 = nn.Conv2d(in_channels=self.planes, out_channels=self.r, kernel_size=3, 
                                     stride=self.stride, padding=1,dilation=1,groups=1,bias=False,)
        self.lora_up_2 = nn.Conv2d(in_channels=self.r,out_channels=self.planes,kernel_size=1,stride=1,padding=0,bias=False,)


        self.lora_down_3 = nn.Conv2d(in_channels=self.planes, out_channels=self.r, kernel_size=1, 
                                     stride=1, padding=0,dilation=1,groups=1,bias=False,)
        self.lora_up_3 = nn.Conv2d(in_channels=self.r,out_channels=self.planes * self.expansion,kernel_size=1,stride=1,padding=0,bias=False,)

        self.dropout = nn.Dropout(0.1)
        self.selector = nn.Identity()
        self.scale = 1.0

        nn.init.normal_(self.lora_down_1.weight, std=1 / self.r)
        nn.init.zeros_(self.lora_up_1.weight)
        nn.init.normal_(self.lora_down_2.weight, std=1 / self.r)
        nn.init.zeros_(self.lora_up_2.weight)
        nn.init.normal_(self.lora_down_3.weight, std=1 / self.r)
        nn.init.zeros_(self.lora_up_3.weight)


# @BACKBONES.register_module()
# class ResNet(nn.Module):
#     def __init__(self, block, layers, num_classes=1000, deep_base=True):
#         super(ResNet, self).__init__()
#         self.deep_base = deep_base
#         if not self.deep_base:
#             self.inplanes = 64
#             self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
#             self.bn1 = BatchNorm(64)
#             self.relu = nn.ReLU(inplace=True)
#         else:
#             self.inplanes = 128
#             self.conv1 = conv3x3(3, 64, stride=2)
#             self.bn1 = BatchNorm(64)
#             self.relu1 = nn.ReLU(inplace=True)
#             self.conv2 = conv3x3(64, 64)
#             self.bn2 = BatchNorm(64)
#             self.relu2 = nn.ReLU(inplace=True)
#             self.conv3 = conv3x3(64, 128)
#             self.bn3 = BatchNorm(128)
#             self.relu3 = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self._make_layer(block, 64, layers[0])
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
#         self.avgpool = nn.AvgPool2d(7, stride=1)
#         self.fc = nn.Linear(512 * block.expansion, num_classes)

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, BatchNorm):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

#     def _make_layer(self, block, planes, blocks, stride=1):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv2d(self.inplanes, planes * block.expansion,
#                           kernel_size=1, stride=stride, bias=False),
#                 BatchNorm(planes * block.expansion),
#             )

#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample))
#         self.inplanes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(block(self.inplanes, planes))

#         return nn.Sequential(*layers)

#     def forward(self, x):
#         x = self.relu1(self.bn1(self.conv1(x)))
#         if self.deep_base:
#             x = self.relu2(self.bn2(self.conv2(x)))
#             x = self.relu3(self.bn3(self.conv3(x)))
#         x = self.maxpool(x)

#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)

#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)

#         return x


@BACKBONES.register_module()
class LoRAResNet(nn.Module):
    # def __init__(self, block, layers, num_classes=1000, deep_base=True):
    def __init__(self, layers, block=LoRABottleneck, num_classes=1000, deep_base=False, pretrained=None, **kwargs):
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
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(16, stride=1)
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
        
    # def _init_lora_fc(self):
    #     r=4
    #     self.r = r
    #     self.lora_fc_down = nn.Linear(self.in_fc_dim, r, bias=False)
    #     self.dropout_fc = nn.Dropout(0.1)
    #     self.lora_fc_up = nn.Linear(r, self.out_fc_dim, bias=False)
    #     self.scale = 1
    #     self.selector = nn.Identity()

    #     nn.init.normal_(self.lora_fc_down.weight, std=1 / r)
    #     nn.init.zeros_(self.lora_fc_up.weight)

    # def _make_layer(self, block, planes, blocks, stride=1):
    #     downsample = None
    #     if stride != 1 or self.inplanes != planes * block.expansion:
    #         downsample = nn.Sequential(
    #             nn.Conv2d(self.inplanes, planes * block.expansion,
    #                       kernel_size=1, stride=stride, bias=False),
    #             BatchNorm(planes * block.expansion),
    #         )

    #     layers = []
    #     layers.append(block(self.inplanes, planes, stride, downsample))
    #     self.inplanes = planes * block.expansion
    #     for i in range(1, blocks):
    #         layers.append(block(self.inplanes, planes))

    #     return nn.Sequential(*layers)
    
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
        x = self.layer1(x) #(bs, 256, 128, 128) 
        x = self.layer2(x) #(bs, 512, 64, 64)
        x = self.layer3(x) #(bs, 1024, 32, 32)
        x = self.layer4(x) #(bs, 2024, 16, 16)

        x_global = self.avgpool(x).squeeze()
        x_local = x
        
        visual_embedding = x_local / x_local.norm(dim=1, keepdim=True) ##ADDED_Norm
        features.append(visual_embedding)
        # x = x.view(x.size(0), -1) # (bs, )
        # x = self.fc(x) + self.dropout_fc(self.lora_fc_up(self.selector(self.lora_fc_down(x)))) * self.scale
        ## get embedding:
        global_embedding = x_global / x_global.norm(dim=1, keepdim=True) ##ADDED_Norm
        proto_embedding = x_local ##ADDED_Norm, fake proto
        
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


# def resnet50(pretrained=True, **kwargs):
#     """Constructs a ResNet-50 model.

#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
#     if pretrained:
#         # model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
#         model_path = './initmodel/resnet50_v2.pth'
#         model.load_state_dict(torch.load(model_path), strict=False)
#     return model


# def resnet101(pretrained=False, **kwargs):
#     """Constructs a ResNet-101 model.

#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
#     if pretrained:
#         # model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
#         model_path = './initmodel/resnet101_v2.pth'
#         model.load_state_dict(torch.load(model_path), strict=False)
#     return model
