# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

import torch
from torch import Tensor
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet50(nn.Module):

    def __init__(
            self,
            block: Type[Union[BasicBlock, Bottleneck]] = Bottleneck,
            layers: List[int] = [3, 4, 6, 3],
            num_classes: int = 1000,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet50, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.num_features = 512 * block.expansion
        self.classifier = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)


    def forward(self, x, only_fc=False, only_feat=False, **kwargs):
        """
        Args:
            x: input tensor, depends on only_fc and only_feat flag
            only_fc: only use classifier, input should be features before classifier
            only_feat: only return pooled features
        """

        if only_fc:
            return self.fc(x)

        x = self.extract(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        if only_feat:
            return x

        out = self.classifier(x)
        result_dict = {'logits':out, 'feat':x}
        return result_dict
    
    def extract(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
    
    def group_matcher(self, coarse=False, prefix=''):
        # This function is for algorithms/comatch, crmatch, remixmatch, simamatch,
        matcher = dict(stem=r'^{}conv1|^{}bn1|^{}maxpool'.format(prefix, prefix, prefix), blocks=r'^{}layer(\d+)'.format(prefix) if coarse else r'^{}layer(\d+)\.(\d+)'.format(prefix))
        return matcher

    def no_weight_decay(self):
        nwd = []
        for n, _ in self.named_parameters():
            if 'bn' in n or 'bias' in n:
                nwd.append(n)
        return nwd


def resnet50(pretrained=False, pretrained_path=None, **kwargs):
    print('Training from scratch..')
    model = ResNet50(**kwargs)
    return model

####################################################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# from https://github.com/DianCh/AdaContrast/blob/master/classifier.py
# We implement the below models for the source-free source adaptation.

class ResnetForDA(nn.Module):
    def __init__(self, torch_model, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.bottleneck_dim = 256 # TODO: get from args
        self.weight_norm_dim = 0  # TODO: get from args
        self.use_bottleneck = self.bottleneck_dim > 0
        self.use_weight_norm = self.weight_norm_dim >= 0
        
        # 1) Original ResNet
        if not self.use_bottleneck:
            model = torch_model
            modules = list(model.children())[:-1]
            self.encoder = nn.Sequential(*modules)
            self.output_dim = model.fc.in_features
        # 2) Variant ResNet for Source-free source adaptation
        else:
            model = torch_model
            model.fc = nn.Linear(model.fc.in_features, self.bottleneck_dim)
            last_bn1d = nn.BatchNorm1d(self.bottleneck_dim)
            self.encoder = nn.Sequential(model, last_bn1d)
            self.output_dim = self.bottleneck_dim

        self.fc = nn.Linear(self.output_dim, self.num_classes)
        if self.use_weight_norm:
            self.fc = nn.utils.weight_norm(self.fc, dim=self.weight_norm_dim)
    
    def forward(self, x, only_fc=False, only_feat=False, **kwargs):
        if only_fc:
            return self.fc(x)
        
        x = self.encoder(x)
        x = torch.flatten(x, 1)
        out = self.fc(x)
        
        if only_feat:
            return x
        
        result_dict = {'logits':out, 'feat':x}
        return result_dict
    
    def get_params(self):
        """
        Backbone parameters use 1x lr; extra parameters use 10x lr.
        """
        backbone_params = []
        classifier_params = []

        # 1) Original ResNet
        if not self.use_bottleneck:
            backbone_params.extend(self.encoder.parameters())
        # 2) Variant ResNet for Source-free source adaptation
        else:
            resnet = self.encoder[0]
            for module in list(resnet.children())[:-1]:
                backbone_params.extend(module.parameters())
            classifier_params.extend(resnet.fc.parameters())
            classifier_params.extend(self.encoder[1].parameters())
            classifier_params.extend(self.fc.parameters())

        # exclude frozen params
        backbone_params = [param for param in backbone_params if param.requires_grad]
        classifier_params = [param for param in classifier_params if param.requires_grad]

        return backbone_params, classifier_params

def resnet34_torch(pretrained=True, pretrained_path=None, **kwargs): 
    # To facilitate debugging, we copy and paste the torchvision model into our directory.
    # from torchvision.models.resnet import resnet50
    from .resnet_from_torch import resnet34
    model_kwargs = dict(**kwargs)
    if pretrained:
        print('Loading imageNet pretrained model..')
        torch_model = resnet34(weights='ResNet34_Weights.IMAGENET1K_V1') 
    else:
        torch_model = resnet34(weights=None)
    return ResnetForDA(torch_model, **model_kwargs)


def resnet50_torch(pretrained=True, pretrained_path=None, **kwargs): 
    # To facilitate debugging, we copy and paste the torchvision model into our directory.
    # from torchvision.models.resnet import resnet50
    from .resnet_from_torch import resnet50
    model_kwargs = dict(**kwargs)
    if pretrained:
        print('Loading imageNet pretrained model..')
        torch_model = resnet50(weights='ResNet50_Weights.IMAGENET1K_V1') 
    else:
        torch_model = resnet50(weights=None)
    return ResnetForDA(torch_model, **model_kwargs)


def resnet101_torch(pretrained=True, pretrained_path=None, **kwargs):
    # To facilitate debugging, we copy and paste the torchvision model into our directory.
    # from torchvision.models.resnet import resnet101
    from .resnet_from_torch import resnet101
    model_kwargs = dict(**kwargs)
    if pretrained:
        print('Loading imageNet pretrained model..')
        torch_model = resnet101(weights='ResNet101_Weights.IMAGENET1K_V1') 
    else:
        print('Training from scratch..')
        torch_model = resnet101(weights=None)
    return ResnetForDA(torch_model, **model_kwargs)




