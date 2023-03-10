import torch
import torch.nn as nn
import torch.nn.functional as F
import pretrainedmodels
from torchvision import models
import ssl

# 全局取消证书验证
ssl._create_default_https_context = ssl._create_unverified_context

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y

class BaseModel(nn.Module):
    def __init__(self, model_name, num_classes=2, pretrained=True, pool_type='max') -> None:
        super().__init__()
        self.model_name = model_name
        print(model_name)
        assert model_name in ["resnext50", "se_resnext50"]
        if model_name == 'resnext50':
            backbone = nn.Sequential(*list(models.resnext50_32x4d(pretrained=pretrained).children())[:-2])
        else:
            if pretrained:
                model = pretrainedmodels.__dict__["se_resnext50_32x4d"](num_classes=1000, pretrained='imagenet')
            else:
                model = pretrainedmodels.__dict__['se_resnext50_32x4d'](pretrained=None)
            backbone = nn.Sequential(*list(model.children())[:-2])
        plane = 2048

        self.backcone = backbone

        assert pool_type in ['avg', 'max']
        if pool_type == 'avg':
            self.pool = nn.AdaptiveMaxPool2d((1,1))
        if pool_type == 'max':
            self.pool = nn.AdaptiveMaxPool2d((1,1))

        # if down:
        #     if pool_type == 'cat':
        #         self.down = nn.Sequential(
        #             nn.Linear(plane * 2, plane),
        #             nn.BatchNorm1d(plane),
        #             nn.Dropout(0.2),
        #             nn.ReLU(True)
        #         )
        #     else:
        #         self.down = nn.Sequential(
        #             nn.Linear(plane, plane),
        #             nn.BatchNorm1d(plane),
        #             nn.Dropout(0.2),
        #             nn.ReLU(True)
        #         ) 

        self.se = SELayer(plane)
        self.hidden = nn.Linear(plane, plane)
        self.relu = nn.ReLU(True)

        self.metric = nn.Linear(plane, num_classes)

    def forward(self, x):
        feat = self.pool(self.backcone(x))        
        se = self.se(feat).view(feat.size(0), -1)
        feat_flat = feat.view(feat.size(0), -1)
        feat_flat = self.relu(self.hidden(feat_flat) * se)
        out = self.metric(feat_flat)
        return out
    

if __name__ == '__main__':
    model = BaseModel(model_name='resnext50').eval()
    x = torch.randn((1,3,520,520))
    out = model(x)
    print(out.size())

