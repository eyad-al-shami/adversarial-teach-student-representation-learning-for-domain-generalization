import torch
import torch.nn as nn
from torch.nn import functional as F
import functools
import torchmetrics
class ResnetBlock(nn.Module):

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super().__init__()
        self.conv_block = self.build_conv_block(
            dim, padding_type, norm_layer, use_dropout, use_bias
        )

    def build_conv_block(
        self, dim, padding_type, norm_layer, use_dropout, use_bias
    ):
        conv_block = []
        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        else:
            raise NotImplementedError(
                "padding [%s] is not implemented" % padding_type
            )

        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
            nn.ReLU(True),
        ]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        else:
            raise NotImplementedError(
                "padding [%s] is not implemented" % padding_type
            )
        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
        ]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class Augmenter(nn.Module):
    """Fully convolutional network."""

    def __init__(
        self,
        input_nc,
        output_nc,
        nc=32,
        n_blocks=3,
        use_cgtx=True,
        norm_layer=nn.BatchNorm2d,
        use_dropout=False,
        padding_type="reflect",
    ):
        super().__init__()

        p=0
        backbone = [nn.ReflectionPad2d(1)]
        backbone += [
            nn.Conv2d(
                input_nc, nc, kernel_size=3, stride=1, padding=p, bias=False
            )
        ]
        backbone += [norm_layer(nc)]
        backbone += [nn.ReLU(True)]

        for _ in range(n_blocks):
            backbone += [
                ResnetBlock(
                    nc,
                    padding_type=padding_type,
                    norm_layer=norm_layer,
                    use_dropout=use_dropout,
                    use_bias=False,
                )
            ]
        self.backbone = nn.Sequential(*backbone)
        
        self.gctx_fusion = nn.Sequential(
            nn.Conv2d(
                2 * nc, nc, kernel_size=1, stride=1, padding=0, bias=False
            ),
            norm_layer(nc),
            nn.ReLU(True),
        )

        self.regress = nn.Sequential(
            nn.Conv2d(
                nc, output_nc, kernel_size=1, stride=1, padding=0, bias=False
            ),
            nn.Tanh(),
        )

        self.use_gctx = use_cgtx

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): input mini-batch.
            lmda (float): multiplier for perturbation.
            return_p (bool): return perturbation.
            return_stn_output (bool): return the output of stn.
        """
        input = x
        x = self.backbone(x)
        if (self.use_gctx):
            c = F.adaptive_avg_pool2d(x, (1, 1))
            c = c.expand_as(x)
            x = torch.cat([x, c], 1)
            x = self.gctx_fusion(x)

        p = self.regress(x)
        x_p = input + p
        return x_p
    
    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
    
    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True

def init_network_weights(model, init_type="normal", gain=0.02):

    def _init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if init_type == "normal":
                nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_type == "xavier":
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == "kaiming":
                nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                nn.init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError(
                    "initialization method {} is not implemented".
                    format(init_type)
                )
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:
            nn.init.constant_(m.weight.data, 1.0)
            nn.init.constant_(m.bias.data, 0.0)
        elif classname.find("InstanceNorm2d") != -1:
            if m.weight is not None and m.bias is not None:
                nn.init.constant_(m.weight.data, 1.0)
                nn.init.constant_(m.bias.data, 0.0)

    model.apply(_init_func)

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class BackBone(nn.Module):
    def __init__(self, cfg, component='teacher'):
        super(BackBone, self).__init__()
        resnet = torch.hub.load(
            'pytorch/vision:v0.10.0', 
            cfg.MODEL[component.upper()].BACKBONE_NAME, 
            pretrained=cfg.MODEL[component.upper()].PRETRAINED
            )
        # use resnet as backbone without the last layer
        resnet.fc = Identity()
        # self.net = nn.Sequential(*list(resnet.children())[:-1])
        self.net = resnet

    def forward(self, x):
        b, c, _,__ = x.shape
        return self.net(x).view(b, -1)
  
    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
    
    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True

class ClassifierLayer(nn.Module):
    # the output is the logits and not the softmax
    def __init__(self, cfg):
        super(ClassifierLayer, self).__init__()
        self.fc = nn.Linear(cfg.MODEL.BACKBONE_OUT_DIM, cfg.DATASET.CLASSES)
    def forward(self, x):
        return self.fc(x)

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
    
    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True

def build_augmenter():
    norm_layer = functools.partial(
            nn.InstanceNorm2d, affine=False, track_running_stats=False
        )
    net = Augmenter(3, 3, nc=64, n_blocks=3, norm_layer=norm_layer)
    init_network_weights(net, init_type="normal", gain=0.02)
    return net

class Metrics_Monitor():
    def __init__(self, cfg) -> None:
        self.metrics = {}
        if "accuracy" in cfg.METRICS.MONITOR:
            self.metrics["acc"] = torchmetrics.Accuracy(task=cfg.METRICS.ACCURACY.TASK, num_classes=cfg.DATASET.CLASSES)
        if "loss" in cfg.METRICS.MONITOR:
            self.metrics["loss"] = torchmetrics.MeanMetric()
    def reset(self):
        for m in self.metrics.values():
            m.reset()
