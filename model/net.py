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
        use_cgtx=False,
        norm_layer=nn.BatchNorm2d,
        use_dropout=False,
        padding_type="reflect",
    ):
        super().__init__()

        backbone = []

        p = 0
        backbone += [
            # nn.ReflectionPad2d(1),
            nn.Conv2d(
                input_nc, nc, kernel_size=1, stride=1, padding=p, bias=False
            ),
            norm_layer(nc),
            nn.ReLU(True)
        ]

        for _ in range(n_blocks):
            backbone += [
                ResnetBlock(
                    nc,
                    padding_type=padding_type,
                    norm_layer=norm_layer,
                    use_dropout=use_dropout,
                    use_bias=True,
                )
            ]

        backbone += [
                nn.Conv2d(
                    nc, output_nc, kernel_size=1, stride=1, padding=p, bias=False
                ),
                nn.Tanh()
            ]

        self.backbone = nn.Sequential(*backbone)
        
        # self.gctx_fusion = nn.Sequential(
        #     nn.Conv2d(
        #         2 * nc, nc, kernel_size=1, stride=1, padding=0, bias=False
        #     ),
        #     norm_layer(nc),
        #     nn.ReLU(True),
        # )

        # self.regress = nn.Sequential(
        #     nn.Conv2d(
        #         nc, output_nc, kernel_size=1, stride=1, padding=0, bias=False
        #     ),
        #     nn.Sigmoid(),
        # )

        # self.use_gctx = use_cgtx
        # print(f"build augmenter with gctx: {self.use_gctx}")


        

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
        return x
        # if (self.use_gctx):
        #     c = F.adaptive_avg_pool2d(x, (1, 1))
        #     c = c.expand_as(x)
        #     x = torch.cat([x, c], 1)
        #     x = self.gctx_fusion(x)

        # p = self.regress(x)
        # if (self.use_gctx):
        #     x_p = input + p
        #     return x_p
        # return p
    
    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
    
    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True

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

def build_augmenter(cfg):
    # if (cfg.MODEL.AUGMENTER.NORM_LAYER == "IN"):
    #     norm_layer = functools.partial(
    #             nn.InstanceNorm2d, affine=False, track_running_stats=False
    #         )
    # else:
    #     norm_layer = nn.BatchNorm2d
    net = Augmenter(3, 3, nc=64, n_blocks=3, norm_layer=nn.BatchNorm2d, use_cgtx=cfg.MODEL.AUGMENTER.GTX)
    # init_network_weights(net, init_type="normal", gain=0.02)
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

    