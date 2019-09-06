from . import unet


def dn161_unet(num_classes, num_channels=3, pretrained=True):
    return unet.densenet_unet(seg_classes=num_classes, backbone_arch='densenet161')


def dn161_unet_fatter(num_classes, num_channels=3, pretrained=True):
    return unet.densenet_unet(seg_classes=num_classes, backbone_arch='densenet161_fatter')


def dn161_sota_unet(num_classes, num_channels=3, pretrained=True):
    return unet.densenet_unet(seg_classes=num_classes, backbone_arch='densenet161_sota')


def dn121_unet(num_classes, num_channels=3, pretrained=True):
    return unet.densenet_unet(seg_classes=num_classes, backbone_arch='densenet121')


def srx50_unet(num_classes, num_channels=3, pretrained=True):
    return unet.scse_unet(seg_classes=num_classes, backbone_arch='seresnext50')


def sn154_unet(num_classes, num_channels=3, pretrained=True):
    return unet.se_unet(seg_classes=num_classes, backbone_arch='senet154')


def pd_rn154_unet(num_classes, num_channels=3, pretrained=True):
    return unet.resnet_unet(seg_classes=num_classes, backbone_arch='pd_resnet154')


def pd_dn161_unet(num_classes, num_channels=3, pretrained=True):
    return unet.densenet_unet(seg_classes=num_classes, backbone_arch='pd_densenet161')


def rn50_unet(num_classes, num_channels=3, pretrained=True):
    return unet.resnet_unet(seg_classes=num_classes, backbone_arch='resnet50')


def convt_rn50_unet(num_classes, num_channels=3, pretrained=True):
    return unet.convt_resnet_unet(seg_classes=num_classes, backbone_arch='resnet50')


def rn34_unet(num_classes, num_channels=3, pretrained=True):
    return unet.resnet_unet(seg_classes=num_classes, backbone_arch='resnet34')


def rn18_unet(num_classes, num_channels=3, pretrained=True):
    return unet.resnet_unet(seg_classes=num_classes, backbone_arch='resnet18')


def rx101_unet(num_classes, num_channels=3, pretrained=True):
    return unet.resnet_unet(seg_classes=num_classes, backbone_arch='resnext101')
