import timm

from configs import CFG


def build_model():
    if CFG.MODEL.NAME == 'resnet-18':
        model = timm.create_model('resnet18', pretrained=True, num_classes=CFG.DATASET.NUM_CLASSES)
    elif CFG.MODEL.NAME == 'resnet-34':
        model = timm.create_model('resnet34', pretrained=True, num_classes=CFG.DATASET.NUM_CLASSES)
    elif CFG.MODEL.NAME == 'resnet-50':
        model = timm.create_model('resnet50', pretrained=True, num_classes=CFG.DATASET.NUM_CLASSES)
    elif CFG.MODEL.NAME == 'resnet-101':
        model = timm.create_model('resnet101', pretrained=True, num_classes=CFG.DATASET.NUM_CLASSES)
    elif CFG.MODEL.NAME == 'resnet-152':
        model = timm.create_model('resnet152', pretrained=True, num_classes=CFG.DATASET.NUM_CLASSES)
    elif CFG.MODEL.NAME == 'seresnet-18':
        model = timm.create_model('seresnet18', pretrained=True, num_classes=CFG.DATASET.NUM_CLASSES)
    elif CFG.MODEL.NAME == 'seresnet-34':
        model = timm.create_model('seresnet34', pretrained=True, num_classes=CFG.DATASET.NUM_CLASSES)
    elif CFG.MODEL.NAME == 'seresnet-50':
        model = timm.create_model('seresnet50', pretrained=True, num_classes=CFG.DATASET.NUM_CLASSES)
    elif CFG.MODEL.NAME == 'seresnet-101':
        model = timm.create_model('seresnet101', pretrained=True, num_classes=CFG.DATASET.NUM_CLASSES)
    elif CFG.MODEL.NAME == 'seresnet-152':
        model = timm.create_model('seresnet152', pretrained=True, num_classes=CFG.DATASET.NUM_CLASSES)
    elif CFG.MODEL.NAME == 'seresnet-152d':
        model = timm.create_model('seresnet152d', pretrained=True, num_classes=CFG.DATASET.NUM_CLASSES)
    elif CFG.MODEL.NAME == 'efficientnet-b0':
        model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=CFG.DATASET.NUM_CLASSES)
    elif CFG.MODEL.NAME == 'efficientnet-b1':
        model = timm.create_model('efficientnet_b1', pretrained=True, num_classes=CFG.DATASET.NUM_CLASSES)
    elif CFG.MODEL.NAME == 'efficientnet-b3':
        model = timm.create_model('efficientnet_b3', pretrained=True, num_classes=CFG.DATASET.NUM_CLASSES)
    elif CFG.MODEL.NAME == 'efficientnet-b4':
        model = timm.create_model('efficientnet_b4', pretrained=True, num_classes=CFG.DATASET.NUM_CLASSES)
    elif CFG.MODEL.NAME == 'noisystudent':
        model = timm.create_model('tf_efficientnet_b0_ns', pretrained=True, num_classes=CFG.DATASET.NUM_CLASSES)
    elif CFG.MODEL.NAME == 'mixnet-xl':
        model = timm.create_model('mixnet_xl', pretrained=True, num_classes=CFG.DATASET.NUM_CLASSES)
    elif CFG.MODEL.NAME == 'mobilenet-v3-large':
        model = timm.create_model('mobilenetv3_large_100', pretrained=True, num_classes=CFG.DATASET.NUM_CLASSES)
    else:
        raise NotImplementedError(f'invalid model: {CFG.MODEL.NAME}')

    return model
