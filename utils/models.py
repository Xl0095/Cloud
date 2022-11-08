from torchvision.models import resnet18, resnet50, resnet101


def ResNet18(class_num, pretrained):
    return resnet18(pretrained=pretrained, num_classes=class_num)


def ResNet50(class_num, pretrained):
    return resnet50(pretrained=pretrained, num_classes=class_num)


def ResNet101(class_num, pretrained):
    return resnet101(pretrained=pretrained, num_classes=class_num)
