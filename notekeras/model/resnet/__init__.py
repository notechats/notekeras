from notekeras.layer import BatchNormalizationFreeze
from .classifiers import ResNet152Classifier, ResNet200Classifier
from .classifiers import ResNet18Classifier, ResNet34Classifier, ResNet50Classifier, ResNet101Classifier
from .core import FPN2D, FPN2D18, FPN2D34, FPN2D50, FPN2D101, FPN2D152, FPN2D200
from .core import ResNet1D, ResNet1D18, ResNet1D34, ResNet1D50, ResNet1D101, ResNet1D152, ResNet1D200
from .core import ResNet2D, ResNet2D18, ResNet2D34, ResNet2D50, ResNet2D101, ResNet2D152, ResNet2D200
from .core import ResNet3D, ResNet3D18, ResNet3D34, ResNet3D50, ResNet3D101, ResNet3D152, ResNet3D200
from .core import TimeDistributedResNet, TimeDistributedResNet18, TimeDistributedResNet34, TimeDistributedResNet50
from .core import TimeDistributedResNet101, TimeDistributedResNet152, TimeDistributedResNet200

ResNet = ResNet2D
ResNet18 = ResNet2D18
ResNet34 = ResNet2D34
ResNet50 = ResNet2D50
ResNet101 = ResNet2D101
ResNet152 = ResNet2D152
ResNet200 = ResNet2D200

custom_objects = {'BatchNormalization': BatchNormalizationFreeze, }
