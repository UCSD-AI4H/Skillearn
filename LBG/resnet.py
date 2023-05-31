import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


from torch.autograd import Variable

# __all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

    
class ResNet(nn.Module):
    def __init__(self, in_planes = 32, num_classes=10):
    # def __init__(self, in_planes = 16, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = in_planes
        # num_blocks = [1, 1, 1, 1] #resnet18
        num_blocks = [2, 2, 2, 2] #resnet18
        # num_blocks = [9, 9, 9, 9] #resnet56
        block = BasicBlock
        self.conv1 = nn.Conv2d(3, in_planes, kernel_size=3,stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.layer1 = self._make_layer(block, in_planes, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 2*in_planes, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 4*in_planes, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 8*in_planes, num_blocks[3], stride=2)
        self.linear = nn.Linear(8*in_planes*block.expansion, num_classes)
        self.dropout = nn.Dropout(p=0.2)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        #################dropout
        out = self.dropout(out)
        #################
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        # out = F.avg_pool2d(out, 4)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        # print(out.size())
        return out
    



# class ResNet(nn.Module):
#     def __init__(self, block = BasicBlock, num_blocks = [2, 2, 2, 2], num_classes=10): #18
#     # def __init__(self, block = BasicBlock, num_blocks = [9, 9, 9, 9], num_classes=10): #56
#     # def __init__(self, block=Bottleneck, num_blocks=[3, 4, 6, 3], num_classes=10): #50
#         super(ResNet, self).__init__()
#         self.in_planes = 16

#         self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(16)
#         self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
#         self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
#         self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
#         self.linear = nn.Linear(64, num_classes)
#         # self.log_soft = nn.LogSoftmax(1)
#         self.dropout = nn.Dropout(p=0.2)
        
#         self.apply(_weights_init)

#     def _make_layer(self, block, planes, num_blocks, stride):
#         strides = [stride] + [1]*(num_blocks-1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_planes, planes, stride))
#             self.in_planes = planes * block.expansion

#         return nn.Sequential(*layers)

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         #################dropout
#         out = self.dropout(out)
#         #################
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = F.avg_pool2d(out, out.size()[3])
#         if out.size(0)==0:
#             print(out.size())
#             print(out)
#         out = out.view(out.size(0), -1)
#         out = self.linear(out)
#         #################log
#         # out = self.log_soft(out)
#         return out
    

# import torch
# import torch.nn as nn
# import torch.nn.functional as functional
# import torch.nn.init as init


# def _weights_init(m):
#     if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
#         init.kaiming_normal_(m.weight)


# class LambdaLayer(nn.Module):
#     def __init__(self, lambd):
#         super(LambdaLayer, self).__init__()
#         self.lambd = lambd

#     def forward(self, x):
#         return self.lambd(x)


# class BasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self, in_planes, planes, stride=1, option="A"):
#         super(BasicBlock, self).__init__()
#         self.conv1 = nn.Conv2d(
#             in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
#         )
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(
#             planes, planes, kernel_size=3, stride=1, padding=1, bias=False
#         )
#         self.bn2 = nn.BatchNorm2d(planes)

#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != planes:
#             if option == "A":
#                 """
#                 For CIFAR10 ResNet paper uses option A.
#                 """
#                 self.shortcut = LambdaLayer(
#                     lambda x: functional.pad(
#                         x[:, :, ::2, ::2],
#                         (0, 0, 0, 0, planes // 4, planes // 4),
#                         "constant",
#                         0,
#                     )
#                 )
#             elif option == "B":
#                 self.shortcut = nn.Sequential(
#                     nn.Conv2d(
#                         in_planes,
#                         self.expansion * planes,
#                         kernel_size=1,
#                         stride=stride,
#                         bias=False,
#                     ),
#                     nn.BatchNorm2d(self.expansion * planes),
#                 )

#     def forward(self, x):
#         out = functional.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out += self.shortcut(x)
#         out = functional.relu(out)
#         return out


# class ResNet32(nn.Module):
#     def __init__(self, num_classes=10, block=BasicBlock, num_blocks=[5, 5, 5]):
#         super(ResNet32, self).__init__()
#         self.in_planes = 16

#         self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(16)
#         self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
#         self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
#         self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
#         self.linear = nn.Linear(64, num_classes)

#         self.apply(_weights_init)

#     def _make_layer(self, block, planes, num_blocks, stride):
#         strides = [stride] + [1] * (num_blocks - 1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_planes, planes, stride))
#             self.in_planes = planes * block.expansion

#         return nn.Sequential(*layers)

#     def forward(self, x):
#         out = functional.relu(self.bn1(self.conv1(x)))
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = functional.avg_pool2d(out, out.size()[3])
#         out = out.view(out.size(0), -1)
#         out = self.linear(out)
#         return out


# class ResNet32MWN(nn.Module):
#     def __init__(self, num_classes=10, block=BasicBlock, num_blocks=[5, 5, 5]):
#         super(ResNet32MWN, self).__init__()
#         self.in_planes = 16
#         self.num_classes = num_classes

#         self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(16)
#         self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
#         self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
#         self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
#         self.linear = nn.Linear(64, num_classes)

#         self.fc = nn.Linear(2 * num_classes, 1)

#         self.apply(_weights_init)

#     def _make_layer(self, block, planes, num_blocks, stride):
#         strides = [stride] + [1] * (num_blocks - 1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_planes, planes, stride))
#             self.in_planes = planes * block.expansion

#         return nn.Sequential(*layers)

#     def forward(self, x, y):
#         out = functional.relu(self.bn1(self.conv1(x)))
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = functional.avg_pool2d(out, out.size()[3])
#         out = out.view(out.size(0), -1)
#         out = self.linear(out)
#         out = functional.softmax(out, dim=-1)

#         one_hot = functional.one_hot(y, num_classes=self.num_classes)
#         out = torch.cat([out, one_hot], dim=1)
#         out = self.fc(out)
#         return torch.sigmoid(out)


# class HiddenLayer(nn.Module):
#     def __init__(self, input_size, output_size):
#         super(HiddenLayer, self).__init__()
#         self.fc = nn.Linear(input_size, output_size)
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         return self.relu(self.fc(x))


# class MLP(nn.Module):
#     def __init__(self, hidden_size=500, num_layers=1):
#         super(MLP, self).__init__()
#         self.first_hidden_layer = HiddenLayer(1, hidden_size)
#         self.rest_hidden_layers = nn.Sequential(
#             *[HiddenLayer(hidden_size, hidden_size) for _ in range(num_layers - 1)]
#         )
#         self.output_layer = nn.Linear(hidden_size, 1)

#     def forward(self, x):
#         x = self.first_hidden_layer(x)
#         x = self.rest_hidden_layers(x)
#         x = self.output_layer(x)
#         return torch.sigmoid(x)
