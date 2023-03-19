import torch
import torch.nn as nn


class LinNet(nn.Module):

	def __init__(self, num_classes, *args):
		super().__init__()
		self.conv1 = nn.Conv2d(3, 32, 7, 1, 3)
		self.bn1 = nn.BatchNorm2d(32)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
		self.bn2 = nn.BatchNorm2d(64)
		self.block1 = Bottleneck(64, 128)
		self.block2 = Bottleneck(128, 256)
		self.block3 = Bottleneck(256, 512)
		self.block4 = Bottleneck(512, 1024)
		self.global_avgpool = nn.AdaptiveAvgPool2d(1)
		self.fc1 = nn.Linear(1024, 512)
		self.classifer = nn.Linear(512, num_classes)

		self._init_params()

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)
		x = self.conv2(x)
		x = self.bn2(x)
		x = self.relu(x)
		x = self.maxpool(x)
		x = self.block1(x)
		x = self.maxpool(x)
		x = self.block2(x)
		x = self.maxpool(x)
		x = self.block3(x)
		x = self.maxpool(x)
		x = self.block4(x)
		x = self.global_avgpool(x)
		x = x.view(x.size(0), -1)
		x = self.fc1(x)
		if not self.training:
			return x

		out = self.classifer(x)

		return out, x


	def _init_params(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")


class Bottleneck(nn.Module):

	def __init__(self, inc, outc):
		super().__init__()
		self.conv1 = nn.Conv2d(inc, outc, 3, 1, 1)
		self.bn1 = nn.BatchNorm2d(outc)
		self.conv2 = nn.Conv2d(outc, inc, 1)
		self.bn2 = nn.BatchNorm2d(inc)
		self.conv3 = nn.Conv2d(inc, outc, 3, 1, 1)
		self.bn3 = nn.BatchNorm2d(outc)
		self.relu = nn.ReLU(inplace=True)
		self.downsample = nn.Conv2d(inc, outc, 1, 1)

	def forward(self, x):
		residual = x
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.conv2(x)
		x = self.bn2(x)
		x = self.relu(x)
		x = self.conv3(x)
		x = self.bn3(x)
		x = self.relu(x)
		out = self.downsample(residual) + x 

		return self.relu(out)


def linnet16(num_classes, **kwargs):
	model = LinNet(num_classes=num_classes)
	return model
