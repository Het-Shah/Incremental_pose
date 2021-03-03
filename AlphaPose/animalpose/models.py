import torch 
import torch.nn as nn
import torchvision

class DeepLabCut(nn.Module):
	def __init__(self):
		super(DeepLabCut, self).__init__()
		self.resnet50 = torchvision.models.resnet50(pretrained=True)
		self.resnet50 = nn.Sequential(*(list(self.resnet50.children())[:-2]))

		self.deconv1 = nn.Sequential(nn.ConvTranspose2d(2048, 512, kernel_size=(1, 1), stride=(2,2), output_padding=(1,1)),
									 nn.ReLU())

		self.deconv2 = nn.Sequential(nn.ConvTranspose2d(512, 128, kernel_size=(1, 1), stride=(2,2), output_padding=(1,1)),
									 nn.ReLU())
		
		# # self.deconv3 = nn.Sequential(nn.ConvTranspose2d(256, 128, kernel_size=(1, 1), stride=(2,2), output_padding=(1,1)),
		# 							#  nn.ReLU())

		self.deconv4 = nn.Sequential(nn.ConvTranspose2d(128, 17, kernel_size=(1, 1), stride=(2,2), output_padding=(1,1)),
									 nn.ReLU())

	def forward(self, x):
		x = self.resnet50(x)
		x = self.deconv1(x)
		x = self.deconv2(x)
		# x = self.deconv3(x)
		x = self.deconv4(x)
		return x