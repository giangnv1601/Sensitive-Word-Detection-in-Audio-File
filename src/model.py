import torch
import torch.nn as nn

class AudioCNN (nn.Module):
  def __init__(self, num_classes = 35):
    super().__init__()
    self.conv1 = self.make_block(in_channels=1, out_channels=8)
    self.conv2 = self.make_block(in_channels=8, out_channels=16)
    self.conv3 = self.make_block(in_channels=16, out_channels=32)
    self.conv4 = self.make_block(in_channels=32, out_channels=64)
    self.conv5 = self.make_block(in_channels=32, out_channels=32)

    self.fc1 = nn.Sequential(
      nn.Dropout(p=0.5),
      nn.Linear(in_features=512, out_features=521),
      nn.LeakyReLU()
    )

    self.fc2 = nn.Sequential(
      nn.Dropout(p=0.5),
      nn.Linear(in_features=521, out_features=1024),
      nn.LeakyReLU()
    )

    self.fc3 = nn.Sequential(
      nn.Dropout(p=0.5),
      nn.Linear(in_features=1024, out_features=num_classes),
    )

  def make_block(self, in_channels, out_channels):
    return nn.Sequential(
      nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding="same"),
      nn.BatchNorm2d(num_features=out_channels),
      nn.LeakyReLU(),
      nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding="same"),
      nn.BatchNorm2d(num_features=out_channels),
      nn.LeakyReLU(),
      nn.MaxPool2d(kernel_size=2)
    )

  def forward(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.conv4(x)
    # x = self.conv5(x)
    x = x.view(x.shape[0], -1)
    x = self.fc1(x)
    x = self.fc2(x)
    x = self.fc3(x)
    return x

if __name__ == '__main__':
  model = AudioCNN()
  input_data = torch.rand(8, 1, 64, 32)
  if torch.cuda.is_available():
    model.cuda()
    input_data = input_data.cuda()
  while True:
    result = model(input_data)
    print(result.shape)
    break