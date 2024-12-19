import torch.optim
from dataset import SoundDS
from model import AudioCNN
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import classification_report

if __name__ == '__main__':
  num_epochs = 100
  data_path = "D:/projects/project3/data"
  train_dataset = SoundDS(data_path=data_path, dataset_type="train")
  train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4,
    drop_last=True
  )
  test_dataset = SoundDS(data_path=data_path, dataset_type="test")
  test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=4,
    drop_last=False
  )
  model = AudioCNN(num_classes=35)
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
  num_iters = len(train_dataloader)
  if torch.cuda.is_available():
    model.cuda()
  for epoch in range(num_epochs):
    model.train()
    for iter, (aug_sgrams, labels) in enumerate(train_dataloader):
      if torch.all(aug_sgrams == 0):
        continue
      
      if torch.cuda.is_available():
        aug_sgrams = aug_sgrams.cuda()
        labels = labels.cuda()

      # forward
      outputs = model(aug_sgrams)
      loss_value = criterion(outputs, labels)
      if (iter + 1) % 10 == 0:
        print("Epoch {}/{}. Iteration {}/{}. Loss {:.3f}".format(epoch+1, num_epochs, iter+1, num_iters, loss_value))

      # backward
      optimizer.zero_grad()
      loss_value.backward()
      optimizer.step()

    model.eval()
    all_predictions = []
    all_labels = []
    for iter, (aug_sgrams, labels) in enumerate(test_dataloader):
      all_labels.extend(labels)
      if torch.cuda.is_available():
        aug_sgrams = aug_sgrams.cuda()
        labels = labels.cuda()

      with torch.no_grad():
        predictions = model(aug_sgrams)
        indeces = torch.argmax(predictions, dim=1)
        all_predictions.extend(indeces)
        loss_value = criterion(predictions, labels)

    all_labels = [label.item() for label in all_labels]
    all_predictions = [prediction.item() for prediction in all_predictions]
    print("Epoch {}".format(epoch+1))
    print(classification_report(all_labels, all_predictions))
    # print(all_labels)
    # print("-----------------")
    # print(all_predictions)
