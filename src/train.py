import torch.optim
from dataset import SoundDS
from model import AudioCNN
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import classification_report, accuracy_score

from argparse import ArgumentParser
from tqdm.autonotebook import tqdm
from torch.utils.tensorboard import SummaryWriter

def get_args():
  parser = ArgumentParser(description="CNN Training")
  parser.add_argument("--root", "-r", type=str, default="D:/projects/project3/data", help="Root of the dataset")
  parser.add_argument("--epochs", "-e", type=int, default=100, help="Number of epochs")
  parser.add_argument("--batch_size", "-b", type=int, default=8, help="Batch size")
  parser.add_argument("--logging", "-l", type=str, default="tensorboard")
  args = parser.parse_args()
  return args

if __name__ == '__main__':
  args = get_args()
  train_dataset = SoundDS(data_path=args.root, dataset_type="train")
  train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=4,
    drop_last=True
  )
  test_dataset = SoundDS(data_path=args.root, dataset_type="test")
  test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=4,
    drop_last=False
  )
  writer = SummaryWriter(args.logging)
  model = AudioCNN(num_classes=35)
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
  num_iters = len(train_dataloader)
  if torch.cuda.is_available():
    model.cuda()
  for epoch in range(args.epochs):
    model.train()
    progress_bar = tqdm(train_dataloader, colour="cyan")
    for iter, (aug_sgrams, labels) in enumerate(progress_bar):
      if torch.all(aug_sgrams == 0):
        continue
      
      if torch.cuda.is_available():
        aug_sgrams = aug_sgrams.cuda()
        labels = labels.cuda()

      # forward
      outputs = model(aug_sgrams)
      loss_value = criterion(outputs, labels)
      
      progress_bar.set_description("Epoch {}/{}. Iteration {}/{}. Loss {:.3f}".format(epoch+1, args.epochs, iter+1, num_iters, loss_value))
      writer.add_scalar("Train/Loss", loss_value, epoch*num_iters+iter)

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
    accuracy = accuracy_score(all_labels, all_predictions)
    print("Epoch {}: Accuracy: {}".format(epoch+1, accuracy))
    writer.add_scalar("Val/Accuracy", loss_value, epoch)

    # print(classification_report(all_labels, all_predictions))
    # print(all_labels)
    # print("-----------------")
    # print(all_predictions)
