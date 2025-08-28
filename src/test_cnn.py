from argparse import ArgumentParser

import torch
import torch.nn as nn
from dataset import visualize_spectrogram
from model import AudioCNN
from preprocessing import AudioUtil

def get_args():
  parser = ArgumentParser(description="CNN inference")
  parser.add_argument("--audio_file", "-a", type=str, default="D:/projects/project3/data/four/3c4aa5ef_nohash_0.wav")
  parser.add_argument("--checkpoint", "-c", type=str, default="trained_models/best_cnn.pt")
  args = parser.parse_args()
  return args
if __name__ == '__main__':
  categories = ['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero']
  args = get_args()
  if torch.cuda.is_available():
    device = torch.device("cuda")
  else:
    device = torch.device("cpu")

  model = AudioCNN(num_classes=35).to(device)
  if args.checkpoint:
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint["model"])
  else:
    print("No checkpoint found!")
    exit(0)

  model.eval()
  aud = AudioUtil.open(args.audio_file) 
  reaud = AudioUtil.resample(aud, 16000)
  rechan = AudioUtil.rechannel(reaud, 1)
  dur_aud = AudioUtil.pad_trunc(rechan, 1000)
  sgram = AudioUtil.spectro_gram(dur_aud, n_mels=64, n_fft=1024, hop_len=None)
  aug_sgram = sgram.unsqueeze(0).to(device)
  softmax = nn.Softmax()
  with torch.no_grad():
    output = model(aug_sgram)
    # print(output)
    probs = softmax(output)
    # print(probs)

  max_idx = torch.argmax(probs)
  predicted_class = categories[max_idx]
  print("The test audio is about {} with confident score of {:.2f}".format(predicted_class, probs[0, max_idx]*100))
  visualize_spectrogram(aug_sgram.squeeze(0), predicted_class)
  