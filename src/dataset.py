from torch.utils.data import DataLoader, Dataset, random_split
import torch
import os
from preprocessing import AudioUtil

import matplotlib.pyplot as plt

def visualize_spectrogram(spectrogram, label, title="Spectrogram Visualization"):
    plt.figure(figsize=(10, 4))
    plt.title(f"{title} - Label: {label}")
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.imshow(spectrogram[0].numpy(), aspect="auto", origin="lower", cmap="viridis")
    plt.colorbar(format='%+2.0f dB')
    plt.show()

class SoundDS(Dataset):
  def __init__(self, data_path, dataset_type):
    self.data_path = data_path
    self.duration = 1000
    self.sr = 16000
    self.channel = 1
    self.shift_pct = 0.2
  
    # print(os.listdir(self.data_path))
    self.categories = ['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero']
    self.audios_path = []
    self.labels = []

    if dataset_type == "train":
      list_file_path = os.path.join(self.data_path, "training_list.txt")
    elif dataset_type == "validation":
      list_file_path = os.path.join(self.data_path, "validation_list.txt")
    elif dataset_type == "test":
      list_file_path = os.path.join(self.data_path, "testing_list.txt")
    else:
      raise ValueError("Invalid dataset type. Must be 'train', 'validation', or 'test'.")
    
    with open(list_file_path, "r") as f:
      lines = f.readlines()
      for line in lines:
        relative_path = line.strip() # Loại bỏ khoảng trắng dư thừa ở đầu và cuối dòng
        full_path = os.path.join(self.data_path, relative_path) 
        self.audios_path.append(full_path)

        # Xác định nhãn từ danh mục lớp
        category = relative_path.split('/')[0]  
        if category in self.categories:
          label = self.categories.index(category)
          self.labels.append(label)
        else:
          raise ValueError(f"Unknown category '{category}' in dataset.")

  def __len__(self):
    return len(self.labels)   

  def __getitem__(self, idx):
    # Thêm cơ chế ghi lại tệp bị lỗi
    error_log = []
    audio_file = self.audios_path[idx]
    label = self.labels[idx]

    try:
      aud = AudioUtil.open(audio_file)  # Đọc tệp âm thanh
      reaud = AudioUtil.resample(aud, self.sr)
      rechan = AudioUtil.rechannel(reaud, self.channel)
      dur_aud = AudioUtil.pad_trunc(rechan, self.duration)
      shift_aud = AudioUtil.time_shift(dur_aud, self.shift_pct)
      sgram = AudioUtil.spectro_gram(shift_aud, n_mels=64, n_fft=1024, hop_len=None)
      aug_sgram = AudioUtil.spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)

      return aug_sgram, label

    except Exception as e:
      error_log.append(audio_file)
      # Nếu có lỗi khi đọc tệp, bỏ qua tệp và in thông báo lỗi
      print(f"Error processing {audio_file}: {e}")
      return torch.zeros(1, 64, 32), label
  
if __name__ == '__main__':
  data_path = "D:/projects/project3/data"
  dataset = SoundDS(data_path=data_path, dataset_type="train")

  # In các mục trong data_path
  print(dataset.data_path)

  # Tính số loại nhãn khác nhau
  unique_labels = set(dataset.labels)
  print(len(unique_labels))

  # In ra một cặp audio - label sau khi xử lý
  aug_sgram, label = dataset.__getitem__(6234)
  print(aug_sgram.shape)
  print(type(aug_sgram))
  print(label)
  
  # Visualize để xem 
  visualize_spectrogram(aug_sgram, label)