import os
import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset
import ast  # <--- 新增，用于安全地解析字符串为Python对象

# ========================
# 修改1：STFT计算函数改为同时计算幅度谱和相位谱
# 原来使用的是 torchaudio.transforms.Spectrogram，现改用 torch.stft 获取复数谱，再分离出幅度和相位
# ========================
def compute_stft_for_allchannels(signal_nch):
    # 参数设置
    n_fft = 1024
    hop_length = 64 # 64
    window = torch.hann_window(n_fft)
    
    # 分别计算每个通道的STFT，得到幅度谱和相位谱，并拼接成4通道输出
    magnitude_list = []
    phase_list = []
    for c in range(signal_nch.shape[0]):
        # signal_nch[c, :] shape: [signal_length]
        channel_signal = signal_nch[c, :]
        # 使用 torch.stft 返回复数张量
        stft_result = torch.stft(channel_signal, n_fft=n_fft, hop_length=hop_length, window=window,
                                   return_complex=True)
        # 计算幅度谱和相位谱
        mag = torch.abs(stft_result)
        phase = torch.angle(stft_result)
        magnitude_list.append(mag.unsqueeze(0))  # 新增一个维度，作为channel维度
        phase_list.append(phase.unsqueeze(0))
    # 将4个通道的幅度谱和相位谱分别拼接，然后沿channel维度拼接，得到8通道数据
    mag_all = torch.cat(magnitude_list, dim=0)      # shape: [8, freq, time]
    phase_all = torch.cat(phase_list, dim=0)          # shape: [8, freq, time]
    combined_spec = torch.cat((mag_all, phase_all), dim=0)  # shape: [8, freq, time]
    return combined_spec



class MyNoiseDataset(Dataset):
    """
    返回 (spectrogram, (distance_class, azimuth_class, elevation_class, regression_targets))
    regression_targets: 8个浮点数
    """
    def __init__(self, folder, annotations_file):
        self.folder = folder
        self.annotations_file = pd.read_csv(os.path.join(folder, annotations_file))
        
    def __len__(self):
        return len(self.annotations_file)
    
    def __getitem__(self, index):
        # 获取2个音频样本路径
        audio_sample_path1, audio_sample_path2, audio_sample_path3, audio_sample_path4 = self._get_audio_sample_paths(index)
        # 获取三个分类标签 + 回归标签
        labels = self._get_audio_sample_labels(index)
        
        # 加载2个音频信号
        signal1, _ = torchaudio.load(os.path.join(self.folder, audio_sample_path1))
        signal2, _ = torchaudio.load(os.path.join(self.folder, audio_sample_path2))
        signal3, _ = torchaudio.load(os.path.join(self.folder, audio_sample_path3))
        signal4, _ = torchaudio.load(os.path.join(self.folder, audio_sample_path4))
        
        # 合并成 [4, signal_length]
        signal = torch.cat((signal1, signal2, signal3, signal4), dim=0)
        
        # 计算 STFT，输出8通道数据（4幅度+4相位）
        spec = compute_stft_for_allchannels(signal)
        # 进行正则化
        # spec = normalize_spec(spec)
        
        return spec, labels
        
    def _get_audio_sample_paths(self, index):
        path1 = self.annotations_file.iloc[index, 0]
        path2 = self.annotations_file.iloc[index, 1]
        path3 = self.annotations_file.iloc[index, 2]
        path4 = self.annotations_file.iloc[index, 3]
        return path1, path2, path3, path4
    
    def _get_audio_sample_labels(self, index):
        # 距离、方位角、仰角分类
        label_distance = int(self.annotations_file.iloc[index, 4])
        label_elevation = int(self.annotations_file.iloc[index, 5])
        label_azimuth = int(self.annotations_file.iloc[index, 6])
        
        # ** 新增：回归向量 (8个数字) **
        regression_str = self.annotations_file.iloc[index, 7]
        regression_list = ast.literal_eval(regression_str)
        regression_tensor = torch.tensor(regression_list, dtype=torch.float32)
        
        return label_distance, label_elevation, label_azimuth, regression_tensor

class MyNoiseDataset1(Dataset):
    """
    返回 (audio_paths, spectrogram, (distance_class, azimuth_class, elevation_class, regression_targets))
    regression_targets: 8个浮点数
    """
    def __init__(self, folder, annotations_file):
        self.folder = folder
        self.annotations_file = pd.read_csv(os.path.join(folder, annotations_file))
        
    def __len__(self):
        return len(self.annotations_file)
    
    def __getitem__(self, index):
        # 获取2个音频样本路径
        audio_sample_path1, audio_sample_path2, audio_sample_path3, audio_sample_path4 = self._get_audio_sample_paths(index)
        # 获取三个分类标签 + 回归标签
        labels = self._get_audio_sample_labels(index)
        
        # 加载2个音频信号
        signal1, _ = torchaudio.load(os.path.join(self.folder, audio_sample_path1))
        signal2, _ = torchaudio.load(os.path.join(self.folder, audio_sample_path2))
        signal3, _ = torchaudio.load(os.path.join(self.folder, audio_sample_path3))
        signal4, _ = torchaudio.load(os.path.join(self.folder, audio_sample_path4))
        
        # 合并成 [4, signal_length]
        signal = torch.cat((signal1, signal2, signal3, signal4), dim=0)
        
        # 计算 STFT，输出8通道数据（4幅度+4相位）
        spec = compute_stft_for_allchannels(signal)
        # 进行正则化
        # spec = normalize_spec(spec)
        
        return (audio_sample_path1, audio_sample_path2, audio_sample_path3, audio_sample_path4), spec, labels
        
    def _get_audio_sample_paths(self, index):
        path1 = self.annotations_file.iloc[index, 0]
        path2 = self.annotations_file.iloc[index, 1]
        path3 = self.annotations_file.iloc[index, 2]
        path4 = self.annotations_file.iloc[index, 3]
        return path1, path2, path3, path4
    
    def _get_audio_sample_labels(self, index):
        # 距离、方位角、仰角分类
        label_distance = int(self.annotations_file.iloc[index, 4])
        label_elevation = int(self.annotations_file.iloc[index, 5])
        label_azimuth = int(self.annotations_file.iloc[index, 6])
        
        # ** 新增：回归向量 (8个数字) **
        regression_str = self.annotations_file.iloc[index, 7]
        regression_list = ast.literal_eval(regression_str)
        regression_tensor = torch.tensor(regression_list, dtype=torch.float32)
        
        return label_distance, label_elevation, label_azimuth, regression_tensor