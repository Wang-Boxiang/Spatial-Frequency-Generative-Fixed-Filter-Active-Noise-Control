# import numpy as np
# import torch

# class Fixed_filter_controller_2x1x1():
#     def __init__(self, CF_library, ID_vector, frame_length):
#         """
#         2x1x1 ANC 系统的动态固定滤波器控制器。
        
#         参数：
#         - CF_library: 控制滤波器库，形状为 [class, num_channels, filter_length]，
#                       其中 num_channels = 2（对应两个参考麦克风）。
#         - ID_vector: 每帧预测的滤波器索引，形状为 [num_frames, 1]，
#                      每个元素为选用的滤波器类别索引。
#         - frame_length: 每帧包含的采样数。
#         """
#         self.CF_library = CF_library      # 固定滤波器库
#         self.ID_vector = ID_vector        # 每帧滤波器索引
#         self.frame_length = frame_length  # 每帧采样数
#         self.filter_length = CF_library.shape[2]
#         self.num_channels = CF_library.shape[1]   # 应为 2
        
#         # 建立每个参考麦克风的延迟线缓存，形状为 [num_channels, filter_length]
#         self.Xd = torch.zeros(self.num_channels, self.filter_length, dtype=torch.float)
#         # 当前使用的固定滤波器，形状为 [num_channels, filter_length]
#         self.Current_Filter = torch.zeros(self.num_channels, self.filter_length, dtype=torch.float)

#     def noise_cancellation(self, Dis, Fx):
#         """
#         执行固定滤波器噪声消除。
        
#         参数：
#         - Dis: 误差信号（1D数组或 tensor，长度为 N）
#         - Fx: 参考信号，形状为 [N, num_channels]（此处 num_channels = 2）
        
#         返回：
#         - Erro: 每个采样点的误差信号（tensor）
#         """
#         N = Dis.shape[0]            # 总样本数
#         Erro = torch.zeros(N)       # 初始化误差信号输出
#         j = 0                       # ID_vector 的帧索引

#         print(f"Dis dtype: {Dis.dtype}, Fx dtype: {Fx.dtype}, Current_Filter dtype: {self.Current_Filter.dtype}")

#         for ii in range(N):
#             # 更新每个参考麦克风的延迟线（滚动更新）
#             for ch in range(self.num_channels):
#                 self.Xd[ch] = torch.roll(self.Xd[ch], 1, 0)
#                 self.Xd[ch, 0] = Fx[ii, ch]
            
#             # 计算控制信号：各通道滤波器与对应延迟线的内积之和
#             yt = 0
#             for ch in range(self.num_channels):
#                 yt += torch.dot(self.Current_Filter[ch], self.Xd[ch])
            
#             # 计算误差信号
#             e = Dis[ii] - yt
#             Erro[ii] = e.item()
            
#             # 每帧更新一次固定滤波器：假设每 frame_length 个采样点更新一次
#             if (ii + 1) % self.frame_length == 0 and j < len(self.ID_vector):
#                 # 取出当前帧的滤波器索引，注意 ID_vector[j] 为 [filter_index]
#                 filter_index = int(self.ID_vector[j])
#                 # 直接从 CF_library 中提取滤波器，形状为 [num_channels, filter_length]
#                 self.Current_Filter = torch.from_numpy(self.CF_library[filter_index]).float()
#                 j += 1

#         return Erro

import numpy as np
import torch

class Fixed_filter_controller_2x1x1():
    def __init__(self, CF_library, ID_vector, frame_length):
        """
        2x1x1 ANC 系统的动态固定滤波器控制器。
        
        参数：
        - CF_library: 控制滤波器库，形状为 [class, num_channels, filter_length]，
                      其中 num_channels = 2（对应两个参考麦克风）。
        - ID_vector: 每帧预测的滤波器索引，形状为 [num_frames, 1]，
                     每个元素为选用的滤波器类别索引。
        - frame_length: 每帧包含的采样数。
        """
        self.CF_library = CF_library.astype(np.float64)  # 确保库中的滤波器是 float64
        self.ID_vector = ID_vector        
        self.frame_length = frame_length  
        self.filter_length = CF_library.shape[2]
        self.num_channels = CF_library.shape[1]   
        
        # 设定延迟线和当前滤波器为 float64
        self.Xd = torch.zeros(self.num_channels, self.filter_length, dtype=torch.double)
        self.Current_Filter = torch.zeros(self.num_channels, self.filter_length, dtype=torch.double)

    def noise_cancellation(self, Dis, Fx):
        """
        执行固定滤波器噪声消除。
        
        参数：
        - Dis: 误差信号（1D数组或 tensor，长度为 N）
        - Fx: 参考信号，形状为 [N, num_channels]（此处 num_channels = 2）
        
        返回：
        - Erro: 每个采样点的误差信号（tensor）
        """
        N = Dis.shape[0]            
        Erro = torch.zeros(N, dtype=torch.double)  

        if not isinstance(Dis, torch.Tensor):
            Dis = torch.tensor(Dis, dtype=torch.double)
        else:
            Dis = Dis.double()

        if not isinstance(Fx, torch.Tensor):
            Fx = torch.tensor(Fx, dtype=torch.double)
        else:
            Fx = Fx.double()
        
        j = 0  # 滤波器更新索引
        for ii in range(N):
            for ch in range(self.num_channels):
                self.Xd[ch] = torch.roll(self.Xd[ch], 1, 0)
                self.Xd[ch, 0] = Fx[ii, ch]
            
            yt = 0
            for ch in range(self.num_channels):
                yt += torch.dot(self.Current_Filter[ch], self.Xd[ch])
            
            e = Dis[ii] - yt
            Erro[ii] = e.item()
            
            # 更新固定滤波器
            if (ii + 1) % self.frame_length == 0 and j < len(self.ID_vector):
                filter_index = int(self.ID_vector[j])
                self.Current_Filter = torch.from_numpy(self.CF_library[filter_index]).double()
                j += 1
        # print(f"Dis dtype: {Dis.dtype}, Fx dtype: {Fx.dtype}, Current_Filter dtype: {self.Current_Filter.dtype}")
        return Erro



class Fixed_filter_controller_2x1x1_modified():
    def __init__(self, CF_library, frame_length):
        """
        2x1x1 ANC 系统的动态固定滤波器控制器（ID_vector 在 noise_cancellation() 中传入）。
        
        参数：
        - CF_library: 控制滤波器库，形状为 [class, num_channels, filter_length]，
                      其中 num_channels = 2。
        - frame_length: 每帧包含的采样数。
        """
        self.CF_library = CF_library      # 固定滤波器库
        self.frame_length = frame_length  # 每帧采样数
        self.filter_length = CF_library.shape[2]
        self.num_channels = CF_library.shape[1]   # 应为 2
        
        # 建立每个参考麦克风的延迟线缓存
        self.Xd = torch.zeros(self.num_channels, self.filter_length, dtype=torch.float)
        # 当前使用的固定滤波器，形状为 [num_channels, filter_length]
        self.Current_Filter = torch.zeros(self.num_channels, self.filter_length, dtype=torch.float)
        
    def noise_cancellation(self, Dis, Fx, ID_vector):
        """
        执行固定滤波器噪声消除，同时每帧更新一次固定滤波器。
        
        参数：
        - Dis: 误差信号，1D数组或 tensor，长度为 N
        - Fx: 参考信号，形状为 [N, num_channels]（此处 num_channels = 2）
        - ID_vector: 每帧预测的滤波器索引，形状为 [num_frames, 1]
        
        返回：
        - Erro: 每个采样点的误差信号（tensor）
        """
        N = Dis.shape[0]
        Erro = torch.zeros(N)
        j = 0  # 帧索引

        for ii in range(N):
            # 更新各参考麦克风延迟线
            for ch in range(self.num_channels):
                self.Xd[ch] = torch.roll(self.Xd[ch], 1, 0)
                self.Xd[ch, 0] = Fx[ii, ch]
            
            # 计算控制信号
            yt = 0
            for ch in range(self.num_channels):
                yt += torch.dot(self.Current_Filter[ch], self.Xd[ch])
            
            # 计算误差信号
            e = Dis[ii] - yt
            Erro[ii] = e.item()
            
            # 每帧更新固定滤波器
            if (ii + 1) % self.frame_length == 0 and j < len(ID_vector):
                filter_index = int(ID_vector[j])
                self.Current_Filter = torch.from_numpy(self.CF_library[filter_index]).float()
                j += 1
                
        return Erro
    
    
class Fixed_filter_controller_2x1x1_GFANC(): # 直接改num_channels=4应该就行
    def __init__(self, frame_length, control_filters, num_channels=2, filter_length=1024):
        """
        2x1x1 ANC 系统的固定滤波器控制器（新版）。
        每一帧更新控制滤波器，直接由 control_filters 数组提供对应帧的滤波器系数，
        参考之前选择滤波器的逻辑，每帧结束更新下一帧的滤波器。
        
        参数：
        - frame_length: 每帧包含的采样数。
        - num_channels: 参考信号通道数（默认为2）。
        - filter_length: 滤波器长度（需与 control_filters 内滤波器的尺寸匹配）。
        """
        self.frame_length = frame_length
        self.num_channels = num_channels
        self.filter_length = filter_length
        self.control_filters = control_filters

        # 初始化延迟线，每个通道对应一条延迟线，数据类型为 float64
        self.Xd = torch.zeros(self.num_channels, self.filter_length, dtype=torch.float)
        # 当前滤波器初始化为全零，后续将在每帧更新为对应的 control filter
        self.Current_Filter = torch.zeros(self.num_channels, self.filter_length, dtype=torch.float)

    def noise_cancellation(self, Dis, Fx):
        """
        对整段信号执行噪声消除，每一帧使用传入的 control_filters 数组中对应的滤波器系数。
        
        参数：
        - Dis: 误差信号（1D 数组或 tensor，长度为 N）。
        - Fx: 参考信号，形状为 [N, num_channels]，通道数需与初始化时一致。
        
        返回：
        - Erro: 每个采样点的误差信号（tensor）。
        """
        N = Dis.shape[0]
        Erro = torch.zeros(N, dtype=torch.float)
        
        # 确保 Dis 与 Fx 均为 torch tensor 且类型为 float64
        if not isinstance(Dis, torch.Tensor):
            Dis = torch.tensor(Dis, dtype=torch.float)
        else:
            Dis = Dis.float()
            
        if not isinstance(Fx, torch.Tensor):
            Fx = torch.tensor(Fx, dtype=torch.float)
        else:
            Fx = Fx.float()
        
        j = 0  # 控制滤波器更新索引（每一帧一个）
        for ii in range(N):
            # 更新每个通道的延迟线：向右滚动，将当前参考信号插入最前端
            for ch in range(self.num_channels):
                self.Xd[ch] = torch.roll(self.Xd[ch], 1, 0)
                self.Xd[ch, 0] = Fx[ii, ch]
            
            # 计算滤波器输出（卷积运算）
            yt = 0
            for ch in range(self.num_channels):
                yt += torch.dot(self.Current_Filter[ch], self.Xd[ch])
            
            # 计算误差信号
            e = Dis[ii] - yt
            Erro[ii] = e.item()
            
            # 按照帧边界更新当前滤波器（每帧结束后更新下一帧的控制滤波器）
            if (ii + 1) % self.frame_length == 0 and j < self.control_filters.shape[0]:
                self.Current_Filter = torch.tensor(self.control_filters[j], dtype=torch.float)
                j += 1
        
        return Erro


# class Fixed_filter_controller_4x1x1_GFANC:
#     def __init__(self, frame_length, control_filters, filter_length=1024):
#         """
#         4x1x1 ANC 系统的固定滤波器控制器（GFANC）。
#         每帧使用 control_filters 中对应的 4 通道滤波器系数。

#         参数：
#         - frame_length    : 每帧样本数
#         - control_filters : 形状 (num_frames, 4, filter_length) 的滤波器系数数组
#         - filter_length   : 每个滤波器的长度
#         """
#         self.frame_length    = frame_length
#         self.num_channels    = 4
#         self.filter_length   = filter_length
#         self.control_filters = control_filters

#         # 四通道的延迟线 (num_channels × filter_length)
#         self.Xd = torch.zeros(self.num_channels, self.filter_length, dtype=torch.float)
#         # 当前正在使用的 4×filter_length 控制滤波器
#         self.Current_Filter = torch.zeros(self.num_channels, self.filter_length, dtype=torch.float)

#     def noise_cancellation(self, Dis, Fx):
#         """
#         对整段信号执行噪声消除，每帧更新一次滤波器。

#         参数：
#         - Dis: 误差信号，shape (N,)
#         - Fx : 参考信号，shape (N, 4)

#         返回：
#         - Erro: 去噪后每个时刻的误差信号，shape (N,)
#         """
#         # 保证输入为 float Tensor
#         Dis = torch.as_tensor(Dis, dtype=torch.float)
#         Fx  = torch.as_tensor(Fx,  dtype=torch.float)
#         N   = Dis.shape[0]

#         Erro = torch.zeros(N, dtype=torch.float)
#         frame_idx = 0  # 当前帧索引

#         for i in range(N):
#             # 更新四个通道的延迟线
#             for ch in range(self.num_channels):
#                 # 向右滚动一位，然后插入最新的参考值
#                 self.Xd[ch].roll(shifts=1, dims=0)
#                 self.Xd[ch, 0] = Fx[i, ch]

#             # 计算当前输出 y = Σ ch (w_ch · x_ch)
#             y = 0.0
#             for ch in range(self.num_channels):
#                 y = y + torch.dot(self.Current_Filter[ch], self.Xd[ch])

#             # 误差信号
#             e = Dis[i] - y
#             Erro[i] = e

#             # 每帧结束后更新一次滤波器
#             if (i + 1) % self.frame_length == 0 and frame_idx < self.control_filters.shape[0]:
#                 # control_filters[frame_idx] 的形状应为 (4, filter_length)
#                 self.Current_Filter = torch.as_tensor(self.control_filters[frame_idx], dtype=torch.float)
#                 frame_idx += 1

#         return Erro

