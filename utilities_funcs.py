import numpy as np
import torch
import scipy.io as sio
import math

def mapping_function(distance_class, azimuth_class, elevation_class):
    # 定义 distance_class 为 0 的映射关系
    base_mapping = {
        (0, 0): 11,  # (azimuth, elevation) = (0, 0)
        (1, 0): 10,  # (1, 0)
        (2, 0): 9,   # (2, 0)
        (3, 0): 8,   # (3, 0)
        (4, 0): 13,  # (4, 0)
        (5, 0): 12,  # (5, 0)
        (0, 1): 5,   # (0, 1)
        (1, 1): 4,   # (1, 1)
        (2, 1): 3,   # (2, 1)
        (3, 1): 2,   # (3, 1)
        (4, 1): 7,   # (4, 1)
        (5, 1): 6,   # (5, 1)
        (0, 2): 1,   # (0, 2)
    }
    key = (azimuth_class, elevation_class)
    if key not in base_mapping:
        raise ValueError(f"无效的 (方位, 仰角) 组合：{key}") # 要是elevation是2 统一变成1？
    
    base_value = base_mapping[key]
    # 对于 distance_class 为 1、2、3，映射值在基本值上加上 distance_class * 13
    return base_value + distance_class * 13


def normalize_waveform(wave):
    max_val = np.max(np.abs(wave))
    if max_val != 0:
        return wave / max_val
    return wave

def Construt_filter(sub_filters, soft_labels):
    soft_labels = np.expand_dims(soft_labels, axis=0)
    novel_filter = np.matmul(soft_labels, sub_filters) # reconstructed filter based on soft outputs
    return novel_filter, soft_labels

def Load_Pretrained_filters_to_tensor(MAT_FILE):
    mat_contents = sio.loadmat(MAT_FILE)
    Wc_vectors = mat_contents['W']
    # Wc_vectors = Wc_vectors[:8, :] # !!! 8 sub control filters
    Wc_vectors = Wc_vectors[:, :] # !!! 32 sub control filters
    return torch.from_numpy(Wc_vectors).type(torch.float)

# def additional_noise(signal, snr_db): # 对torch类型变量
#     signal_power = signal.norm(p=2)
#     length = signal.shape[1]
#     additional_noise = np.random.randn(length)
#     additional_noise = torch.from_numpy(additional_noise).type(torch.float32).unsqueeze(0)
#     noise_power = additional_noise.norm(p=2)
#     snr = math.exp(snr_db / 10)
#     scale = snr * noise_power / signal_power
#     noisy_signal = (scale * signal + additional_noise) / 2
#     return noisy_signal

def awgn(signal, snr_db): # 对numpy类型变量
    """
    为 numpy 格式的信号添加加性白高斯噪声 (AWGN)，类似于 Matlab 中的 awgn 函数。
    
    参数：
      signal : np.array
          输入信号，可以是1维或多维数组。
      snr_db : float
          想要添加噪声后的信噪比（dB）。
    
    返回：
      noisy_signal : np.array
          添加了白噪声后的信号。
    """
    # 计算输入信号的平均功率（平方均值）
    signal_power = np.mean(signal**2)
    
    # 将 dB 转换为线性信噪比
    snr_linear = 10**(snr_db / 10)
    
    # 根据信号功率和 SNR 计算噪声的功率
    noise_power = signal_power / snr_linear
    
    # 生成与信号同形状的高斯白噪声，标准差为噪声功率的平方根
    noise = np.sqrt(noise_power) * np.random.randn(*signal.shape)
    
    # 将噪声添加到信号
    noisy_signal = signal + noise
    return noisy_signal

def nlms_filter(x, d, filter_len=256, step_size=0.2):
    """
    简易NLMS(归一化LMS)滤波器实现, 近似对应MATLAB的:
    dsp.LMSFilter('Method','Normalized LMS','StepSize', step_size,'Length', filter_len)
    
    参数:
    x          : 输入信号 (参考信号), shape = [N, ]
    d          : 期望输出信号,         shape = [N, ]
    filter_len : 滤波器长度
    step_size  : 步长(学习率)
    
    返回:
    y          : 滤波器输出信号
    e          : 误差信号( e[n] = d[n] - y[n] )
    w          : 最终收敛得到的滤波器系数
    """
    N = len(x)
    w = np.zeros(filter_len)  # 滤波器系数初始化
    e = np.zeros(N)           # 误差信号
    y = np.zeros(N)           # 输出信号
    eps = 1e-8                # 防止除 0 的小量

    # 逐点更新
    for n in range(N):
        # 取当前及之前的若干点作为卷积段
        x_n = np.zeros(filter_len)
        start_index = n - filter_len + 1
        
        # 如果 n < filter_len-1, 相当于左端不够, 补零
        if start_index < 0:
            # x[:n+1] 是当前可用的信号, 倒序填充到 x_n 末端
            x_n[-(n+1):] = x[0 : n+1][::-1]  
        else:
            # 正常情况, 取 x 中一段
            x_n = x[start_index : n+1][::-1]
        
        # 滤波器输出
        y[n] = np.dot(w, x_n)
        # 误差
        e[n] = d[n] - y[n]
        # NLMS 更新 (归一化)
        norm_x_n = np.dot(x_n, x_n) + eps
        w = w + (step_size * e[n] * x_n / norm_x_n)
    
    return y, e, w


def distance_between_points(point1, point2):
    """
    计算两个三维点之间的欧几里得距离。

    参数:
        point1: list or numpy array, e.g., [x1, y1, z1]
        point2: list or numpy array, e.g., [x2, y2, z2]

    返回:
        float: 两点之间的距离
    """
    point1 = np.array(point1)
    point2 = np.array(point2)
    distance = np.linalg.norm(point1 - point2)
    return distance

def nlms_filter_delay(x, d, filter_len=256, step_size=0.2, delay=0):
    """
    修改后的NLMS滤波器实现：
    - 误差信号 e 正常计算：e[n] = d[n] - y[n]
    - 更新控制滤波器系数时，使用延时 delay 后的误差信号，即 e[n+delay]
    
    参数:
      x          : 输入信号 (参考信号), shape = [N, ]
      d          : 期望输出信号,         shape = [N, ]
      filter_len : 滤波器长度
      step_size  : 步长（学习率）
      delay      : 更新时使用延迟的采样点数
      
    返回:
      y          : 滤波器输出信号 (长度为 N)
      e          : 误差信号，计算为 e[n] = d[n]- y[n]
      w          : 最终收敛得到的滤波器系数
    """
    import numpy as np

    N = len(x)
    w = np.zeros(filter_len)   # 初始化滤波器系数
    y = np.zeros(N)            # 输出信号
    e = np.zeros(N)            # 误差信号
    eps = 1e-8                 # 防止除0的小量

    for n in range(N):
        # 构造时刻 n 对应的输入向量（考虑零填充）
        x_n = np.zeros(filter_len)
        start_index = n - filter_len + 1
        if start_index < 0:
            # 当左侧不足 filter_len 个采样时用零填充
            x_n[-(n+1):] = x[0:n+1][::-1]
        else:
            x_n = x[start_index:n+1][::-1]
        
        # 计算当前时刻的输出
        y[n] = np.dot(w, x_n)
        # 正常计算误差信号，不引入延迟
        e[n] = d[n] - y[n]
        
        # 更新控制滤波器时，利用延时后的误差 e[n+delay]
        if n + delay < N:
            norm_x_n = np.dot(x_n, x_n) + eps
            w = w + (step_size * e[n + delay] * x_n / norm_x_n)
    
    return y, e, w


def ambeo_vr_mic_positions(center=(0,0,0),
                               radius=0.0125,
                               azimuths_deg=(0, 180, 90, 270), # Mic 1 (upper, +x), Mic 2 (upper, –x), Mic 3 (lower, +y), Mic 4 (lower, -y)
                               elevations_deg=(35.264,  35.264, -35.264, -35.264)):
    """
    使用球面坐标（azimuth, elevation）计算四点的位置。
    """
    az = np.deg2rad(azimuths_deg)
    el = np.deg2rad(elevations_deg)
    pts = []
    for ai, ei in zip(az, el):
        x = radius * np.cos(ei) * np.cos(ai)
        y = radius * np.cos(ei) * np.sin(ai)
        z = radius * np.sin(ei)
        pts.append([x, y, z])
    return np.array(pts) + np.asarray(center)
# •1: Front Left Up (FLU)
# •2: Front Right Down (FRD)
# •3: Back Left Down (BLD)
# •4: Back Right Up (BRU)


def generate_spherical_data(center,
                            dist_range,
                            el_range,
                            az_range,
                            N,
                            distance_candidates,
                            elevation_candidates,
                            azimuth_candidates):
    """
    生成 N 条随机球面数据，并对距离、仰角、方位角打标签为最接近的候选索引，
    同时返回生成的球坐标。

    Parameters
    ----------
    center : array-like, shape (3,)
        球心坐标 [x0, y0, z0]（单位：m）。
    dist_range : tuple (d_min, d_max)
        随机距离范围（单位：m）。
    az_range : tuple (az_min, az_max)
        随机方位角范围（°，0° 对应 x 轴正方向，逆时针为正）。
    el_range : tuple (el_min, el_max)
        随机仰角范围（°，0° 在赤道面，正向上）。
    N : int
        样本数量。
    distance_candidates : array-like, shape (M_d,)
        距离候选列表。
    azimuth_candidates : array-like, shape (M_az,)
        方位角候选列表（°）。
    elevation_candidates : array-like, shape (M_el,)
        俯仰角候选列表（°）。

    Returns
    -------
    coords : ndarray, shape (N, 3)
        Cartesian 坐标 [x, y, z]。
    labels : ndarray, shape (N, 3), dtype=int
        每条样本的分类索引，按顺序 [dist_idx, el_idx, az_idx]。
    sph_coords : ndarray, shape (N, 3)
        生成的球坐标 [distance, elevation (°), azimuth (°)]。
    """
    center = np.asarray(center).reshape(1,3)
    d_cands = np.asarray(distance_candidates)
    el_cands= np.asarray(elevation_candidates)
    az_cands= np.asarray(azimuth_candidates)

    # 随机生成球坐标
    dists = np.random.uniform(dist_range[0], dist_range[1], size=N)
    els   = np.random.uniform(el_range[0],   el_range[1],   size=N)
    azs   = np.random.uniform(az_range[0],   az_range[1],   size=N)

    # 分类索引（保持 [dist, el, az] 顺序）
    dist_idx = np.abs(dists[:,None] - d_cands[None,:]).argmin(axis=1)
    el_idx   = np.abs(els[:,None]  - el_cands[None,:]).argmin(axis=1)
    az_idx   = np.abs(azs[:,None]  - az_cands[None,:]).argmin(axis=1)
    labels   = np.stack([dist_idx, el_idx, az_idx], axis=1)

    # 球坐标 → 笛卡尔
    az_rad = np.deg2rad(azs)
    el_rad = np.deg2rad(els)
    x = dists * np.cos(el_rad) * np.cos(az_rad)
    y = dists * np.cos(el_rad) * np.sin(az_rad)
    z = dists * np.sin(el_rad)
    coords = np.column_stack([x, y, z]) + center

    # 组合球坐标输出：distance, elevation, azimuth
    sph_coords = np.column_stack([dists, els, azs])

    return coords, labels, sph_coords

def extract_second_number(file_path):
    """
    提取文件名中第二个数字。
    假设文件名格式为：Ref_signal_x_y_z_w.wav，
    分割后 parts[3] 为第二个数字（y）。
    """
    parts = file_path.split('_')
    try:
        return int(parts[3])
    except (IndexError, ValueError):
        # 如果无法提取，返回一个较大的数字保证排在后面
        return float('inf')


def generate_spherical_data_clean(center,
                            dist_range,
                            el_range,
                            az_range,
                            N,
                            distance_candidates,
                            elevation_candidates,
                            azimuth_candidates):
    """
    生成 N 条随机球面数据，并对距离、仰角、方位角打标签为最接近的候选索引，
    同时返回生成的球坐标。

    修改后：
    - 距离仅从 distance_candidates 中选，再在每个中心 ±1cm 内均匀扰动
    - 角度仰角／方位角仍按原随机策略采样

    Parameters
    ----------
    center : array-like, shape (3,)
        球心坐标 [x0, y0, z0]（单位：m）。
    dist_range : tuple (d_min, d_max)
        随机距离范围（单位：m），用于 clip 防超界。
    az_range : tuple (az_min, az_max)
        随机方位角范围（°，0° 对应 x 轴正方向，逆时针为正）。
    el_range : tuple (el_min, el_max)
        随机仰角范围（°，0° 在赤道面，正向上）。
    N : int
        样本数量。
    distance_candidates : array-like, shape (M_d,)
        距离候选中心列表（m）。
    azimuth_candidates : array-like, shape (M_az,)
        方位角候选列表（°）。
    elevation_candidates : array-like, shape (M_el,)
        仰角候选列表（°）。

    Returns
    -------
    coords : ndarray, shape (N, 3)
        Cartesian 坐标 [x, y, z]。
    labels : ndarray, shape (N, 3), dtype=int
        每条样本的分类索引，按顺序 [dist_idx, el_idx, az_idx]。
    sph_coords : ndarray, shape (N, 3)
        生成的球坐标 [distance (m), elevation (°), azimuth (°)]。
    """
    center = np.asarray(center).reshape(1,3)
    d_cands  = np.asarray(distance_candidates)
    el_cands = np.asarray(elevation_candidates)
    az_cands = np.asarray(azimuth_candidates)

    # 1) 距离采样：先选候选中心，再 ±1cm 内扰动，最后 clip 到 dist_range
    # chosen_centers = np.random.choice(d_cands, size=N)
    # perturb = np.random.uniform(-0.01, 0.01, size=N)   # ±0.01 m = ±1 cm
    # dists = chosen_centers + perturb
    # dists = np.clip(dists, dist_range[0], dist_range[1]) # 如果不是扰动非常大用不到
    dists = np.random.choice(d_cands, size=N) # 极端情况：只在candidate里面取

    # 2) 角度随机采样
    els = np.random.uniform(el_range[0], el_range[1], size=N)
    azs = np.random.uniform(az_range[0], az_range[1], size=N)

    # 3) 生成分类索引
    dist_idx = np.abs(dists[:, None] - d_cands[None, :]).argmin(axis=1)
    el_idx   = np.abs(els[:, None]  - el_cands[None, :]).argmin(axis=1)
    az_idx   = np.abs(azs[:, None]  - az_cands[None, :]).argmin(axis=1)
    labels   = np.stack([dist_idx, el_idx, az_idx], axis=1)

    # 4) 球坐标 → 笛卡尔
    az_rad = np.deg2rad(azs)
    el_rad = np.deg2rad(els)
    x = dists * np.cos(el_rad) * np.cos(az_rad)
    y = dists * np.cos(el_rad) * np.sin(az_rad)
    z = dists * np.sin(el_rad)
    coords = np.column_stack([x, y, z]) + center

    # 球坐标输出
    sph_coords = np.column_stack([dists, els, azs])

    return coords, labels, sph_coords


# def sabine_t60(room_sz, beta):
#    alpha = 1 - beta**2
#    Sa = (alpha[0]+alpha[1]) * room_sz[1]*room_sz[2] + \
#         (alpha[2]+alpha[3]) * room_sz[0]*room_sz[2] + \
#         (alpha[4]+alpha[5]) * room_sz[0]*room_sz[1]
#    V = np.prod(room_sz)
#    t60 =  0.161 * V / Sa
#    return t60

def sabine_t60(room_sz, alpha):
   Sa = (alpha[0]+alpha[1]) * room_sz[1]*room_sz[2] + \
        (alpha[2]+alpha[3]) * room_sz[0]*room_sz[2] + \
        (alpha[4]+alpha[5]) * room_sz[0]*room_sz[1]
   V = np.prod(room_sz)
   t60 =  0.161 * V / Sa
   return t60

from scipy.signal import firwin, lfilter, fftconvolve, resample
import soundfile as sf
def process_noise(file_path, target_fs, target_length, cutoff):
    """处理噪声文件的辅助函数"""
    noise_data, noise_sr = sf.read(file_path)
    
    # 如果是立体声，只取一个通道
    if len(noise_data.shape) > 1:
        noise_data = noise_data[:, 0]
    
    # 重采样到目标采样率
    if noise_sr != target_fs:
        num_samples = int(len(noise_data) * target_fs / noise_sr)
        noise_data = resample(noise_data, num_samples)
    
    # 应用2kHz低通滤波器
    lowpass = firwin(1024, cutoff, fs=target_fs, pass_zero=True)
    noise_data = lfilter(lowpass, [1.0], noise_data)
    
    # 标准化噪声幅度
    noise_data = noise_data / np.sqrt(np.mean(noise_data**2)) * 0.3
    
    # 确保噪声长度足够
    if len(noise_data) < target_length:
        repetitions = int(np.ceil(target_length / len(noise_data)))
        noise_data = np.tile(noise_data, repetitions)
    
    return noise_data[:target_length]