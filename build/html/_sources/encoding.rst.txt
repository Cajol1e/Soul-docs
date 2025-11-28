编码方式
========

本章，我们将介绍Soul工具包支持的编码方式，通过不同的编码方式，我们可以将外部输入信息（如图像、音频、传感器数据）转换为各种脉冲神经网络可识别的脉冲序列的规则或方法。
Soul工具包支持以下编码方式：

binary_parallel
----------------
介绍：
    并行二进制编码：在时间步 t=0…B-1（t 小于 B 时）对 B 个比特进行编码，具体方式为生成同步脉冲，并在时间窗口内重复（本文中简化为仅在 t<B 时发射一次）。
引用：
    Auge et al., Survey (Correlation & synchrony – parallel binary).

binary_sequential
----------------
介绍：
    时序二进制编码：将每个数值量化为 B 位整数，并在不同时间步上发射比特（时间步 t 对应比特位序号）。输入范围为 [0,1]。
引用：
    Auge et al., Survey (Global referenced binary – sequential).

bsa
----------------
介绍：
    具有形状感知路由的本氏脉冲算法（Ben's Spiker Algorithm）：
        * 视觉（输入输出维度）：(C,H,W) -> (T,C,H,W)
        * 运动（输入输出维度）：(W,C) -> (T,W,C)
    per-channel min-max normalization; causal kernel; greedy residue update;
    强度 -> 延迟映射
引用：
    Schrauwen & Van Campenhout, IJCNN 2003 (BSA)
    Auge et al., Neural Processing Letters 2021 (Survey)

burst
----------------
介绍：
    爆发式编码编码器。更强的输入会在 T 个时间步内发射更密集的脉冲包（即 “爆发脉冲”）。每个元素会发射 B 个脉冲，脉冲之间的间隔（ISI，Inter-Spike Interval）由输入决定，且发射时序遵循确定性调度规则。
引用：
    Guo et al., "Neural Coding in Spiking Neural Networks: A Comparative Study for Robust Neuromorphic Systems",Frontiers in Neuroscience, 2021 (burst definition & properties)

direct
----------------
介绍：
    脉冲神经网络（SNN）输入的直接编码机制实现。

phase
----------------
介绍：
    输入的二进制相位编码（多脉冲型）。该编码方式通过贪心二进制分数展开，将 [0,1] 区间内的归一化数值映射为跨 T 个相位的脉冲序列。
引用：
    Hwang & Kung, "One-Spike SNN: Single-Spike Phase Coding with Base Manipulation for ANN-to-SNN Conversion Loss Minimization", IEEE 2025.

poisson
----------------
介绍：
    泊松编码。在编码器内部将输入归一化到 [0,1] 区间，随后在每个时间步以概率 p（该概率等于归一化后的输入值）发射一个脉冲。输出遵循 “时间优先” 的维度顺序，且空间维度形状与输入保持一致。
引用：
    snnTorch spikegen.rate (Poisson/binomial spike generator)
    BindsNET PoissonEncoder
    Diehl & Cook (2015), MNIST pixels to Poisson spikes

population
----------------
介绍：
    群体编码。其公式与广播语义均与 Norse 库的 PopulationEncoder（群体编码器）完全一致：采用高斯调谐曲线（Gaussian tuning curves），且曲线中心在 [low, high] 区间内均匀分布。该封装器提供了一个双参数接口encode(inputs, num_steps)：若num_steps>1，编码后的张量会沿新增的前置时间维度重复，以适配 “时间优先” 的处理流程，且不改变编码本身的内容。
引用：
    Norse docs: norse.torch.module.encode.PopulationEncoder https://norse.github.io/norse/generated/norse.torch.module.encode.PopulationEncoder.html
    Norse docs: norse.torch.functional.encode (population encoding) https://norse.github.io/norse/auto_api/norse.torch.functional.encode.html

rank_order
----------------
介绍：
    排序编码。对于每个样本，沿通道维度（或最后一个维度）对数值进行排序，将排序后的等级映射到离散时间步，然后在映射得到的时间步上发射一个脉冲。
引用：
    Alan Jeffares et al., "Spike-inspired rank coding for fast and accurate recurrent neural networks", Proc. ICLR 2022.https://github.com/codingrank

temporal
----------------
介绍：
    脉冲神经网络（SNN）输入的时序 / 延迟编码机制实现
引用：
    JK Eshraghian et al., "Training Spiking Neural Networks Using Lessons From Deep Learning", Proc. IEEE'2023. https://github.com/jeshraghian/snntorch

rate
----------------
介绍：
    脉冲神经网络（SNN）输入的速率编码机制实现
引用：
    JK Eshraghian et al., "Training Spiking Neural Networks Using Lessons From Deep Learning", Proc. IEEE'2023. https://github.com/jeshraghian/snntorch

sdr
----------------
介绍：
    SDR 式同步性：在每个时间片内，激活每个位置上的前 k 个通道。该方式会跨通道生成稀疏且同步的通道组。
引用：
    Auge et al., Survey (Correlation & synchrony; SDR).

tcr_mw
----------------
介绍：
    具有形状感知路由的时间对比度移动窗口（Temporal-Contrast MW）：
        * 运动 / 声学（输入输出维度）：(W,C) -> (T,W,C)，沿 W 维度进行移动窗口均值计算
        * 视觉（输入输出维度）：(C,H,W) -> (T,C,H,W)，通过平均池化（avg-pool）计算局部均值

tcr_sf
----------------
介绍：
    具有形状感知路由的时间对比度步进滤波（Temporal-Contrast SF）：
        * 运动 / 声学（输入输出维度）：(W,C) -> (T,W,C)，沿 W 维度进行基线步进计算
        * 视觉（输入输出维度）：(C,H,W) -> (T,C,H,W)，基线 = 局部均值 + 微小泄漏项

tcr_tbr
----------------
介绍：
    具有形状感知路由的时间对比度时间基线重置（Temporal-Contrast TBR）：
        * 运动 / 声学（输入输出维度）：(W,C) -> (T,W,C)，沿 W 维度计算差值
        * 视觉（输入输出维度）：(C,H,W) -> (T,C,H,W)，计算相对于局部均值的对比度
引用：
    Auge et al., "A Survey of Encoding Techniques for Signal Processing in SNNs"