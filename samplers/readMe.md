efficacy_guided_sampler.py
    按有效学习缺口采样：
    P(c) ∝ (1 - psi_c)^alpha

class_frequency_sampler.py
    按类别频率采样：
    P(c) ∝ 1 / freq_c^alpha

balanced_sampler.py
    按类别均匀采样：
    P(c) = 1 / |C|