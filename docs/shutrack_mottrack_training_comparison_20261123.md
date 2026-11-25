# MoTTrack 与 SHUTrack 运动训练机制比较（2026-11-23）

本轮只读核对了 mamba_kalman_filter 项目源码及已有实验产物；当前训练和模型实现保持原样。以下是代码事实、已有结果与后续建议的区分。

## 已有 SportsMOT train-only 结果

MoTTrack：
- automation_results/baseline_trainonly_20e/result.json：seed42，最佳 e12，val HOTA 83.8391。
- automation_results/component_trainonly_multiseed_12e/summary_mean_std.csv：Full 三种子 HOTA 83.69887 ± 0.14040；各种子在 e10/11/12 中按 val 最佳选点。
- 相同消融清单的 Fixed KF 为81.4127；run_component_trainonly_multiseed_12e.py:157-175 明确 fixed 分支使用 --kf-type kf。
- 因此其相对 Fixed KF 的增益对应普通 KF 参考，SHUTrack 使用 NSA KF。两个项目的缓存、运行时和选点协议尚未逐项完全对齐。
- 当前 SHUTrack A_e20 为83.66593。以上绝对成绩接近；这些记录不能把架构/训练差异单独归因为某项收益。

## 1. 学习的量不同

MoTTrack mamba_kalmanfilter.py：
- 363-415：Q 网络输出8维过程噪声对角项，R 网络输出4维观测噪声对角项。
- 336-358：输出经 softplus 得到正值。
- 603-644：Q 进入 P_pred=F P F^T+Q。
- 646-708：R 进入 S=H P H^T+R，再计算 Kalman gain 和更新状态/协方差。
- F 为固定常速度模型；Q 通过后续增益间接影响均值，不能把它表述为直接学习了加速度或运动转移矩阵。
- 可按不同状态维度改变增益；SHUTrack 原设计以一个 [0,1] gate 缩放已有 NSA 更新。

## 2. 训练会递推模型自己的状态

train.py:115-253：
- 每批是轨迹片段；dataset.py:304-395 将长轨迹切成至多200个GT记录的片段。
- t循环先GMC，再predict/update；mean/cov由模型结果继续传到下一步（217-226），该处没有detach。
- 后续框误差可以沿状态链反传到此前决策；历史Q/R输入来自模型当前KF估计。
- HISTORY_WINDOW_SIZE=16限制网络读取的历史窗口，不等于损失只覆盖16步。
- 固定GT身份轨迹训练不包含完整在线离散关联优化；训练/推理仍有差别。

SHUTrack 原D：
- motion_box_diou_loss在固定采集轨迹的g0/g1下一帧端点间插值，计算t+1 GT DIoU。
- 当前训练gate不会在batch内重建后续轨迹的KF状态、协方差和未来输入。
- 该训练没有直接覆盖gate改变协方差后，对下一次及更后面KF增益的影响。

## 3. 数据在每个epoch有动态扰动

dataset.py:587-652：
- seed+epoch+idx确定样本噪声，每个epoch重新生成。
- GT框上加入位置/尺寸高斯噪声；模拟漏检及连续漏检，之后按图像尺寸归一化。
- 上述SportsMOT严格train-only参考使用位置/尺寸噪声1像素、基础漏检概率0.004，burst=0；不能把实现支持的所有增强都写成实际开启。
- 增强的是观测误差/缺失模式，GT运动轨迹本身没有因此产生新的运动模式。

## 4. 实际损失与消融

trainer/loss_utils.py:96-188：
- 预测状态损失 = DIoU + 500 * 四维速度L1（坐标按图像尺寸归一化）。
- 实际SportsMOT参考配置 prediction=1、update=0，另含权重0.03的IMC/whitening；不能称其对当前帧更新框使用了非零直接位置监督。
- 即使update位置损失为0，R和更新仍可从后续预测损失获得梯度。
- 严格三种子 Full - no_IMC 的HOTA差分别0.0259/0.0003/0.0086，均值约0.0116 pp；本组记录里主要提升不能归给IMC。
- 速度监督在SHUTrack历史实验中已探索过，尚无稳定净收益，暂不照抄其数值权重。

## 保持现有架构的候选方向

优先：短序列递推训练（初步考虑5帧），同一query_film_bilinear输出双gate，按既有NSA/EMA规则执行，在模型产生的连续状态上监督当前有效输出及后续预测。原外观rank/column保持。该建议尚未实施/验证，不能从MoTTrack的整体成绩推出必然收益。

先核对连续窗口、完整初始KF状态/协方差、检测/GMC及历史输入可用性。现有g0/g1端点不足以精确替代多次KF更新；可复用检测/ReID缓存，可能需要组织连续事件或补少量状态回放。改动会增加训练计算量，先测真实batch时间和内存，再确定窗口与预算。

后续：一致的数据增强。若扰动观测，应同步重算对应KF状态、证据、端点及监督，保留ReID/几何匹配语义；避免仅给13维证据随意加噪却让标签保持原样。先利用真实检测残差标定扰动范围。

当前D/E续训与此研究独立。后续同类实验仅在最终checkpoint做完整val，按用户要求安排。
