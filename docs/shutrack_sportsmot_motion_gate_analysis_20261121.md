# SportsMOT 运动 gate 接近 1：代码与现有训练数据诊断

日期：2026-11-21

## 结论

主因有直接证据：当前 SportsMOT 一步 GT 框 DIoU 目标强烈偏向完整写入；
模型很早就收敛到接近恒开的策略，sigmoid 饱和又削弱了后续逐事件区分。
这不是“所有事件都应该完整写入”的证明：相当一部分事件仍有抑制收益。

本轮只读已经完成的训练分片中的小型端点框、GT 框和 mask，未加载 ReID 特征，
未做新回放、模型前向或训练，未修改正在运行的采集进程与配置。
SportsMOT 诊断用时 1.96 秒，进程峰值 RSS 41.45 MiB；
MOT20 对照用时 4.03 秒，峰值 RSS 225.87 MiB。
此耗时仅是小数组分析耗时，不是采集测速。

## 1. 当前到底训练什么

tools/train_supervision_groups.py:978-998：
appearance_row_column / appearance_rank 的运动损失均只有 motion_box_diou_loss。
D 在外观路额外加入 column，未加入运动速度或 gap3 损失。
group 分支决定实际使用什么；配置里存在 speed_weight/gap3_weight，
不代表 D 启用了它们。

shutrack/training/loss.py:231-267：
- g = sigmoid(motion_logit)
- b(g) = b0 + g * (b1 - b0)
- L = mean_valid[1 - DIoU(b(g), GT_t+1)]

b0、b1 是同一写前状态分别不写入/完整写入，再经过 GMC+KF predict 的 t+1 框。
没有“运动 gate 应当是多少”的 soft 标签参与这个 D 运动分支。

采集时 full_write 只规定基准回放轨迹，并不是把训练目标设成 1。
训练仍使用两端点和 GT；from-scratch 分支创建独立新模型。
运行时禁用 gate 的开关来自 CLI，标准评估没有传入这些禁用开关；
评估日志也存在小于 1 的运动门。训练日志自身已经接近 1，故不是
仅在推理时被强制覆盖产生的现象。

## 2. 现有训练数据上的直接证据

使用每个配置中的训练序列名单，不读当前正在补采的 MOT20-05：
- SportsMOT：45 条，305631 样本，300457 个 motion_gt_valid。
- MOT20：02/03 两条，452937 样本，431335 个 motion_gt_valid。

float64 NumPy 实现与当前 DIoU 相同公式；
在 0:0.05:1 及 0.99/0.999/0.9999 扫描，按有效样本数加权。

| 诊断 | SportsMOT | MOT20 |
| --- | ---: | ---: |
| 固定 g=0 的平均 DIoU loss | 0.202764 | 0.156539 |
| 固定 g=0.5 | 0.164360 | 0.152737 |
| 固定 g=0.9 | 0.139687 | 0.152076 |
| 固定 g=1 | 0.135220 | 0.152255 |
| 扫描中最好的固定 gate | 1.00 | 0.80 |
| g=1 比 g=0 更好的样本占比 | 86.38% | 56.96% |
| g=0 比 g=1 更好的样本占比 | 13.62% | 43.04% |
| g=1 与逐事件网格最优相当（容差1e-6） | 71.10% | 47.23% |
| 逐事件网格最优 gate 均值 | 0.8408 | 0.5657 |
| 网格最优 gate <=0.5 的样本占比 | 15.14% | 43.75% |
| GT 逐事件网格最优 loss | 0.127231 | 0.143314 |

关键区别：SportsMOT 的整体损失曲线在扫描点上一直随 g 增大而降低。
MOT20 接近 g=1 时，继续增大 g 反而增加平均损失。

因此，在 SportsMOT 上，学一个接近恒开的简单策略，就能取得相当好的当前目标值。
“更相信最新检测有利”由现有端点数据支持；它是否具体由运动速度、
KF 噪声设置、检测精度或 GT 框抖动造成，本轮没有分别归因。

## 3. 优化信号确实不均衡

统一在 g=0.5 处，用中心差分 h=1e-5 估计当前目标导数：
- SportsMOT：85.25% 样本推动增大 gate，14.75% 推动减小。
- 推动增大的梯度绝对量约为推动减小的 11.4 倍。
- 在 g=0.99 附近，平均导数仍为负，即继续推动开门。
- MOT20：g=0.99 附近平均导数为正，即存在把门拉回来的整体信号。

这些是同一个固定 gate 位置下的数据目标导数，不是实际网络参数梯度。

## 4. 饱和发生得很早

新 SportsMOT 模型的逐分片轮次日志：
- epoch 1：运动 gate 均值 0.885268
- epoch 2：0.999312
- epoch 4：0.999839
- epoch 20：0.999962

旧架构同样快速趋向 1：前四轮 0.7201 / 0.9761 / 0.9902 / 0.9957。
因此不能把恒开现象归咎于新的双线性头。

g=sigmoid(z) 的导数为 g*(1-g)。在 g=0.9999 时为约0.0001，
而 g=0.5 时为0.25。相同 dL/dg 下，传到 logit 的梯度约小2500倍。
结合多数样本持续推动开门，后续更难学出少数应当抑制的事件。
这解释了为何加长训练或增大容量没有自然解决该现象，但未证明
任何特定训练改动一定有效。

## 5. 恒开仍不是逐事件最优

SportsMOT：
- 约28.9%的有效样本，减小 gate 能改善当前一步 DIoU（网格诊断）。
- 约15.1%的样本网格最优 gate <=0.5。
- 约16.7%的样本，相对 g=1，逐事件网格选择可降低 loss 超过0.01。
- 全体 loss 从固定g=1的0.135220降到逐事件网格最优0.127231，
  约有5.9%的相对下降空间。

这需要 GT 才知道，不能作为可部署结果，也不能换算成 HOTA 的提升上限。
是否能从现有在线输入识别这些事件，以及是否稳定改善关联，还未验证。

## 6. 当前监督的范围限制

一步 DIoU 不直接优化：
- 后续多帧的状态误差；
- KF 协方差经多次量测更新的长期影响；
- effective_box/history/corner velocity 等写回字段对后续关联的作用；
- 身份互换是否发生或多久恢复。

旧 rollout 运动收益可用未来5帧生成，但 D 当前不使用这个 motion_benefit，
而只使用 t+1 GT 框。因此“改为GT监督”和“只看下一帧”是两个不同选择：
GT监督也可以设计为多帧，不能把当前限制归因于GT本身。

身份标签来自写前历史投票，至少3票、占比>=0.75才有效。
投票不可靠或无下一帧GT时，这个运动loss不产生直接监督。代码支持该机制；
它在困难交叉窗口具体遗漏了多少关键事件，本轮未量化，不能当作已经证实的主因。
运动/外观虽有独立输出头，但共享编码器；外观loss也可能间接改变运动表示，
不能由“输出通道梯度不串路”推导出两个分支完全独立。

## 7. 建议

当前继续原采集，不改协议、不停进程、不为这个诊断补采。

后续若验证改进，先使用现有训练端点识别“写入明显有害”的组，
检查模型对它们的门值和识别能力；再做同预算、同种子的单因素验证：
- 保留GT DIoU，对真正有害事件做适度采样/加权；
- 验证早期饱和是否妨碍条件决策，可考虑温度/初始化/训练课程，逐项比较。
- 只有仍与长时关联错误相关时，再研究多帧GT目标。

不要为了让门均值好看而强行把gate拉到0.5；
不要直接认定乘大运动loss系数能改变目标本身的偏好；
不在当前采集期间启动这些训练或修改采集格式。

## 复现与文件

分析入口：
tools/analyze_motion_gate_targets.py

命令：
.venv/bin/python tools/analyze_motion_gate_targets.py --config outputs/agentguard/experiments/shutrack_architecture_night_20261120/ablation/SportsMOT/groups/appearance_row_column/config.json --out outputs/agentguard/experiments/shutrack_collection_fix_20261121/motion_gate_sportsmot_analysis.json

MOT20 对照使用同一路径下 ablation/MOT20 的 config.json。

结果：
outputs/agentguard/experiments/shutrack_collection_fix_20261121/motion_gate_sportsmot_analysis.json
outputs/agentguard/experiments/shutrack_collection_fix_20261121/motion_gate_mot20_analysis.json

本报告是数据目标与日志诊断，不是新增训练、在线恢复验证或多seed显著性实验。
