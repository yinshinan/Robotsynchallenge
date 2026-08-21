# item_assembly 失败排查与非侵入修复

## 结论

本次检查了 2026-08-18 的 5 个评估视频、任务配置、成功判定、ACT 执行循环和 2026-08-11 的随机环境调试日志。三类现象不是同一个原因：

1. **第 0 帧没有目标物体**：随机配置独立地把桌面高度改变 `±0.05 m`，但 `guijiao1/2` 仍按固定绝对高度 `z=0.8` 生成。桌面向上随机时，物体和干扰物会被埋入桌面。视频 seed `398764591`、`441365315` 的第 0 帧正好表现为目标件完全消失。
2. **看似拼接成功但系统判失败**：ACT 一次执行 10 个控制步，外层评估器只在动作块结束后检查成功。有效接触可能只持续 1 步，因此会漏检。日志还证明：第 261 步曾出现 `success=True/contact=True`，创建约束后最终检查却因接触传感器瞬时为 `False` 改判失败；原 `is_task_success()` 同时会删除已经创建的约束。
3. **拼接过程中掉落**：动作块中的短暂张爪命令不可逆地造成掉件；失败后评估继续执行到超时，又会执行示教尾部的松爪动作。一次性的第 261 步约束检查和成功漏检进一步放大了掉落。

## 新增的非侵入修复

`eval_item_assembly_fixed.py` 只在运行时包装环境实例和评估器，不修改任何原任务、配置或 ACT 文件：

- 环境创建前只在内存配置副本中禁用不安全的桌高事件，原 JSON 不变；之所以不能在 reset 后再搬动物体，是因为原事件已经先执行 100 个物理稳定步，此时物体已经掉到地面；
- 在环境创建后保存两件目标物的干净参考位姿；第一次 reset 保持原样，仅在同进程第二轮以后于原 reset 前恢复，修复相对位姿事件从上一轮最终/落地位姿继续累加的问题；
- 严格成功一旦出现就在本 episode 内锁存；
- 同时检测向量环境自动 reset 导致的步数回退，清空上一 episode 的成功与夹爪锁存，防止跨 episode 假阳性；
- 成功查询不再删除已创建的拼接约束；
- 把 ACT 的成功检查频率从每 10 步提高到每 1 步；
- 每个夹爪只有经历“张开 -> 闭合”后才锁定闭合，避免初始闭合状态被误锁，也避免拼接中途误张开；
- 可选的最终阶段闭环纠偏在第 258–280 步、双夹爪已锁定且目标仍在工作区时，逐步消除两管的轴向角度和横向偏差，并依据缩放后网格长度把中心距收敛到约 0.195 m 以形成约 7 mm 的物理插入接触，同时允许原一次性约束检查重试；严格的角度、20 mm 横向偏差和接触阈值保持不变；
- 专用配置把单次评估上限限制为 361 步，避免失败轨迹无意义运行到 1000 步。

注意：`item_assembly_alignment_assist` 使用仿真物体真值位姿，适合诊断、稳定数据生成或规则允许特权状态的评估。如果正式比赛规定策略只能使用相机/机器人观测，应关闭该选项，并用失败 seed 做数据增强和模型微调。

## 验证结果

- 7 个非侵入逻辑测试全部通过，新增脚本也通过 Python 语法检查。
- 2026-08-21 最终采用“每 seed 独立进程”复评 5 轮，严格成功 `4/5`：`209652396`、`924231285`、`1478610112`、`441365315` 成功，`398764591` 失败。完整机器可读结果见 `eval_result/item_assembly_isolated/2026-08-21_14-56-58/manifest.json`。
- `398764591` 已不再发生物体消失：reset 后目标高度为 `0.842/0.846 m`；其真实失败点是第 261 步横向偏差 `0.02241 m`，比严格阈值 `0.020 m` 多约 `2.4 mm`。该 seed 曾在单独回归中第 264 步成功，说明系统修复有效但策略对临界随机状态仍不稳定，不能把这类真实对准失败伪装成成功。
- `924231285`（原拼接/掉落类）在最终隔离复评中成功。
- 20 条随机评估最初为 `13/20`，7 条失败均属于最终错位。加入晚触发闭环纠偏和轴向接触推进后，对这 7 个原失败 seed 做最终隔离验收，结果 `7/7` 全部严格成功；manifest 为 `eval_result/item_assembly_isolated/2026-08-21_15-31-29/manifest.json`。
- 最终执行 `git diff --exit-code` 校验，原任务实现、原随机配置、原 ACT 执行器及其原配置均无改动。

## 使用方式

从 `RoboSynChallenge` 目录运行：

推荐使用每个 seed 独立进程的批量入口，彻底隔离环境、机器人和 ACT 状态：

```bash
python policy/act/item_assembly/run_item_assembly_isolated.py
```

下面的入口适合单个 seed 调试；不建议用它在同一个仿真进程内连续跑多轮：

```bash
python policy/act/item_assembly/eval_item_assembly_fixed.py \
  --config policy/act/item_assembly/deploy_policy_fixed.yml \
  --overrides \
  --checkpoint_path outputs/act_item_assembly_official1000_custom_v0_bs8_v0/checkpoints/080000/pretrained_model \
  --max_episodes 1
```

固定复现某个失败 seed：

```bash
python policy/act/item_assembly/eval_item_assembly_fixed.py \
  --config policy/act/item_assembly/deploy_policy_fixed.yml \
  --overrides \
  --checkpoint_path outputs/act_item_assembly_official1000_custom_v0_bs8_v0/checkpoints/080000/pretrained_model \
  --eval_fixed_episode_seed 398764591 \
  --max_episodes 1
```

删除本目录新增的 `eval_item_assembly_fixed.py`、`run_item_assembly_isolated.py`、`deploy_policy_fixed.yml`、测试和本报告即可撤销该方案，不会影响原代码。
