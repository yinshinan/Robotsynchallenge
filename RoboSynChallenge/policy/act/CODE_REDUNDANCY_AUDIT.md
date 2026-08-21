# 代码重复与冗余评估

评估日期：2026-08-20。此次只评估，不删除文件。

## 可直接清理

- `--connect-timeout`、`--proxy`：两个零字节文件，不是代码，名称表明是命令参数被误创建成文件。
- `.venv/` 之外的 `__pycache__/` 与 `.pyc`：共 38 个、约 553 KiB，均可由 Python 重新生成；其中根目录缓存还包含已经不存在的旧脚本。
- `deploy_policy.py.before_gripper_obs_fix`、`deploy_policy.py.最佳版本`：与当前 `deploy_policy.py` 字节级完全相同，保留当前文件即可。
- `deploy_policy.py.bak`：与 `baseline_before_new_eval_20260814_135056/deploy_policy.py` 字节级完全相同；两者最多保留一个历史副本。

## 建议合并后再清理

- `click_bell/debug_teacher_forced_act.py`、`debug_teacher_chunk1.py`、`debug_teacher_chunk2.py`：主体代码相同，只有预测动作选择分别为队列动作、chunk 索引 1、chunk 索引 2。建议给主脚本增加 `--chunk-index` 参数后删除两个派生脚本。
- 根目录的多个 `deploy_policy.py.before_*`、`deploy_policy.py.metrics.bak` 和 `debug_backups/`：除上述完全重复项外内容确有差异，但属于历史实验版本。若变更已经进入 Git 历史，可统一移入单一 `archive/` 或删除，避免被误当运行入口。
- `baseline_before_new_eval_20260814_135056/`：是完整基线快照而非工作代码，其中部署脚本另有重复。若确实需要可复现实验，应整体压缩归档；否则 Git 标签/提交比散落副本更清晰。

## 不建议删除

- 十个 `official_source/`：它们与当前项目任务实现有意重复，是按任务保存的当前工作版本快照，不作为运行入口；是否属于未经修改的官方原版需要另行核验。
- 各任务目录自己的 `__init__.py`：内容很小，作用是保持明确的 Python 包边界。
- `drawer_open_place/` 的四个专项脚本：分别承担基础加权训练、严格完成微调、隔离评测和连续评测，职责不同。
- `sample_loading/` 的两个脚本：一个运行诊断矩阵，一个分析结果，职责不同。

## 更正与恢复记录

- `click_bell/official_source_backup/`：此前被误判为可由当前版本取代，现已于 2026-08-20 从 Git 完整恢复。该目录经用户确认为真正的官方原版，必须保留。`click_bell/official_source/` 是修改后版本；两者功能差异不能作为删除官方原版的依据。

## 上游目录中的历史文件

官方源目录本身共有 33 个 `.bak`、`before_*` 等历史文件，其中 `item_assembly` 13 个、`handle_basket` 7 个最集中。这些文件没有复制进 `official_source/`。它们也建议后续独立归档，但不属于本次 `policy/act` 目录整理范围。
