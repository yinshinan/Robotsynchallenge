# ACT 代码目录

本目录按 RoboSynChallenge 的十个正式任务组织。任务专用的训练、评测、诊断脚本放在同名任务目录中；每个任务的 `official_source/` 保存当前官方任务实现和标准配置快照。根目录中的 `deploy_policy.py`、`deploy_policy.yml`、`eval.sh`、`finetune.sh`、`scripts/` 和 `pyproject.toml` 是所有任务共用的 ACT 基础代码。

| 任务目录 | 当前专项内容 |
| --- | --- |
| `click_bell/` | 按铃任务诊断脚本、训练说明和官方源代码备份 |
| `item_assembly/` | 暂无任务专用脚本 |
| `mixer_operating/` | 暂无任务专用脚本 |
| `water_pouring/` | 暂无任务专用脚本 |
| `handle_basket/` | 暂无任务专用脚本 |
| `manipulate_pipette/` | 暂无任务专用脚本 |
| `table_rearrangement/` | 暂无任务专用脚本 |
| `drawer_open_place/` | 抽屉任务专项训练与严格评测脚本 |
| `sample_loading/` | 上样任务诊断矩阵及结果分析脚本 |
| `items_handover/` | 暂无任务专用脚本 |

没有任务专用 ACT 脚本的目录仍保留官方源码快照，后续新增代码时可直接按任务归档。快照范围见 `OFFICIAL_SOURCE_MANIFEST.md`，重复与冗余评估见 `CODE_REDUNDANCY_AUDIT.md`。
