# 源码快照清单与来源说明

快照日期：2026-08-20。

十个任务的 `official_source/` 均包含以下六个从当前项目工作目录复制的生效文件：

- `implementation/__init__.py`
- `implementation/action_bank.py`
- `implementation/<task_name>.py`
- `configs/action_config.json`
- `configs/clear/gym_config.json`
- `configs/random/gym_config.json`

实现文件复制自 `RoboSynChallenge/robosynchallenge/tasks/<task_name>/`，配置复制自 `RoboSynChallenge/configs/<task_name>/`。这些是当前项目版本的快照，不能仅凭路径认定为未经修改的官方原版。任务包括：

1. `click_bell`
2. `item_assembly`
3. `mixer_operating`
4. `water_pouring`
5. `handle_basket`
6. `manipulate_pipette`
7. `table_rearrangement`
8. `drawer_open_place`
9. `sample_loading`
10. `items_handover`

快照刻意排除了 `__pycache__/`、`.pyc`、`.bak`、`before_*`、诊断配置、临时测试配置和其他历史版本。`official_source/` 仅用于保存与查阅；运行时仍以仓库原始任务包和 `configs/` 为准。

特别说明：`click_bell/official_source_backup/` 中四个文件经用户确认为真正的官方原版，并已通过 Git 对象校验；`click_bell/official_source/` 则是项目修改后的版本。其他九个任务在获得可信的官方原始来源前，也只能标记为“当前项目快照”。
