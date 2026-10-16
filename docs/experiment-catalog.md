# 实验数据库（第一阶段）

本阶段使用 Python 标准库 SQLite，不需要安装数据库服务或额外依赖。

## 存储边界

- 数据库：方案、配置快照版本、运行记录、逐轮指标、客户端/检测/聚合明细、文件索引、批次记录。
- 原目录：CSV、原始日志、图片、模型与配置快照。数据库仅记录文件位置、大小、修改时间和 SHA-256，不存储这些大文件本体。
- 默认数据库位置为项目目录下 `platform_state/catalog.sqlite3`。可在启动后端前设置 `FL_BACKDOOR_DATABASE` 指定其他路径。
- 数据库及 WAL/SHM 文件已被 Git 忽略，不要提交数据库或实验数据。

第一阶段仍以实验文件为导入源，数据库是统一的查询副本：训练端不同时写两套指标；也不提供独立修改数据库指标的接口。后端启动、历史列表/详情查询以及每 5 秒后台轮询都会增量导入。变化期间读入失败会保留上一次完整快照，下一轮重试，不会提交半份实验。没有变化的文件不会重新解析或计算校验值。

## 旧数据导入

以下命令从**项目根目录**在 PowerShell 中执行：

```powershell
& '.\fl_env\Scripts\python.exe' -B '.\scripts\catalog.py' import
& '.\fl_env\Scripts\python.exe' -B '.\scripts\catalog.py' status
& '.\fl_env\Scripts\python.exe' -B '.\scripts\catalog.py' list --limit 20
```

也可指定其他结果目录与数据库：

```powershell
& '.\fl_env\Scripts\python.exe' -B '.\scripts\catalog.py' --database '.\platform_state\catalog.sqlite3' import --results-dir '.\results'
```

重复导入不会增加重复记录。已有运行 UUID 原样保留；没有 UUID 的旧实验保留空值，另有稳定的 `catalog_id` 作为数据库主键，不回写 `run.json`，也不改变原来的删除/监控校验含义。

一个损坏 CSV 或批次文件不会中止其他记录导入，导入报告会列出警告。日志独立存放，不会解释或执行日志文本。模型只读取字节校验，不会反序列化或执行。

## API

- `GET /api/experiments` 保持原有数组响应兼容旧页面。
- 可添加 `search`（名称包含）、`status`、`dataset`、`limit`（1～200）、`offset`。
- 添加 `paginated=true` 返回 `{items, total, limit, offset}`。
- `GET /api/experiments/{name}` 从数据库读取详情，原目录不存在也可查看名称、UUID、配置快照和已入库曲线。
- `GET /api/catalog/status` 返回数据库版本、完整性、表记录数量和最近同步报告。
- `POST /api/catalog/sync` 强制重新读取原文件和计算校验值。

结果根目录参与记录身份，不会把两个目录下的同名实验混在一起。数据库不代替当前任务调度，也不会在后端重启后自动恢复训练。批次表保存导入时的持久记录，不把旧运行状态误当作活动任务。

主动删除实验仍需要原有的运行/清理/报告占用校验，通过后删除剩余文件及对应数据库记录；即使附件已丢失，也可以主动删除数据库记录。外部手动移除目录或文件不会隐藏或删除已入库记录，也不会清空已有配置、曲线和明细。同一运行的增量导入会保留此前已保存的数据。

列表和详情返回 `available_files`、`missing_files`、`files_available` 与 `attachments_missing`，将附件状态与实验数据分开。文件缺失时显示中文警告，禁用相应下载、日志和原始 CSV 绘图入口；已入库曲线仍直接绘制。当前版本不从数据库重新创建原始文件或 PNG，重新生成图片仍需要原始 CSV。旧版数据库中已被标记为缺失的记录也能重新显示，无需恢复附件或重新导入。若同一路径被其他 UUID 的运行替换，只展示该位置最新运行，不将其附件归给旧运行。

## 备份与恢复

```powershell
& '.\fl_env\Scripts\python.exe' -B '.\scripts\catalog.py' backup '.\platform_state\catalog-backup.sqlite3'
& '.\fl_env\Scripts\python.exe' -B '.\scripts\catalog.py' restore '.\platform_state\catalog-backup.sqlite3' '.\platform_state\catalog-restored.sqlite3'
```

备份通过 SQLite 在线备份接口生成并校验，不直接复制正在使用的数据库文件。备份目标不能已存在。恢复只允许新文件，绝不覆盖当前后端正在使用的数据库；校验应用标识、模式版本和完整性后才恢复。

切换到恢复文件：停止后端，在新终端中设置环境变量后按原启动方式运行：

```powershell
$env:FL_BACKDOOR_DATABASE = "$PWD\platform_state\catalog-restored.sqlite3"
& '.\fl_env\Scripts\python.exe' -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

**数据库备份不包含原日志、图片或模型。**完整备份需同时保留结果目录。若仅恢复数据库而没有文件，不能凭记录还原原图片或模型。若文件路径改变，在新位置重新导入，旧位置的记录仍保留但查询按所选根目录隔离。

## 当前边界

- 目前只提供默认本地工作空间，没有用户认证、资源授权或跨机器调度；不要因此将本机服务直接暴露到公网。
- 数据库模式有版本与事务化初始化，拒绝修改其他应用数据库或更新版本的数据库；未来模式变更仍需增加逐版本迁移。
- 文件与 SQLite 之间不是跨系统原子事务，采用幂等同步和下次重试恢复一致性。文件导出仍以原目录为准。
- 数据库不可用时返回中文 503，不偷偷切回文件扫描造成两种不一致结果。