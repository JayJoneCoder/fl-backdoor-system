# Windows 首次配置、启动器与发行适配

已完成**可迁移私有 Python 环境**与 **Inno Setup 安装器**的构建脚本，详见 [`packaging/README.md`](../packaging/README.md)。目标发行平台为 Windows x64 / CPython 3.11；不要把当前 `fl_env` 直接复制给评委（它含开发机绝对路径）。

## 评委使用流程

1. 安装后从开始菜单启动 `pythonw.exe -B "{app}\desktop\start_gui.py"`（无控制台窗口，日志写入 `%LOCALAPPDATA%\LianAnZhiDun\logs\launcher.log`）。
2. 首次自动打开本地浏览器配置窗口，不要求 Node.js、系统 Python 或 PyTorch 已存在。
3. 选择下载源（上海交大 / 思源 / 阿里云 / PyTorch 官方 / 官方 CDN），可先点“测试各源速度”：页面会显示延迟、带宽并标出推荐源。
4. 只下载 `torch==2.8.0+<build>`、`torchvision==0.23.0+<build>`。其他 95 个 Python 依赖随发行包提供；安装命令固定 `--no-deps`。
5. 显示安装日志；失败可更换下载源重试。精确版本的 Windows wheel 不在索引中时直接解释失败，不用其他构建代替。
6. CPU 和 GPU 的卷积、反向传播、NumPy 桥接、torchvision 运算通过后才激活新环境。
7. GPU 不可用时显示警告及解决指引，CPU 运行需主动确认；强制 GPU 不可用则阻止。
8. 使用内置前端，浏览器打开平台。启动器管理单实例、命名互斥量（供安装器检测）、健康检查及所属进程清理。关闭浏览器不会自动停止后台；关闭启动器会结束所属实验进程。
9. 若配置窗口被直接关闭，180 秒无轮询后启动器自动取消并以原生弹窗提示，不会永久挂起。
10. **默认使用本地执行后端（不启动 Ray）**：客户端在仿真进程内顺序执行，结果可复现；如需 Ray，设置 `FL_BACKDOOR_SIM_BACKEND=ray`，届时会先做 Ray 自检（成功缓存 12 小时、失败 5 分钟）。

## 仿真后端：本地（默认）/ Ray（可选）

Flower 1.29 的模拟引擎只带 Ray 后端，而 Ray 在 Windows 上是实验性支持。桌面包通过 `fl_backdoor.simulation.entry` 覆盖 `flwr-simulation` 入口，注册 `LocalBackend` 并把 `backend="local"` 传给 Flower：

- 不启动 GCS/raylet，不依赖 Ray 的原生进程；
- 客户端顺序执行，调用前后保存/恢复全局 RNG，避免影响服务端的客户端选择；
- 服务端回复按 `node_id` 排序后再做检测/聚合，Ray 与本地后端结果逐位一致；
- 仍保留 Ray 作为可选后端（`FL_BACKDOOR_SIM_BACKEND=ray`），仅用于对比或排查。

## Windows Ray 故障排查（仅在显式选择 Ray 时）

`Exit Code: 700` 是 Flower 的 `SIMULATION_EXCEPTION`；随后常见 `finished:failed No heartbeat received from the run`，两者是同一事件：raylet/GCS 未启动或中途退出。

本项目已做的加固：

- `RAY_TEMP_DIR` 固定到 ASCII 路径 `C:\Users\Public\LianAnZhiDun\ray`（Ray 无法处理中文路径）；
- 放宽 Ray 健康检查超时（`RAY_health_check_timeout_ms=120000` 等），适配慢机器/杀毒软件扫描；
- 关闭 Ray 用量统计、调整加速器环境变量告警；
- 启动前自检（`fl_backdoor/utils/ray_probe.py`），报告写入 `%LOCALAPPDATA%\LianAnZhiDun\platform_state\ray-probe.json` 与实验目录 `ray_runtime.json`；
- 失败时保留 `superlink.log` 与 `ray-logs/`（raylet/gcs/dashboard 日志）到实验目录，供排查。

仍失败时依次检查：杀毒软件白名单（安装目录 + `%LOCALAPPDATA%\LianAnZhiDun` + `C:\Users\Public\LianAnZhiDun`）、VC++ 2015-2022 x64 运行库、磁盘空间、端口占用；最后可换机器或在 WSL2 中运行。


首次下载需联网；成功后训练默认离线，数据集随包只读。关闭平台后用 `--setup` 重新配置；`--diagnose` 为只读检查。

## 开发机验证入口

```powershell
& 'D:\GithubProjects\fl-backdoor-system\fl_env\Scripts\python.exe' -B -m desktop.launcher --diagnose
& 'D:\GithubProjects\fl-backdoor-system\fl_env\Scripts\python.exe' -B -m desktop.launcher --setup
& 'D:\GithubProjects\fl-backdoor-system\fl_env\Scripts\python.exe' -B -m desktop.dependencies
& 'D:\GithubProjects\fl-backdoor-system\fl_env\Scripts\python.exe' -B -m desktop.dependencies --allow-missing-external
```

`--allow-missing-external` 用于校验“已随包但故意不安装 torch/torchvision”的发行环境；torch 的 Windows 运行时依赖由内置清单继续遍历。`--setup` 会创建当前用户目录、配置并打开浏览器；只有点击安装才下载。

## 发行资源契约

安装目录应包含：

- 可迁移的真实 Python 3.11 解释器、pip、除 torch/torchvision 外的依赖闭包（包括 NumPy/Pillow/SymPy 等 ML 传递依赖）。不能只复制 venv，不能将完整训练解释器冻结成 EXE。bootstrap 可以以后用轻量原生/冻结启动器调用真实解释器。
- `backend`、`desktop`（含 `setup.html`）、`fl_backdoor`、`scripts`、`pyproject.toml`、README 和许可证。
- 已构建 `frontend/dist`，不发行 Node/Vite。
- `data/MNIST/raw` 的未压缩原始文件、`data/cifar-10-batches-py` 的完整文件。离线训练使用 `download=False`。
- 需要演示的模板/批量 JSON 应另行整理为无私人信息的资源；本次未自动导入桌面上的参赛材料。
- **不包含**开发账户数据库、历史结果、开发配置备份、editable 路径、开发环境 console EXE。

在验证好的构建环境中执行 `-m desktop.dependencies --output <路径>` 可导出锁定版本清单（当前 95 个包）。实际构建由 [`packaging/build_portable.py`](../packaging/build_portable.py) 用同平台 wheelhouse 离线安装，并由 `verify_runtime.py` 复验闭包、版本和迁移性，不要手动删除依赖目录。

Flower 自身会继续使用 console entry point，因此每次实验在临时目录根据**实际私有解释器**重新生成 Flower 入口，避免构建机内嵌的绝对路径。发行包中的 `Scripts\*.exe` 与应用 `.pyc` 都会在构建时清理；运行时设置 `PYTHONDONTWRITEBYTECODE=1`，不在安装目录生成 `__pycache__`。

## 用户目录与覆盖

默认桌面状态位于 `%LOCALAPPDATA%\LianAnZhiDun`：

- `pyproject.toml`：首次从安装资源初始化，之后不覆盖。
- `platform_state`：身份数据库、目录索引、用户配置、运行环境激活记录、单实例锁。
- `runtimes/torch-<build>-<uuid>`：首次下载的 ML 包，独立 staging 通过检测后原子切换；失败不损坏已有环境。旧成功环境暂时保留，可在关闭平台后手动删除。
- `results`、`logs/launcher.log`、`logs/backend.log`、`logs/runtime-install.log`、`cache`、`tmp`。
- 首次下载的 wheel 缓存位于 `cache\pip`，安装成功后可以清理。

路径覆盖变量：`FL_BACKDOOR_RESOURCE_DIR`、`FL_BACKDOOR_USER_DIR`、`FL_BACKDOOR_CONFIG`、`FL_BACKDOOR_DATA_DIR`、`FL_BACKDOOR_RESULTS_DIR`、`FL_IDENTITY_DATABASE`、`FL_BACKDOOR_DATABASE`。桌面开关 `FL_BACKDOOR_DESKTOP=1` 由启动器设置。

`runtime-device` 可配置 `auto/cpu/cuda`。auto 遵循启动器策略；实验覆盖可显式选择其他设备。API 开始实验前校验，worker 在启动 Flower 前再次做实际计算检查，服务端与客户端统一决策。客户端请求 1 CPU；CPU 最多 2 个并行 actor，GPU 每 actor 请求 1 GPU 且只暴露 1 张卡，避免默认 0 GPU 导致客户端不可见显卡。CPU 线程固定为 1。实际探测和最终选择写入实验 `device_runtime.json`、`run.json` 与日志；不支持训练中静默回退。

## 验收状态

- 单元测试模拟缺包、CPU wheel、有驱动但不可用、架构/内核不兼容、GPU 成功、安装失败回滚、精确版本与镜像索引检查。
- 本地 HTTP 测试覆盖配置窗口 token/来源校验，以及真实 API 端口绑定、健康检查、静态前端、身份保护和所属进程清理。
- 已实测：`build/stage` 可整体改名后运行；安装器静默安装/卸载；`pythonw -B` 无控制台启动并写日志；SJTU 源真实下载 CPU wheel 后完成 1 轮 2 客户端 MNIST 训练（ACC 0.8887）。
- **尚需真实干净 Windows 主机**（未装 Python/Node/PyTorch）验收：断网启动、CPU/CUDA 两种 wheel 全量下载、GPU 调度完整训练、安装/卸载与 SmartScreen 表现。详见 [`packaging/README.md`](../packaging/README.md#干净机验收清单尚未执行)。
