# Windows 首次配置与发布适配

当前是**源码适配阶段**，尚未生成私有 Python、发行目录、EXE 或 Inno Setup 安装器。目标发行平台为 Windows x64 / CPython 3.11；不要把当前 `fl_env` 直接复制给评委。

## 评委使用流程

1. 安装发行包后，由安装器调用随包私有 Python 的 `-m desktop.launcher`（后续安装器对接）。
2. 首次自动打开本地浏览器配置窗口，不要求 Node.js、系统 Python 或 PyTorch 已存在。
3. 选择下载源：上海交通大学 PyTorch 专用镜像（第三方）或 PyTorch 官方源。选择 CPU / CUDA 12.6 / CUDA 12.8。
4. 只下载 `torch==2.8.0+<build>`、`torchvision==0.23.0+<build>`。其他 Python 依赖均应随发行包提供；安装命令固定 `--no-deps`，不让评委临时补下载。
5. 显示安装日志；失败可更换下载源重试。精确版本的 Windows wheel 不在索引中时直接解释失败，不用其他构建代替。不自动切换第三方源，不禁用 HTTPS 校验。
6. CPU 和 GPU 的卷积、反向传播、NumPy 桥接、torchvision 运算通过后才激活新环境。CPU 可用不等于 GPU 可用；CUDA Toolkit 是否安装不是决定因素。
7. GPU 不可用时显示警告及解决指引，CPU 运行需主动确认；强制 GPU 不可用则阻止。
8. 使用内置前端，浏览器打开平台。启动器管理单实例、回环监听、健康检查及所属进程清理。关闭浏览器不会自动停止后台；关闭启动器会结束所属实验进程。

首次下载需联网；成功后训练默认离线，数据集随包只读。关闭平台后用 `--setup` 重新配置；`--diagnose` 为只读检查。

## 开发机验证入口

```powershell
& 'D:\GithubProjects\fl-backdoor-system\fl_env\Scripts\python.exe' -B -m desktop.launcher --diagnose
& 'D:\GithubProjects\fl-backdoor-system\fl_env\Scripts\python.exe' -B -m desktop.launcher --setup
& 'D:\GithubProjects\fl-backdoor-system\fl_env\Scripts\python.exe' -B -m desktop.dependencies
```

`--setup` 会创建当前用户目录、配置并打开浏览器；只有点击安装才下载。正常源码后端入口不强制切换到桌面目录。

## 发行资源契约

安装目录应包含：

- 可迁移的真实 Python 3.11 解释器、pip、除 torch/torchvision 外的依赖闭包（包括 NumPy/Pillow/SymPy 等 ML 传递依赖）。不能只复制 venv，不能将完整训练解释器冻结成 EXE。bootstrap 可以以后用轻量原生/冻结启动器调用真实解释器。
- `backend`、`desktop`（含 `setup.html`）、`fl_backdoor`、`scripts`、`pyproject.toml`、README 和许可证。
- 已构建 `frontend/dist`，不发行 Node/Vite。
- `data/MNIST/raw` 的未压缩原始文件、`data/cifar-10-batches-py` 的完整文件。离线训练使用 `download=False`。
- 需要演示的模板/批量 JSON 应另行整理为无私人信息的资源；本次未自动导入桌面上的参赛材料。
- **不包含**开发账户数据库、历史结果、开发配置备份、editable 路径、开发环境 console EXE。

在验证好的构建环境中执行 `-m desktop.dependencies --output <绝对路径>` 可导出锁定版本清单。该命令仅解析本机已安装依赖，包含 ML wheel 的传递依赖、Flower simulation extra、pip；仅排除 torch/torchvision。当前环境导出的清单包含 95 个包。实际构建时须用同平台 wheelhouse 安装并复验闭包，不要通过过滤目录随意删除依赖。

Flower 自身会继续使用 console entry point，因此每次实验在临时目录根据**实际私有解释器**重新生成 Flower 入口，避免开发 venv 内嵌的绝对路径。

## 用户目录与覆盖

默认桌面状态位于 `%LOCALAPPDATA%\LianAnZhiDun`：

- `pyproject.toml`：首次从安装资源初始化，之后不覆盖。
- `platform_state`：身份数据库、目录索引、用户配置、运行环境激活记录、单实例锁。
- `runtimes/torch-<build>-<uuid>`：首次下载的 ML 包，独立 staging 通过检测后原子切换；失败不损坏已有环境。旧成功环境暂时保留，后续安装器可提供显式清理功能。
- `results`、`logs/backend.log`、`logs/runtime-install.log`、`cache`、`tmp`。

路径覆盖变量：`FL_BACKDOOR_RESOURCE_DIR`、`FL_BACKDOOR_USER_DIR`、`FL_BACKDOOR_CONFIG`、`FL_BACKDOOR_DATA_DIR`、`FL_BACKDOOR_RESULTS_DIR`、`FL_IDENTITY_DATABASE`、`FL_BACKDOOR_DATABASE`。桌面开关 `FL_BACKDOOR_DESKTOP=1` 由启动器设置。

`runtime-device` 可配置 `auto/cpu/cuda`。auto 遵循启动器策略；实验覆盖可显式选择其他设备。API 开始实验前校验，worker 在启动 Flower 前再次做实际计算检查，服务端与客户端统一决策。客户端请求 1 CPU；CPU 最多 2 个并行 actor，GPU 每 actor 请求 1 GPU 且只暴露 1 张卡，避免默认 0 GPU 导致客户端不可见显卡。CPU 线程固定为 1。实际探测和最终选择写入实验 `device_runtime.json`、`run.json` 与日志；不支持训练中静默回退。

## 本阶段验收边界

- 单元测试模拟缺包、CPU wheel、有驱动但不可用、架构/内核不兼容、GPU 成功、安装失败回滚、精确版本与镜像索引检查。
- 本地 HTTP 测试覆盖配置窗口 token/来源校验，以及真实 API 端口绑定、健康检查、静态前端、身份保护和所属进程清理。
- 不在开发机重新下载/替换数 GB ML wheel；下载测试只请求索引。
- 尚需真实干净 Windows 主机进行私有解释器迁移、两种 wheel 全量下载、GPU 调度完整联邦训练、断网启动和安装/卸载验收。首次配置的本地浏览器窗口不等于原生安装器页面。