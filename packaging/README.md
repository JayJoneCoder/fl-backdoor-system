# Windows 发行构建（可迁移目录 + Inno Setup 安装器）

本目录把仓库打包成 `dist\LianAnZhiDun-<版本>-win64-setup.exe`。目标平台 **Windows 10/11 x64**，内置 **CPython 3.11.17**（python-build-standalone，可整体移动）。

## 实测结果（2026-10-22）

| 指标 | 数值 |
|---|---|
| 随包依赖 | 95 个（`requirements-lock.txt`），wheels 共 152 MB |
| wheelhouse 构建 | 95 个纯 wheel，无 sdist，约 1 分钟（清华镜像） |
| `build/stage` | 28,521 文件 / **0.87 GiB**（site-packages 605 MB + data 230 MB + Python 及源码） |
| 安装器 | **303 MB**（LZMA2/max + SolidCompression） |
| 安装后占用 | 约 0.9 GiB |
| 首次下载 | CPU 版约 0.6 GB（安装后约 +1.5 GB）；CUDA 12.8 约 3.5 GB（安装后约 +7 GB） |
| 本地全链路验收 | 通过：SJTU 源下载 CPU 版 → 启动 API → 登录 → 1 轮 2 客户端 MNIST → ACC 0.8887 |

## 目录

```
packaging/
  python-runtime.json        # 解释器版本、官方地址、镜像、SHA256
  requirements-lock.txt      # 95 个 pin（由已验证的 fl_env 导出）
  build_portable.py          # 构建 build/stage（下载/解压/装依赖/拷贝/校验）
  verify_runtime.py          # 迁移性与依赖闭包校验
  acceptance.py              # 本地全链路验收（可选，真实下载 CPU wheel）
  liananzhidun.iss           # Inno Setup 6 安装器脚本
  assets/app.ico             # 安装器与快捷方式图标
  vendor/vc_redist.x64.exe   # 可选，微软 VC++ 2015-2022 运行库（不提交）
```

首次构建如需附带 VC++ 运行库（安装器会在缺失时静默安装）：

```powershell
New-Item -ItemType Directory -Force packaging\vendor | Out-Null
curl.exe -sL -o packaging\vendor\vc_redist.x64.exe https://aka.ms/vs/17/release/vc_redist.x64.exe
```

没有该文件时 `.iss` 会自动跳过运行库条目（使用 `#if FileExists`），不影响编译。

## 前置条件

- Windows x64、Node.js（仅构建前端时需要）、Inno Setup 6.7+（`winget install JRSoftware.InnoSetup`）。
- 构建机磁盘至少 5 GB 空闲（stage + wheelhouse + 安装器）。

## 构建步骤

```powershell
# 1. 导出锁定清单（仅当依赖版本变化时执行；需要可用的 fl_env）
fl_env\Scripts\python.exe -B -m desktop.dependencies --output packaging\requirements-lock.txt

# 2. 构建（首次会下载 48 MB 解释器 + 95 个 wheel + 重建前端）
#    --index-url 默认清华镜像；可改为 https://pypi.org/simple
fl_env\Scripts\python.exe -B packaging\build_portable.py --download-wheels --build-frontend

# 3. 迭代时复用已装好的私有环境（不重新下载）
fl_env\Scripts\python.exe -B packaging\build_portable.py --reuse

# 4. 编译安装器
& "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe" /DAppVersion=1.0.0 packaging\liananzhidun.iss

# 5. （可选）便携 ZIP
fl_env\Scripts\python.exe -B packaging\build_portable.py --reuse --zip
```

`build_portable.py` 会：

1. 下载 python-build-standalone（先用镜像，再回退官方，逐字节 SHA256 校验）；
2. 解压到 `build/stage/python`，断言 `python.exe` / `pythonw.exe` 存在；
3. 从 `build/wheelhouse` **离线**安装 95 个依赖（`--no-index --find-links`）；
4. 删除调试符号（`.pdb`，78 MB）和会内嵌构建路径的 `Scripts` 启动器；
5. 复制源码、`frontend/dist`、`data`（排除可再生成的 `*.gz`，数据 230 MB）；
6. 运行 `verify_runtime.py`：版本一致、关键 import、无绝对路径 `.pth`/`pyvenv.cfg`、无 console launcher、无 torch/torchvision；
7. 用私有解释器执行 `desktop.launcher --diagnose`，要求“未装 PyTorch”时给出明确提示；
8. **把 stage 整体改名后重新自检**，证明可迁移；
9. 写出 `build/build-info.json`（版本、commit、lock 哈希、文件数与体积）。

## 安装器行为

- **按用户安装**：`%LOCALAPPDATA%\Programs\LianAnZhiDun`，`PrivilegesRequired=lowest`，不弹 UAC。
- 快捷方式：`pythonw.exe -B "{app}\desktop\start_gui.py"`（无控制台窗口）。
- `AppMutex=LianAnZhiDunPlatform`：平台运行时安装/卸载会提示先关闭。
- 开始菜单附「运行环境自检」（打开控制台执行 `--diagnose`）。
- 缺少 VC++ 2015-2022 x64 时静默安装随包运行库。
- 卸载时询问是否删除 `%LOCALAPPDATA%\LianAnZhiDun`（账号、结果、已下载 torch）；默认保留。
- `[UninstallDelete]` 清理运行时生成的 `__pycache__`。

## 本地全链路验收

```powershell
# 使用 build/stage；首次会真实下载 CPU 版 PyTorch（约 0.6 GB）
fl_env\Scripts\python.exe -B packaging\acceptance.py

# 复用已下载的运行环境，只重跑训练
fl_env\Scripts\python.exe -B packaging\acceptance.py --reuse-runtime
```

验收内容：`install_runtime`（下载 + 实际卷积/反向传播/Numpy/torchvision 自检 + 原子激活）→ 启动 `desktop/api_entry.py` → `/api/health`、`/` 前端 → 默认管理员登录 → 1 轮 / 2 客户端 MNIST → 校验 `run.log`、`run.json`、`device_runtime.json`、CSV 数据行 → 结束进程树。
2026-10-22 实跑结果：`round 0 accuracy 0.1009 → round 1 accuracy 0.8887, loss 0.979`，`torch=2.8.0+cpu`，设备日志 `actual=cpu`；另用 10 客户端 + BadNets + clustering + Krum 单轮跑通。Ray 自检耗时约 3 秒，临时目录为 `C:\Users\Public\LianAnZhiDun\ray`；把该目录设为不可写可复现“自检失败”提示。

## 仿真后端与 Windows 排障

**默认不使用 Ray。** Flower 1.29 的仿真引擎只带 Ray 后端，且 Ray 官方标注 Windows 为实验性支持。桌面构建通过覆盖 `flwr-simulation` 入口注册 `LocalBackend`，并传入 `backend="local"`：

- 不启动 raylet/GCS，消除 `Exit Code: 700` / `No heartbeat` 这类启动失败；
- 客户端在仿真进程内顺序执行；调用前后恢复全局 RNG；
- 服务端按 `node_id` 排序回复后再做检测/聚合，本地与 Ray 结果逐位一致（已用 10 客户端 BadNets + clustering + Krum 实测）；
- 需要 Ray 时设置 `FL_BACKDOOR_SIM_BACKEND=ray`，仍会先做 Ray 自检并保留失败日志。

Ray 模式下的加固仍然保留（仅显式选择 Ray 时生效）：

- Ray 临时目录固定为 `C:\Users\Public\LianAnZhiDun\ray`（ASCII），必要时同时改写 `TEMP/TMP`；
- `RAY_health_check_timeout_ms=120000`、`RAY_health_check_failure_threshold=10` 等，适配慢机器与杀毒扫描；
- 实验启动前执行 `fl_backdoor/utils/ray_probe.py` 自检（成功缓存 12 小时、失败 5 分钟）；
- 失败时把 `superlink.log` 与 `ray-logs/`（raylet/gcs/dashboard）复制到实验目录，并在 `ray_runtime.json` 记录原因；
- `Exit Code: 700` / 心跳丢失会被翻译成中文排查提示，而不是只显示退出码。

现场排查顺序（Ray 模式）：杀毒软件白名单（安装目录、`%LOCALAPPDATA%\LianAnZhiDun`、`C:\Users\Public\LianAnZhiDun`）→ VC++ 2015-2022 x64 → 磁盘空间 → 端口占用 → 换机器/WSL2。

## 干净机验收清单（尚未执行）

在没装 Python / Node / PyTorch 的 Windows 10/11 x64 上：

1. 双击安装器 → 无 UAC 报错 → 安装完成勾选“立即启动”；
2. 浏览器打开配置页，能选源与 CPU/CUDA 12.6/CUDA 12.8；
3. 断网情况下启动：应能进入平台，只是无法下载 PyTorch；
4. 联网下载 CPU 版 → 自检通过 → 跑一轮训练；
5. 有 NVIDIA 显卡时下载 CUDA 12.8 → 自检通过 → 日志出现 `actual=cuda:0`；
6. 关闭浏览器不影响后台；关闭启动器后训练进程结束；
7. 卸载：安装目录清空，`%LOCALAPPDATA%\LianAnZhiDun` 按选择保留或删除。

## 已知限制

- 安装器未做代码签名，SmartScreen 会提示“未知发布者”，需选择「更多信息 → 仍要运行」。
- 国内网络访问 GitHub Releases 可能超时，脚本自动回退 `gh-proxy.com`，并以固定 SHA256 校验文件。
- 首次下载依赖外网（SJTU 镜像或 PyTorch 官方源）；失败可在配置页换源重试。
- 打包后的 `.py` 以源码形式提供（AGPL-3.0-or-later 要求）。
