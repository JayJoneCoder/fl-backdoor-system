---
tags: [fl, backdoor-attack, backdoor-defense, federated-learning, security, privacy]
dataset: [MNIST, CIFAR-10]
framework: [pytorch, flwr]
---

# 面向联邦学习的后门攻击与防御系统
[![License: AGPL-3.0-or-later](https://img.shields.io/badge/License-AGPL--3.0--or--later-blue.svg?logo=gnu)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0-red.svg?logo=pytorch)](https://pytorch.org/)
[![Flower](https://img.shields.io/badge/Flower-1.26+-green.svg?logo=flower)](https://flower.ai/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18+-61DAFB?logo=react)](https://react.dev/)
[![Ant Design](https://img.shields.io/badge/Ant_Design-6.0+-0170FE?logo=antdesign)](https://ant.design/)
[![Node.js](https://img.shields.io/badge/Node.js-18+-339933?logo=nodedotjs)](https://nodejs.org/)
## 项目简介

本项目实现了一个**面向联邦学习的后门攻击与防御系统**，目前支持：

- **5 种后门攻击**：BadNets、WaNet、Frequency Attack（频域，dct / fft）、Distributed Backdoor Attack（DBA）、Full Combination Backdoor Attack（FCBA）
- **3 层防御机制**：
  - 客户端防御（数据层）：Feature Filter
  - 服务端检测（更新层）：Anomaly、Cosine、Score、Clustering Detection
  - 服务端聚合（模型层）：Norm Clipping、Trimmed Mean、Krum
- **完整的评估体系**：ACC、ASR、Precision、Recall、FPR、混淆矩阵、客户端级异常分数
- **Ground Truth 追踪**：支持 random / random_once / fixed 恶意客户端选择，实现检测性能的精确评估
- **Web 可视化平台**：基于 FastAPI + React + Ant Design，支持在线配置实验、实时日志监控、历史结果浏览
- **实验数据库（目录索引）**：用本地 SQLite 保存实验元数据、配置快照、运行状态、逐轮指标和文件索引，支持搜索/筛选/分页、附件缺失提示、配置快照查看和数据库备份恢复；详细说明见 [docs/experiment-catalog.md](docs/experiment-catalog.md)
- **可复现的数据划分**：默认保持 IID 随机划分，也支持 `partition-mode = "dirichlet"` 的标签偏斜 Non-IID 划分；可用 `partition-alpha` 控制偏斜程度、`partition-seed` 独立控制划分随机性。

> 适用于联邦学习安全研究、后门攻击防御对比实验、鲁棒聚合算法验证。

## 安装依赖和运行项目

参考 [flower 官网](https://flower.ai/docs/framework/tutorial-quickstart-pytorch.html)。

### 前置准备
```bash
# 1. 克隆项目
git clone https://github.com/JayJoneCoder/fl-backdoor-system.git

cd fl-backdoor-system

# 2. 创建虚拟环境
python -m venv fl_env
source fl_env/bin/activate   # Windows 用 fl_env\Scripts\activate

# 3. 升级 pip
pip install --upgrade pip

# 4. 安装 pytorch (以 GPU 版为例，推荐 GPU)

nvidia-smi # 查看 cuda 版本

pip install torch==2.8.0+cu118 torchvision==0.23.0+cu118 \
  --index-url https://download.pytorch.org/whl/cu118  

# 此处以 cuda 11.8 为例，注意 torch 和 torchvision 的版本必须符合 pyproject.toml ，即 2.8.0 和 0.23.0

# 5. 安装项目
pip install -e .   # 如果在上一步没有安装 pytorch，这一步会安装一个 cpu 版

# 注：我不确定先安装 GPU torch 再执行这个命令会发生什么。如果想要彻底防止 GPU torch 被顶掉，可以用以下命令
# pip install -e . --no-deps
# 或者也可以先安装项目，后手动安装对应 cuda 版本的 GPU 版 torch 2.8.0 和 torchvision 0.23.0

# 安装 flower
pip install "flwr[simulation]"

# 6. 启动后端 API 服务（命令行界面可忽略）
#    在项目根目录运行，不要在 backend/ 目录里再套一层
uvicorn backend.main:app --host 127.0.0.1 --port 8000

# 我未来可能会加个 setup_env.py 一类的脚本，实现检测 cuda，安装 torch 一类的功能。
```

### 图形界面前端（React 可视化界面）

前端基于 Node.js 开发，需要提前安装 [Node.js](https://nodejs.org/zh-cn/download) (推荐 18+ 或 20+ LTS)。

```bash
# 1. 进入前端目录
cd frontend

# 2. 安装依赖（首次运行）
npm install

# 3. 启动开发服务器
npm run dev
```
前端启动后，访问 http://localhost:5173 即可使用 Web 界面。确保后端服务已启动，否则配置加载会失败。

## 命令行界面使用说明

下文绝大多数功能已经在前端图形界面实现。

#### 仅运行联邦学习实验

```bash
# 运行单次实验（日志输出到终端）
flwr run . --stream
```

> 下面的批量脚本复用后端调度器（与网页批量实验同一个后端），需要先启动后端。


### 批量实验与结果汇总
#### 批量运行多个实验
批量脚本不直接改 `pyproject.toml`，也不自己拉起 Flower，而是把 JSON 配置提交给已启动的后端执行。JSON 格式与网页上传的批量实验一致：

```bash
# 先启动后端，再运行（--config 必填）
python scripts/batch_runner.py --config my_experiments.json

# 提交成功后立即退出，批次继续由后端运行
python scripts/batch_runner.py --config my_experiments.json --no-wait
```

网页“批量实验”页面提供以下控制与监控：

- 实验列表从 1 编号，最右列显示等待、实时轮次进度、完成、失败及退出码／简明原因、取消等状态；列表与批次日志均可收起。
- 全批次进度按已结束项（包括失败与取消）统计；当前实验进度按已记录的 CSV 轮次统计，并实时展示 ACC／ASR 和 Loss 曲线。
- “仅结束当前实验”会等待该实验清理完成，再继续后续队列；该项记为取消，不记为运行失败。
- “暂停队列”不会挂起当前训练进程：当前实验结束后暂停启动后续实验，点击“继续队列”恢复。已暂停的批次仍占用执行槽，避免同时启动其他实验。
- “结束整批／重试清理”取消当前实验和所有等待项。刷新或切换页面会恢复活动批次；后端重启则按中断处理，不会自动恢复训练。
- 列表支持按状态筛选，筛选后保留原始序号；“定位当前实验”可恢复全部列表并滚动到正在运行的实验。
- 修改 JSON 后必须重新解析才能运行，“预览表格”会先解析未应用的修改，“取消编辑”恢复最近一次解析成功的内容。刷新恢复的配置也可以在批次结束后编辑。
- 批次与指标连接异常时自动重连，也可手动重试；已结束且未产生指标的实验会明确显示空数据提示。

#### 汇总试验结果

```bash
python scripts/summarize_results.py
```

该脚本会扫描 `results/` 下所有子目录，提取最后一轮的 ACC、ASR、检测指标等，生成：

```
results/summary.csv       # 所有实验的汇总表

results/summary_table.tex # LaTeX 表格代码，可直接粘贴到论文中
```

#### 提取指标说明

| 类别 | 指标 | 来源 |
|------|------|------|
| 主指标 | `accuracy`, `asr`, `loss`（最终轮） | 主 CSV 最后一行 |
| 检测性能 | `precision`, `recall`, `fpr`, `auc`, `tp`, `fp`, `fn`, `tn`（最终轮） | `*_metrics.csv`（component=detection） |
| 聚合平均指标 | `avg_malicious_removal_rate`, `avg_benign_removal_rate`, `avg_kept_malicious`, `avg_removed_malicious`, `avg_removed_benign`, `avg_kept_benign`, `avg_total_malicious`, `avg_total_benign`, `avg_selected_clients` | `*_metrics.csv` 中 component=aggregation 的跨轮平均值；若无则从 `*_clients.csv` 估算 |

### 画图

系统会将日志信息存储在 `fl-backdoor-system/results` 下，可以使用前端已有的图形化界面画图。

在命令行界面中，用此命令画图：
```bash
# 单个/多个 CSV 文件
python scripts/plot.py results/baseline.csv results/badnets.csv

# 通配符匹配
python scripts/plot.py results/*_clients.csv

# 直接指定目录（递归查找所有 .csv，推荐）
python scripts/plot.py results/
```
### 实验数据库（目录索引）

后端把已生成的实验结果同步到本地 SQLite 数据库（`platform_state/catalog.sqlite3`），数据库只存实验元数据、配置快照、指标和文件索引，日志/图片/模型/原始 CSV 仍保留在 `results/` 下。前端历史页的搜索、筛选、分页、配置快照和附件缺失提示都走这个数据库。

### IID / Non-IID 数据划分

在 `pyproject.toml` 或前端配置中设置：

```toml
# 默认：保持原有 IID 行为
partition-mode = "iid"

# 标签偏斜 Non-IID：alpha 越小，客户端标签分布通常越集中
partition-mode = "dirichlet"
partition-alpha = 0.2
# 可选；省略时使用全局 seed
partition-seed = 123
```

划分使用确定性随机数生成器，不修改进程全局 RNG。每次实验目录的 `config_snapshot.toml` 会保存 `[tool.flwr.app.partition]`，其中包含请求的模式/alpha/seed 来源，以及实际生效的 seed 和客户端数量，便于复现实验。

命令行可单独导入、查询、备份与恢复（不启动训练）：

```bash
# 增量导入旧结果（不修改原始实验文件）
python scripts/catalog.py import

# 查看版本、完整性和记录数量
python scripts/catalog.py status

# 分页查询
python scripts/catalog.py list --limit 20

# 在线备份 / 校验后恢复到新文件（绝不覆盖当前数据库）
python scripts/catalog.py backup platform_state/catalog-backup.sqlite3
python scripts/catalog.py restore platform_state/catalog-backup.sqlite3 platform_state/catalog-restored.sqlite3
```

数据库文件、WAL/SHM 与实验数据一样已被 Git 忽略，不要提交。备份只备份数据库本身，不包含日志/图片/模型，完整备份需同时保留 `results/`。详见 [docs/experiment-catalog.md](docs/experiment-catalog.md)。

## 画图结果

### 主日志

```
*_acc.png                # Accuracy，模型在主任务上的准确率。防御应保持高 ACC
*_asr.png                # Attack Success Rate，后门攻击成功率。防御应使 ASR 趋近 0
*_acc_vs_asr.png         # 横轴 ASR，纵轴 ACC。理想防御应在左上角（高 ACC + 低 ASR）
*_loss.png               # 训练损失曲线。反映训练稳定性，异常波动可能提示攻击或防御失效
```

### per-client 日志：

```
*_score_by_round.png          # 每个 client 的异常分数分布。x轴：client_id，y轴：score，不同 round 用不同颜色/形状标记。恶意 client 分数显著偏高则 detection 有效
*_norm_by_round.png           # 每个 client 模型更新的 L2 范数。恶意 client 的 norm 可能异常大或小（取决于攻击），若无差别说明攻击隐蔽性强
*_cosine_by_round.png         # client 更新与全局更新方向的余弦相似度。恶意 client 的余弦值偏低（方向偏离），理想情况下与良性明显分离
*_score_hist.png              # 异常分数的直方图，按 suspicious（预测）分层。两个分布分离越清晰，检测性能越好
*_suspicious_vs_malicious.png   # 每轮被检测为恶意的 client 数量（红色实线）vs 真实恶意 client 数量（蓝色虚线）。两条线越接近说明检测越准确；红线高于蓝线表示误报过高，低于蓝线表示漏报过多
*_score_box.png               # 异常分数的箱线图，按 suspicious 分组。直观显示两组的中位数、四分位距和离群值
*_score_vs_norm.png           # 异常分数 vs 更新范数，颜色表示 suspicious。高分数 + 高范数区域若集中恶意 client，说明检测特征有效
*_score_vs_cosine.png         # 异常分数 vs 余弦相似度，颜色表示 suspicious。恶意 client 倾向于低余弦 + 高分数
*_score_by_gt.png             # 异常分数按真实标签（is_malicious）着色。验证 GT 与预测的一致性
*_roc.png                     # 包含 AUC 值的 ROC 曲线
```

### metrics 日志：

```
*_detection_confusion.png     # 混淆矩阵计数（TP, FP, FN, TN）随轮次变化。直观展示检测的绝对数量
*_detection_rates.png         # 检测率曲线：Precision, Recall, FPR。论文核心指标
*_detection_summary.png       # 检测汇总统计：总客户端数、保留数、过滤数、可疑数、跳过数
*_detection_scores.png        # 异常分数的统计量（均值、最大值、最小值）+ 余弦相似度均值/最小值 + 更新范数均值/最大值
*_aggregation_metrics.png     # 聚合层防御指标：Krum 保留数、Trimmed Mean 修剪数、Norm Clipping 裁剪比例、更新范数统计等
*_client_defense_metrics.png  # 客户端防御指标：过滤样本数、保留比例、平均分数等
*_detection_overview.png      # 检测总览图：合并 TP/FP/FN + Precision/Recall/FPR + suspicious/kept/filtered 在一张图上（紧凑版）
*_suspicious_vs_malicious.png # 每轮可疑客户端数（检测结果）与真实恶意客户端数对比曲线
*_score_box.png               # 根据 suspicious 标签的得分箱线图
*_score_vs_norm.png           # 得分与参数范数的散点图，颜色表示可疑标签
*_score_vs_cosine.png         # 得分与余弦相似度的散点图
*_score_by_gt.png             # 得分按真实标签（is_malicious）着色
```

### 多实验对比图（自动生成）

当运行 `python scripts/plot.py results/` 且 `results/` 下存在多个实验子目录时，会自动生成：

```
comparison_accuracy.png       # 所有实验的 ACC 曲线叠加对比
comparison_asr.png            # 所有实验的 ASR 曲线叠加对比
comparison_malicious_removal_rate.png # 恶意客户端移除率对比
comparison_benign_removal_rate.png    # 良性客户端移除率对比
summary_acc_vs_asr.png                # 最终 ACC vs ASR 散点图（每个实验一个点）
summary_aggregation_metrics.png       # 聚合性能条形图（恶意/良性移除率、平均保留的恶意客户端数、平均选中客户端数等）。
summary_detection_metrics.png         # 检测器性能柱状图（仅对有检测的实验）
```

## 当前项目架构：
```shell
fl-backdoor-system/
├── backend/                               # FastAPI 后端服务
│   ├── main.py                            # API 入口，WebSocket 日志推送，历史/目录索引查询
│   ├── batch_manager.py                   # GUI 界面的批量实验管理
│   ├── config_manager.py                  # 读写 pyproject.toml，提供前端表单 schema
│   ├── config_validation.py               # 写入配置前的严格校验
│   ├── experiment_runner.py               # 管理实验子进程
│   ├── experiment_worker.py               # Flower 运行时入口，仅在自有进程树内执行
│   ├── owned_processes.py                 # 只回收自己拥有的进程树，不按名字/全局 Ray 杀进程
│   ├── log_stream.py                      # 实验日志与 CSV 指标的独立游标读取
│   ├── attack_preview.py                  # 只读的触发器预览（与实验共用同一套图像变换）
│   ├── catalog.py                         # SQLite 只读模型：实验目录索引、配置快照、指标与文件索引
│   └── results_scanner.py                 # 扫描实验结果，提取指标
├── frontend/                              # React 前端界面
│   ├── src/
│   │   ├── api/client.ts                  # Axios 封装
│   │   ├── components/
│   │   │   ├── BatchMonitor/              # 批量实验日志
│   │   │   ├── ConfigForm/                # 动态配置表单
│   │   │   ├── ExperimentControl/         # 启动/停止 + 实时日志
│   │   │   └── ExperimentHistory/         # 历史实验卡片、详情、配置快照与附件状态
│   │   ├── pages/                         # 配置 / 历史 / 汇总 / 批量四个页面
│   │   └── App.tsx
│   ├── package.json
│   └── vite.config.ts
├── fl_backdoor/                           # 核心源码
│   ├── attacks/                           # 后门攻击实现（BadNets / WaNet / Frequency / DBA / FCBA）+ 恶意客户端选择
│   ├── defenses/                          # 防御系统（核心）：客户端防御 → 检测 → 聚合防御
│   │   ├── client/                        # 客户端防御（数据层）：Feature Filter
│   │   └── server/
│   │       ├── aggregation/               # 聚合层防御：Norm Clipping / Trimmed Mean / Krum
│   │       └── detection/                 # 检测层：Anomaly / Cosine / Score / Clustering
│   ├── dataset/                           # 数据集抽象层（MNIST / CIFAR-10）+ 元信息 + 数据分区
│   ├── client/                            # Flower ClientApp：本地训练 + 攻击注入 + 客户端防御
│   ├── server/                            # Flower ServerApp：攻击/防御 pipeline + 全局评估（ACC/ASR）
│   ├── utils/                             # 日志、指标、运行产物归档、随机种子、Ray 运行时等
│   ├── config.py                          # 统一配置层（ExperimentConfig），解析并校验所有参数
│   └── task.py                            # 模型定义（Net）、数据加载、训练/测试函数
│
├── scripts/                               # 辅助脚本（批量实验、结果汇总、数据库、绘图）
│   ├── batch_runner.py                    # 批量实验客户端：复用后端调度器（不直接改配置或拉起训练）
│   ├── summarize_results.py               # 汇总所有实验结果，生成 summary.csv 和 LaTeX 表格
│   ├── catalog.py                         # 实验数据库：导入旧结果、查询、状态、备份/恢复
│   └── plot.py                            # 绘图脚本：ACC/ASR、检测率、混淆矩阵、客户端分布、多实验对比图
│
├── docs/                                  # 使用说明
│   └── experiment-catalog.md              # SQLite 实验目录索引的存储边界、导入、API、备份与限制
├── results/                               # 实验输出目录（自动生成）
│   └── <experiment_name>/                 # 每个实验一个子目录：CSV / PNG / 日志 / 模型
├── platform_state/                        # 本地平台状态（SQLite 数据库等，Git 已忽略）
├── pyproject.toml                         # 项目配置 + 实验参数（攻击/防御、联邦超参、数据集、客户端数等）
├── final_model.pt                         # 训练完成的全局模型
├── .gitignore
├── LICENSE
└── README.md

```

## 许可证
本项目采用 AGPL-3.0-or-later 许可证，详见 [LICENSE](./LICENSE) 文件。
