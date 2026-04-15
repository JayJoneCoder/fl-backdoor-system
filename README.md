---
tags: [fl, backdoor-attack, backdoor-defense, federated-learning, security, privacy]
dataset: [CIFAR-10]
framework: [pytorch, flwr]
---

# 面向联邦学习的后门攻击与防御系统
[![License: GPL-3.0-or-later](https://img.shields.io/badge/License-GPL--3.0--or--later-blue.svg?logo=gnu)](LICENSE)
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
- **Ground Truth 追踪**：支持 random/fixed 恶意客户端选择，实现检测性能的精确评估
- **Web 可视化平台**：基于 FastAPI + React + Ant Design，支持在线配置实验、实时日志监控、历史结果浏览

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
cd backend
uvicorn main:app --reload --port 8000

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

# 批量实验
python scripts/batch_runner.py --config my_experiments.json
```


### 批量实验与结果汇总
#### 批量运行多个实验
编辑 `scripts/batch_runner.py` 中的 `DEFAULT_EXPERIMENTS` 列表，或提供 JSON 配置文件，然后运行：

```bash
# 默认配置，或手动修改该脚本中的配置
python scripts/batch_runner.py

# 或使用 json 文件自定义配置
python scripts/batch_runner.py --config my_experiments.json
```

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
python plot.py results/baseline.csv results/badnets.csv

# 通配符匹配
python plot.py results/*_clients.csv

# 直接指定目录（递归查找所有 .csv，推荐）
python plot.py results/
```
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

当运行 `python fl_backdoor/utils/plot.py results/` 且 `results/` 下存在多个实验子目录时，会自动生成：

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
│   ├── main.py                            # API 入口，WebSocket 日志推送
│   ├── batch_manager.py                   # GUI 界面的批量实验管理
│   ├── config_manager.py                  # 读写 pyproject.toml，提供前端表单 schema
│   ├── experiment_runner.py               # 管理 flwr run 子进程
│   └── results_scanner.py                 # 扫描实验结果，提取指标
├── frontend/                              # React 前端界面
│   ├── src/
│   │   ├── api/client.ts                  # Axios 封装
│   │   ├── components/
│   │   │   ├── BatchMonitor/              # 批量实验日志
│   │   │   ├── ConfigForm/                # 动态配置表单
│   │   │   ├── ExperimentControl/         # 启动/停止 + 实时日志
│   │   │   └── ExperimentHistory/         # 历史实验卡片与详情
│   │   ├── pages/
│   │   │   ├── ConfigPage.tsx
│   │   │   ├── HistoryPage.tsx
│   │   │   ├── SummaryPage.tsx
│   │   │   └── BatchPage.tsx
│   │   └── App.tsx
│   ├── package.json
│   └── vite.config.ts
├── fl_backdoor/                           # 核心源码
│   ├── attacks/                           # 后门攻击实现（BadNets / WaNet / Frequency / DBA / FCBA）
│   │   ├── __init__.py                    # build_attack() 工厂，支持 attack="none"
│   │   ├── base.py                        # AttackBase + AttackConfig，定义恶意客户端选择接口
│   │   ├── badnets.py                     # 像素级贴片触发器（正方形白块）
│   │   ├── wanet.py                       # 弹性扭曲 + 噪声，隐形后门
│   │   ├── frequency.py                   # 频域攻击（FFT/DCT），篡改高频/低频分量
│   │   ├── dba.py                         # 分布式后门攻击（DBA），将全局触发器拆分为多个局部子模式，由不同恶意客户端协同注入
│   │   ├── fcba.py                        # 全组合后门攻击（FCBA），生成 2^m-2 种局部触发器组合，每个恶意客户端使用唯一组合
│   │   └── selection.py                   # 恶意客户端选择（random / fixed），支持 round‑level 确定性采样
│   │
│   ├── defenses/                          # 防御系统（核心）
│   │   ├── base.py                        # DefenseBase + DefenseConfig，聚合防御基类
│   │   ├── __init__.py                    # build_defense_pipeline() 工厂入口
│   │   ├── pipeline.py                    # DefensePipelineFedAvg，统一调度：client 防御 → detection → 聚合防御
│   │   │
│   │   ├── client/                        # 客户端防御（数据层预处理）
│   │   │   ├── base.py                    # ClientDefenseBase
│   │   │   ├── feature_filter.py          # 基于特征激活值的样本过滤（可疑样本剔除）
│   │   │   └── __init__.py                # build_client_defense()
│   │   │
│   │   └── server/                        # 服务端防御
│   │       ├── aggregation/               # 聚合层防御（拜占庭鲁棒聚合）
│   │       │   ├── norm_clipping.py       # 梯度裁剪：限制单个 client 更新范数
│   │       │   ├── trimmed_mean.py        # 修剪平均：去掉最大/最小更新后平均
│   │       │   ├── krum.py                # Krum：选择距离其他更新最近的 k 个更新
│   │       │   └── __init__.py
│   │       │
│   │       └── detection/                 # 检测层（聚合前过滤恶意更新）
│   │           ├── base.py                # DetectionBase + DetectionReport
│   │           ├── anomaly_detection.py   # 基于 z‑score 的范数异常检测
│   │           ├── cosine_detection.py    # 余弦相似度检测（方向异常）
│   │           ├── score_detection.py     # 百分位数阈值过滤
│   │           ├── clustering_detection.py # K‑means 聚类 + 轮廓系数自动分组
│   │           ├── features.py            # 特征提取（范数、余弦、z‑score、deltas）
│   │           └── __init__.py            # build_detection()
│   │
│   ├── dataset/                           # 数据集抽象层（支持多数据集切换）
│   │   ├── __init__.py                    # get_dataset() 工厂，注册所有支持的数据集
│   │   ├── base.py                        # BaseDataset 抽象基类，定义 load_partition / load_centralized_test
│   │   ├── config.py                      # DatasetMeta 数据类，描述数据集元信息（shape、类别数、均值/方差等）
│   │   ├── cifar10.py                     # CIFAR-10 实现（基于 HuggingFace）
│   │   └── mnist.py                       # MNIST 实现（基于 torchvision，支持离线缓存）
│   │
│   ├── client/
│   │   ├── __init__.py
│   │   └── client.py                      # Flower ClientApp，执行本地训练 + 攻击注入 + 客户端防御
│   │
│   ├── server/
│   │   ├── __init__.py
│   │   └── server.py                      # Flower ServerApp，初始化攻击 + 防御 pipeline + 全局评估（ACC/ASR）
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   └── logger.py                      # CSVLogger：记录主实验指标 + 检测指标 + 客户端级指标
│   │
│   ├── __init__.py
│   ├── config.py                          # 统一配置层（ExperimentConfig），从 run_config 解析所有参数，并提供校验与参数字典
│   └── task.py                            # 模型定义（Net，自适应输入形状）、数据加载、训练/测试函数
│
├── scripts/                               # 辅助脚本（批量实验、结果汇总）
│   ├── batch_runner.py                    # 命令行界面的批量运行实验脚本（自动修改 pyproject.toml，串行执行）
│   ├── summarize_results.py               # 汇总所有实验结果，生成 summary.csv 和 LaTeX 表格
│   └── plot.py                        # 绘图脚本：自动生成 ACC/ASR、检测率、混淆矩阵、客户端分布图、多实验对比图
│
├── results/                               # 实验输出目录（自动生成）
│   ├── <experiment_name>/                 # 每个实验一个子目录
│   │   ├── *_acc.png                      # 准确率曲线
│   │   ├── *_asr.png                      # 后门攻击成功率曲线
│   │   ├── *_detection_rates.png          # Precision / Recall / FPR 曲线（论文核心）
│   │   ├── *_detection_confusion.png      # TP / FP / FN / TN 混淆矩阵计数
│   │   ├── *_suspicious_vs_malicious.png  # 检测出的可疑数 vs 真实恶意数
│   │   ├── *_score_*.png                  # 客户端异常分数相关图
│   │   └── *.csv                          # 原始日志
│   ├── comparison_accuracy.png            # 多实验 ACC 曲线叠加对比
│   ├── comparison_asr.png                 # 多实验 ASR 曲线叠加对比
│   ├── summary_acc_vs_asr.png             # 最终 ACC vs ASR 散点图（防御有效性）
│   ├── summary_detection_metrics.png      # 检测器性能柱状图（Precision/Recall/FPR/AUC）
│   ├── summary.csv                        # 汇总表格（CSV）
│   └── summary_table.tex                  # 汇总表格（LaTeX，可直接用于论文）
│
├── pyproject.toml                         # 项目配置 + 实验参数（攻击/防御类型、联邦学习超参数、数据集、客户端总数等）
├── final_model.pt                         # 训练完成的全局模型
├── .gitignore
├── LICENSE
└── README.md

```

## 许可证
本项目采用 GPL-3.0-or-later 许可证，详见 [LICENSE](./LICENSE) 文件。
