---
tags: [vision, fds]
dataset: [CIFAR-10]
framework: [torch, torchvision]
---

# 面向联邦学习的后门攻击与防御系统

当前项目架构：
```shell
fl-backdoor-system/ 
├── fl_backdoor/ 
│ ├── attacks/ 
│ │ ├── __init__.py 
│ │ ├── badnets.py 
│ │ ├── base.py 
│ │ ├── wanet.py 
│ │ ├── frequency.py 
│ │ └── 未完待续，支持 attack="none"（IdentityAttack） 

│ ├── defenses/                      # 防御系统（核心）（允许任何一层的防御为none，目前已实现的都可以为none）
│ │ ├── base.py                    # DefenseBase / DefenseConfig（聚合层基类）
│ │ ├── init__.py                # build_defense / build_defended_strategy
│ │ ├── pipeline.py
│ │ ├── client/                    # 客户端防御（数据层）
│ │ │ ├── base.py
│ │ │ ├── feature_filter.py
│ │ │ └── __init__.py            # build_client_defense()
│ │ └── server/                    # 服务端防御
│ │    ├── aggregation/           # 聚合层防御（已完成）
│ │    │ ├── norm_clipping.py
│ │    │ ├── trimmed_mean.py
│ │    │ ├── krum.py
│ │    │ └── __init__.py
│ │    └── detection/             # 检测层（刚完成）
│ │       ├── base.py
│ │       ├── anomaly_detection.py
│ │       ├── cosine_detection.py
│ │       └── __init__.py        # build_detection()

│ ├── client/ 
│ │ ├── __init__.py 
│ │ └── client.py 

│ ├── server/ 
│ │ ├── __init__.py 
│ │ └── server.py 

│ ├── utils/ 
│ │ ├── __init__.py 
│ │ ├── logger.py
│ │ └── plot.py 

│ ├── __init__.py 
│ └── task.py 

├── results/    # 项目运行后自动生成
├── .gitignore
├── final_model.pt 
├── LICENSE
├── pyproject.toml
└── README.md

```

### 安装依赖和运行项目

参考[flower 官网](https://flower.ai/docs/framework/tutorial-quickstart-pytorch.html)。

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

# 6. 运行并输出日志
flwr run . --stream

# 我未来可能会加个 setup_env.py 一类的脚本，实现检测 cuda，安装 torch 一类的功能。
```

# 画图

系统会将日志信息存储在 fl-backdoor-system/results 下。
用此命令画图：
```bash
 python ./fl_backdoor/utils/plot.py ./results/name.csv
```

