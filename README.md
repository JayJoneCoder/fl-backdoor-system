---
tags: [quickstart, vision, fds]
dataset: [CIFAR-10]
framework: [torch, torchvision]
---

# fl-backdoor-system


```shell
fl-backdoor-system/ 
├── fl_backdoor/ 
│ ├── attacks/ 
│ │ ├── __init__.py 
│ │ ├── badnets.py 
│ │ ├── base.py 
│ │ ├── wanet.py 
│ │ └── 未完待续，支持 attack="none"（IdentityAttack） 

│ ├── defenses/                      # 防御系统（核心）（应允许任何一层的防御为none，目前已实现的都可以为none）
│ │ ├── base.py                    # DefenseBase / DefenseConfig（聚合层基类）
│ │ ├── __init__.py                # build_defense / build_defended_strategy
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

### Install dependencies and run theproject

Install the dependencies defined in `pyproject.toml` as well as the `fl_backdoor` package.

```bash
pip install -e .

flwr run . --stream
```

# plot
```
 python ./fl_backdoor/utils/plot.py ./results/name.csv
```

