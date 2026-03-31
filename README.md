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
│ ├── defenses/ 
│ │ ├── __init__.py 
│ │ ├── base.py 
│ │ ├── krum.py
│ │ ├── norm_clipping.py 
│ │ ├── trimmed_mean.py  
│ │ └── 未完待续，支持 defense="none" 
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

