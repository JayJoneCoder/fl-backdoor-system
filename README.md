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
│ │ ├── base.py 
│ │ ├── badnets.py 
│ │ ├── wanet.py 
│ │ └── (...) 
│ ├── defenses/ 
│ │ ├── __init__.py 
│ │ ├── base.py 
│ │ ├── norm_clipping.py 
│ │ ├── trimmed_mean.py 
│ │ └── (...)  
│ ├── client/ 
│ │ ├── __init__.py 
│ │ └── client.py 
│ ├── server/ 
│ │ ├── __init__.py 
│ │ └── server.py 
│ ├── __init__.py 
│ └── task.py 
├── pyproject.toml   
└── README.md
```

### Install dependencies and run theproject

Install the dependencies defined in `pyproject.toml` as well as the `fl_backdoor` package.

```bash
pip install -e .

flwr run . --stream
```

