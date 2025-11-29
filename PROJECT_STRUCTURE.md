# 项目结构说明

本文档说明了重构后的X-LoRA项目结构。

## 目录结构

```
X-Lora/
├── src/                        # 源代码目录
│   └── x_lora/                 # 主包目录
│       ├── __init__.py         # 包初始化文件
│       ├── models/             # LoRA模型实现
│       │   ├── __init__.py
│       │   ├── orthogonal_lora.py    # 正交LoRA实现
│       │   ├── structured_lora.py    # 结构化LoRA实现
│       │   └── svd_lora.py           # SVD LoRA实现
│       ├── data/               # 数据处理工具
│       │   ├── __init__.py
│       │   └── utils.py        # 数据加载和预处理
│       └── train/              # 训练相关工具
│           ├── __init__.py
│           ├── methods.py      # 模型构建函数
│           └── callbacks.py    # 训练回调函数
│
├── scripts/                    # 实验脚本
│   ├── run_experiments.sh      # 运行所有实验
│   ├── exp2.sh                 # 实验2
│   ├── exp3.sh                 # 实验3
│   └── ...                     # 其他实验脚本
│
├── tools/                      # 工具脚本
│   ├── aggregate_results.py    # 结果汇总工具
│   └── svd_energy_visual.py    # SVD能量可视化
│
├── configs/                    # 配置文件目录（预留）
├── tests/                      # 测试文件目录（预留）
├── figures/                    # 生成的图表
│
├── train.py                    # 主训练脚本
├── requirements.txt            # Python依赖
├── setup.py                    # 包安装配置
├── README.md                   # 项目说明
├── LICENSE                     # 许可证
└── .gitignore                  # Git忽略文件
```

## 主要变更

### 1. 模块化组织
- 所有核心代码移动到 `src/x_lora/` 包中
- 按功能划分为 `models/`, `data/`, `train/` 三个子模块
- 每个模块都有独立的 `__init__.py` 文件，便于导入

### 2. 脚本管理
- 所有实验脚本移动到 `scripts/` 目录
- 工具脚本移动到 `tools/` 目录

### 3. 导入路径更新
- 旧路径（已删除）：
  - `from data_utils import ...`
  - `from methods import ...`
  - `from OrthogonalLoRA import ...`
  
- 新路径：
  - `from src.x_lora.data import ...`
  - `from src.x_lora.train import ...`
  - `from src.x_lora.models import ...`

### 4. 文件清理
已删除根目录下的旧文件：
- `data_utils.py` → `src/x_lora/data/utils.py`
- `methods.py` → `src/x_lora/train/methods.py`
- `OrthogonalLoRA.py` → `src/x_lora/models/orthogonal_lora.py`
- `StructuredLoRA.py` → `src/x_lora/models/structured_lora.py`
- `SVDInit.py` → `src/x_lora/models/svd_lora.py`

## 使用方法

### 导入模块

```python
# 导入数据工具
from src.x_lora.data import load_sst2, compute_metrics

# 导入模型构建函数
from src.x_lora.train import (
    build_baseline_lora,
    build_svd_lora,
    build_orthogonal_lora,
    build_structured_lora
)

# 导入模型类
from src.x_lora.models import (
    OrthogonalLoRATrainer,
    apply_orthogonal_lora_to_model
)
```

### 运行训练

```bash
# 从项目根目录运行
python train.py --method baseline --rank 8
```

### 安装为包（可选）

```bash
pip install -e .
```

安装后可以直接导入：
```python
from x_lora.data import load_sst2
from x_lora.train import build_baseline_lora
```

## 下一步建议

1. **添加测试**: 在 `tests/` 目录中添加单元测试
2. **配置文件**: 使用 `configs/` 目录存放YAML/JSON配置文件
3. **文档**: 为每个模块添加详细的docstring
4. **CI/CD**: 添加持续集成配置

