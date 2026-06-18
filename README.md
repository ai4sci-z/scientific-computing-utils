# scientific-computing-utils

科学计算常用工具：误差分析、收敛阶验证、绘图、PDE 小工具。

## 模块

| 文件 | 功能 |
|------|------|
| `convergence.py` | L1/L2/Linf 误差、收敛阶计算、对数收敛图 |
| `grid.py` | 均匀网格、非均匀网格、边界条件辅助 |
| `viz.py` | 流场向量图、等值线、误差分布热力图 |

## 快速使用

```python
from convergence import convergence_table
# 传入不同网格的 (N, error) 列表，自动输出收敛表格和图
convergence_table([(50, 1e-3), (100, 6e-5), (200, 4e-6), (400, 2.5e-7)])
```
