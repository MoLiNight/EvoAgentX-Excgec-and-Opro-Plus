# EvoAgentX-Excgec-and-Opro-Plus
## 项目背景
EvoAgentX：https://github.com/EvoAgentX/EvoAgentX

OPRO：https://github.com/google-deepmind/opro

EXCGEC：https://github.com/THUKElab/EXCGEC/tree/main

## 项目主要贡献
- **EXCGEC 任务适配**：

  将中文语法纠错任务适配到 EvoAgentX 框架，包括：
  - 实现中文语法纠错工作流的自动生成与优化
  - 构建 EXCGEC 在 EvoAgentX 框架下的评估基准与评估器
  - 设计加权分数计算器，综合评估 GEC 与 EXP 双维度指标
- **OPRO 优化器集成**：

  在 EvoAgentX 框架下实现 OPRO 算法的集成，并通过实验验证其有效性
- **EvoAgentX 接口扩展与功能扩展**：
  
  在 EvoAgentX 的官方代码中，工作流，评估器与优化器仅支持字符串格式的输入与浮点数格式的输出，在本项目中，上述三者均可接受字典格式的输入和输出

## 环境配置
```Bash
git clone https://github.com/EvoAgentX/EvoAgentX.git
conda create -n evoagentx python=3.11
conda activate evoagentx

git clone https://github.com/THUKElab/EXCGEC.git
cd EXCGEC/LLaMA-Factory
pip install -r eval_requirements.txt
# EvoAgentX 的环境与 EXCGEC 的环境存在部分冲突，以 EvoAgentX 为准，否则代码无法正常运行

cd ..
cd EvoAgentX
pip install -e .  
pip install -r requirements.txt
# 不能使用下列 EvoAgentX 官方提供的 pip 命令进行安装，下列命令安装的 evoagentx 插件存在缺陷
# pip install evoagentx  # lack DEOptimizer, GAOptimizer 
# pip install git+https://github.com/EvoAgentX/EvoAgentX.git  # same lack DEOptimizer, GAOptimizer
```

## 源代码修改
**EXCGEC/benchmarks/xcgec/objects_eval.py**

源代码存在缺陷，添加 Optional 解决 Pydantic 试图将 None 转换 XEdit 而产生的报错
```python
from typing import List, Optional
from pydantic import BaseModel, Field
from .objects import XEdit

class BaseExplanationMetricResult(BaseModel):
    edit_hyp: Optional[XEdit] = Field(default=None)
    edit_ref: Optional[XEdit] = Field(default=None)
    error_description_score: Optional[float] = Field(default=None)
```
其余源代码通过类继承然后重写函数的方式修改

## 项目文件介绍
### 验证代码 `examples/`
- `aflow_math.py`：官方代码，经验证，EvoAgentX 框架中集成的 AFlow 算法有效
- `math_textgrad.py`：官方代码，经验证，EvoAgentX 框架中集成的 TextGrad 算法有效
- `evoprompt_workflow.py`：官方代码，因经济与效率原因放弃使用该方法进行优化
- `math_mipro.py`：官方代码，因源代码接口封装存在问题，MiproOptimizer 类的 MIPROv2._set_hyperparams_from_run_mode（）始终报错，故判断该算法无法使用。经重证，并非模块安装导致的导入问题

### 辅助文件
- `generate_workflow.py`：辅助文件，用于验证 EvoAgentX 框架的工作流自动生成能力，并生成优化前的初始工作流
- `run_workflow.py`：辅助文件，用于判断 `generate_workflow.py` 生成的初始工作流是否正确
- `benchmark_and_evaluation.py`：辅助文件，用于判断 `excgec.py` 内定义的 EXCGEC 类是否正确，并对工作流进行简单的评估

### 核心类实现
- `excgec.py`：创建的，继承自 EvoAgentX 的 Benchmark 类的 EXCGEC 评估基准
- `my_evaluator.py`：重写的，继承自 EvoAgentX 的 Evaluator 类的 EXCGEC 评估器
- `my_score_calculator.py`：创建的，用于将 GEC 和 EXP 的评估结果（json 数据）转换为浮点数的计分器
- `my_textgrad_optimizer.py`：重写的，继承自 EvoAgentX 的 TextGradOptimizer 类的优化器，改为使用加权分数
- `my_opro_optimizer.py`：集成的，OPRO 方法在 EvoAgentX 框架下实现的优化器
- `my_aflow_optimizer.py`：重写的，继承自 EvoAgentX 的 AFlowOptimizer 类的优化器，并在文件内重写 AFlowEvaluator 类与 EvaluationUtils 类

### 优化运行代码
- `excgec_opro.py`：可在 `excgec_opro_output.log` 中 `Ctrl + F：Step` 查看优化详情
- `excgec_aflow.py`：可在 `excgec_aflow_output.log` 中 `Ctrl + F：Score for round` 查看优化详情
- `excgec_textgrad.py`：可在 `excgec_textgrad_output.log` 中 `Ctrl + F：Step` 查看优化详情

### 其他目录介绍
- `aflow/`：
  - `graph.py` 和 `prompt.py`：`excgec_aflow.py` 优化所需文件
  - `optimized/`：`excgec_aflow.py` 输出路径
- `output/`：
  - `original_workflow.json`：`generate_workflow.py` 生成的初始工作流图
  - `results.md`：初始工作流图在 `run_workflow.py` 上运行的返回结果
- `opro/`：`excgec_opro.py` 输出路径
- `textgrad/`：`excgec_textgrad.py` 输出路径

## 部分实验结果
### GEC 指标对比表
| 方法 | p (精确率) | r (召回率) | f (F1分数) | acc (准确率) | fp (假正例) | fn (假负例) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **优化前** | 0.0835 | 0.6500 | 0.0952 | 0.5872 | 3.1400 | 0.4600 |
| **TextGrad** | 0.0780 | 0.6975 | 0.0860 | 0.5815 | 3.2029 | 0.4031 |
| | ❌ -6.6% | ✅ +7.3% | ❌ -9.7% | ❌ -1.0% | ✅ +2.0% | ✅ -12.4% |
| **OPRO** | 0.0814 | 0.6887 | 0.0895 | 0.5896 | 2.6981 | 0.4151 |
| | ❌ -2.5% | ✅ +5.9% | ❌ -6.0% | ✅ +0.4% | ✅ -14.1% | ✅ -9.8% |
| **AFlow** | 0.3234 | 0.5149 | 0.2359 | 0.6721 | 1.2738 | 0.7381 |
| | ✅ +287% | ❌ -20.8% | ✅ +148% | ✅ +14.5% | ✅ -59.4% | ❌ +60.5% |

### EXP 指标对比表
| 方法 | hit_ratio | error_type.f1 | mae | rouge-1 | rouge-2 | rouge-L | meteor | bleu |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **优化前** | 0.1725 | 0.0417 | 0.1683 | 0.0739 | 0.0198 | 0.0428 | 0.0445 | 0.0089 |
| **TextGrad** | 0.1295 | 0.0386 | 0.1999 | 0.0608 | 0.0148 | 0.0338 | 0.0355 | 0.0038 |
| | ❌ -24.9% | ❌ -7.4% | ❌ -18.8% | ❌ -17.7% | ❌ -25.3% | ❌ -21.0% | ❌ -20.2% | ❌ -57.3% |
| **OPRO** | 0.2260 | 0.0723 | 0.3197 | 0.0903 | 0.0228 | 0.0510 | 0.0505 | 0.0066 |
| | ✅ +31.0% | ✅ +73.4% | ❌ -89.9% | ✅ +22.2% | ✅ +15.2% | ✅ +19.2% | ✅ +13.5% | ❌ -25.8% |
| **AFlow** | 0.2580 | 0.1483 | 0.3717 | 0.1267 | 0.0324 | 0.0747 | 0.0694 | 0.0099 |
| | ✅ +49.6% | ✅ +256% | ❌ -121% | ✅ +71.4% | ✅ +63.6% | ✅ +74.5% | ✅ +56.0% | ✅ +11.2% |
