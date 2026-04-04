# EvoAgentX-Excgec-and-Opro-Plus
## 项目背景
EvoAgentX：https://github.com/EvoAgentX/EvoAgentX

OPRO：https://github.com/google-deepmind/opro

EXCGEC：https://github.com/THUKElab/EXCGEC/tree/main

## 项目主要贡献
1. **任务流程设计**：基于 EXCGEC 数据集，设计并实现了面向中文语法纠错与解释生成的联合任务流程，使大语言模型能够同时输出纠错结果和结构化错误解释。

2. **优化方法适配与集成**：将三种自动化优化方法（TextGrad、OPRO、AFlow）引入中文语法纠错任务，解决了通用优化方法在结构化输出场景下的格式解析、评估反馈等技术适配问题，并在 EvoAgentX 框架中实现了统一集成。

3. **系统性实验对比**：在扩展的 EXCGEC 子集上，从优化效率（时间、token 消耗）和优化效果（GEC 与 EXP 双维度指标）两个角度对三种优化方法进行了系统对比，并通过案例分析揭示了不同优化策略的行为差异。

4. **适用性探讨**：结合定量实验结果与定性分析，讨论了自动优化方法在低复杂度语言任务中的适用性与局限性，为后续在类似任务中应用优化方法提供了经验参考。

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

### 评估器相关实现 `eval/`
- `excgec.py`：创建的，继承自 EvoAgentX 的 Benchmark 类的 EXCGEC 评估基准
- `my_evaluator.py`：重写的，继承自 EvoAgentX 的 Evaluator 类的 EXCGEC 评估器
- `my_score_calculator.py`：创建的，用于将 GEC 和 EXP 的评估结果（json 数据）转换为浮点数的计分器

### 优化器相关实现 `optimizers/`
- `my_textgrad_optimizer.py`：重写的，继承自 EvoAgentX 的 TextGradOptimizer 类的优化器，改为使用加权分数
- `my_opro_optimizer.py`：集成的，OPRO 方法在 EvoAgentX 框架下实现的优化器
- `my_aflow_optimizer.py`：重写的，继承自 EvoAgentX 的 AFlowOptimizer 类的优化器，并在文件内重写 AFlowEvaluator 类与 EvaluationUtils 类

### 优化运行代码
- `excgec_aflow.py`：可在 `log/excgec_aflow_output.log` 中 `Ctrl + F：Score for round` 查看优化详情
- `excgec_textgrad.py`：可在 `log/excgec_textgrad_output.log` 中 `Ctrl + F：Step` 查看优化详情
- `excgec_opro.py`：可在 `log/excgec_opro_output.log` 中 `Ctrl + F：Best instruction validation score` 查看优化详情

### 其他文件
- `generate_workflow.py`：辅助文件，用于验证 EvoAgentX 框架的工作流自动生成能力，并生成参考的初始工作流
- `run_workflow.py`：辅助文件，用于判断 `generate_workflow.py` 生成的初始工作流是否正确
- `single_agent.py`，`single_agent_instruct.py`：EvoAgentX 框架 CustomizeAgent 能力探索，可忽略
- `aflow/graph.py`，`aflow/prompt.py`：AFlowOptimizer 优化所需文件

## 部分实验结果
#### GEC 指标对比表，step = 10，dev_data_num & test_data_num = 500
| 方法 | p (精确率) | r (召回率) | f (F1分数) | acc (准确率) | tp | tn | fp (误报) | fn (漏报) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **优化前** | 0.0537 | 0.3137 | 0.0483 | 0.2954 | 0.1259 | 2.2058 | 1.4694 | 0.2636 |
| **TextGrad** | 0.1024 | 0.6275 | 0.0884 | 0.5882 | 0.2106 | 4.2540 | 2.8743 | 0.4802 |
| | ✅ +90.7% | ✅ +100.0% | ✅ +83.0% | ✅ +99.1% | ✅ +67.3% | ✅ +92.8% | ❌ +95.6% | ❌ +82.1% |
| **OPRO** | 0.1338 | 0.6345 | 0.1020 | 0.5960 | 0.2204 | 4.2763 | 2.8026 | 0.5197 |
| | ✅ +149.2% | ✅ +102.3% | ✅ +111.2% | ✅ +101.7% | ✅ +75.1% | ✅ +93.9% | ❌ +90.7% | ❌ +97.2% |
| **AFlow** | 0.3199 | 0.5177 | 0.2215 | 0.6617 | 0.4401 | 3.2650 | 1.3134 | 0.7512 |
| | ✅ +495.7% | ✅ +65.0% | ✅ +358.6% | ✅ +124.0% | ✅ +249.6% | ✅ +48.0% | ✅ -10.6% | ❌ +185.0% |

#### EXP 指标对比表，step = 10，dev_data_num & test_data_num = 500
| 方法 | hit_ratio | error_type.f1 | mae | rouge-1 | rouge-2 | rouge-L | meteor | bleu |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **优化前** | 0.2363 | 0.0853 | 0.2774 | 0.0916 | 0.0233 | 0.0531 | 0.0519 | 0.0064 |
| **TextGrad** | 0.2235 | 0.0826 | 0.2615 | 0.0922 | 0.0241 | 0.0529 | 0.0530 | 0.0069 |
| | ❌ -5.4% | ❌ -3.2% | ✅ -5.7% | ✅ +0.7% | ✅ +3.4% | ❌ -0.4% | ✅ +2.1% | ✅ +7.8% |
| **OPRO** | 0.2471 | 0.0884 | 0.2855 | 0.0991 | 0.0269 | 0.0584 | 0.0555 | 0.0078 |
| | ✅ +4.6% | ✅ +3.6% | ❌ +2.9% | ✅ +8.2% | ✅ +15.5% | ✅ +10.0% | ✅ +6.9% | ✅ +21.9% |
| **AFlow** | 0.3049 | 0.1325 | 0.3278 | 0.1273 | 0.0343 | 0.0752 | 0.0700 | 0.0097 |
| | ✅ +29.0% | ✅ +55.3% | ❌ +18.2% | ✅ +39.0% | ✅ +47.2% | ✅ +41.6% | ✅ +34.9% | ✅ +51.6% |
