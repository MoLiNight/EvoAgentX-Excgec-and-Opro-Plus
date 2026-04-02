# EvoAgentX-Excgec-and-Opro-Plus
## 项目背景
EvoAgentX：https://github.com/EvoAgentX/EvoAgentX

OPRO：https://github.com/google-deepmind/opro

EXCGEC：https://github.com/THUKElab/EXCGEC/tree/main

## 项目主要贡献
1. **统一建模框架**：面向中文语法纠错任务，构建了一个结合纠错（GEC）与解释生成（EXP）的统一建模框架，使模型在输出纠错结果的同时提供相应解释，从而提升结果的可理解性。

2. **多优化方法引入**：将多种优化方法（如 OPRO 与 AFlow）引入该任务中，对提示与推理过程进行自动优化，并系统分析了不同优化策略在中文语法纠错任务中的效果差异。

3. **性能验证与分析**：通过对比优化前后模型性能，验证了基于优化方法的性能提升效果，并分析了优化过程对模型输出行为的影响。

4. **适用性探讨**：结合定量实验与案例分析，探讨了优化方法在低复杂度语言任务中的适用性与局限性，为后续在类似任务中应用自动优化方法提供了参考。

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
#### GEC 指标对比表，step = 10，dev_data_num & test_data_num = 100
| 方法 | p (精确率) | r (召回率) | f (F1分数) | acc (准确率) | tp | tn | fp (误报) | fn (漏报) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **优化前** | 0.0835 | 0.6500 | 0.0952 | 0.5872 | 0.2600 | 4.6400 | 3.1400 | 0.4600 |
| **TextGrad** | 0.0780 | 0.6975 | 0.0860 | 0.5815 | 0.2741 | 4.6173 | 3.2029 | 0.4031 |
| | ❌ -6.6% | ✅ +7.3% | ❌ -9.7% | ❌ -1.0% | ✅ +5.4% | ❌ -0.5% | ❌ +2.0% | ✅ -12.4% |
| **OPRO** | 0.0814 | 0.6887 | 0.0895 | 0.5896 | 0.2453 | 4.0566 | 2.6981 | 0.4151 |
| | ❌ -2.5% | ✅ +5.9% | ❌ -6.0% | ✅ +0.4% | ❌ -5.7% | ✅ -12.6% | ✅ -14.1% | ✅ -9.8% |
| **AFlow** | 0.3234 | 0.5149 | 0.2359 | 0.6721 | 0.4405 | 3.2500 | 1.2738 | 0.7381 |
| | ✅ +287% | ❌ -20.8% | ✅ +148% | ✅ +14.5% | ✅ +69.4% | ✅ -30.0% | ✅ -59.4% | ❌ +60.5% |

#### EXP 指标对比表，step = 10，dev_data_num & test_data_num = 100
| 方法 | hit_ratio | error_type.f1 | mae | rouge-1 | rouge-2 | rouge-L | meteor | bleu |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **优化前** | 0.1725 | 0.0417 | 0.1683 | 0.0739 | 0.0198 | 0.0428 | 0.0445 | 0.0089 |
| **TextGrad** | 0.1295 | 0.0386 | 0.1999 | 0.0608 | 0.0148 | 0.0338 | 0.0355 | 0.0038 |
| | ❌ -24.9% | ❌ -7.4% | ❌ +18.8% | ❌ -17.7% | ❌ -25.3% | ❌ -21.0% | ❌ -20.2% | ❌ -57.3% |
| **OPRO** | 0.2260 | 0.0723 | 0.3197 | 0.0903 | 0.0228 | 0.0510 | 0.0505 | 0.0066 |
| | ✅ +31.0% | ✅ +73.4% | ❌ +89.9% | ✅ +22.2% | ✅ +15.2% | ✅ +19.2% | ✅ +13.5% | ❌ -25.8% |
| **AFlow** | 0.2580 | 0.1483 | 0.3717 | 0.1267 | 0.0324 | 0.0747 | 0.0694 | 0.0099 |
| | ✅ +49.6% | ✅ +256% | ❌ +121% | ✅ +71.4% | ✅ +63.6% | ✅ +74.5% | ✅ +56.0% | ✅ +11.2% |

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
