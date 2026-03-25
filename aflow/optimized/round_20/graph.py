import evoagentx.workflow.operators as operator
import aflow.optimized.round_20.prompt as prompt_custom
from evoagentx.models.model_configs import LLMConfig
from evoagentx.benchmark.benchmark import Benchmark
from evoagentx.models.model_utils import create_llm_instance

class Workflow:

    def __init__(
        self,
        name: str,
        llm_config: LLMConfig,
        benchmark: Benchmark
    ):
        self.name = name
        self.llm = create_llm_instance(llm_config)
        self.benchmark = benchmark
        self.custom = operator.Custom(self.llm)
        self.review = operator.Review(self.llm)  # New operator for post-processing review
    
    async def __call__(self, problem: str):
        solution = await self.custom(
            input=problem,
            instruction=r"""
                对给定的中文句子进行语法纠错，输出纠正后的句子和详细的错误解释。

                注意：
                1. 工作流的输入：
                    {{
                        "source": str                        # 原病句
                    }}

                2. 工作流的输出: 一个 JSON 字符串，JSON 字符串不应该有 Markdown 格式
                    {{
                        "target": str,                       # 模型纠正后的句子
                        "edits": [                           # 模型输出的解释
                            {{
                                "src_interval": List[int],   # [start, end] indicates the character position of the error in the source
                                "tgt_interval": List[int],   # [start, end] indicates the character position of the correction in the target
                                "src_content": str,          # The original erroneous string
                                "tgt_content": str,          # The corresponding correct string
                                "src_tokens": List[str],     # Tokenization result of the original string
                                "tgt_tokens": List[str],     # Tokenization result of the correct string
                                "error_type": str,           # Error type (e.g., "word redundancy", "word missing", "word misuse")
                                "error_severity": int,       # Error severity (1-5, higher number indicates more severe)
                                "error_description": str     # Detailed error description and correction explanation
                            }},
                            ...  # Multiple edit operations can exist
                        ]
                    }}
                
                3. 错误类型范围：(必须从以下的错误中选择进行回答)
                标点冗余，标点丢失，标点误用；字音混淆错误，字形混淆错误，词内部字符异位错误，命名实体拼写错误；
                词语冗余，词语丢失，词语误用；词序不当，逻辑不通，句式杂糅；照应错误，歧义错误，语气不协调；其他错误。

                4. 特别说明：
                    - 如果句子存在多种可能的纠正方式，请优先选择最符合语法规范、最自然的一种。
                    - 如果句子语义模糊或无法确定错误类型，请标注为“其他错误”并给出合理解释。
                    - 请确保每个错误项都包含完整的 src_interval 和 tgt_interval，即使没有实际修改。

                Problem: 
            """
        )
        # Review step to validate output format and ensure no missing fields
        reviewed_solution = await self.review(
            input=solution['response'],
            instruction=r"""
                请检查以下 JSON 输出是否符合以下格式要求：

                1. 必须包含 "target" 和 "edits" 两个字段。
                2. "edits" 是一个数组，每个元素是一个对象，包含以下字段：
                    - src_interval: [start, end]
                    - tgt_interval: [start, end]
                    - src_content: str
                    - tgt_content: str
                    - src_tokens: List[str]
                    - tgt_tokens: List[str]
                    - error_type: str (必须从指定错误类型中选择)
                    - error_severity: int (1-5)
                    - error_description: str

                3. 所有字段必须存在，不能缺失。
                4. JSON 格式必须正确，不能有语法错误。

                请返回一个 JSON，格式如下：
                {{
                    "valid": bool,  # 是否有效
                    "message": str, # 验证结果信息
                    "fixed_json": str  # 如果无效，返回修复后的 JSON
                }}
            """
        )
        return reviewed_solution['response']
