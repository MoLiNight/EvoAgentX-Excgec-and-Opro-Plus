import evoagentx.workflow.operators as operator
import aflow.optimized.round_16.prompt as prompt_custom
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
        self.review = operator.Review(self.llm)  # New operator for validation
    
    async def __call__(self, problem: str):
        # Enhance input with additional context for better error detection
        enhanced_input = f"Problem: {problem}\n\nPlease analyze and correct the sentence, providing detailed explanations for each error found."
        
        solution = await self.custom(
            input=enhanced_input,
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

                Problem: 
            """
        )
        
        # Review the output for consistency and correctness
        review_input = f"Original Sentence: {problem}\nCorrected Sentence: {solution['response']}"
        reviewed_solution = await self.review(input=review_input, instruction=r"""
            请检查以下中文句子的语法纠错结果是否合理，并提供改进建议。如果发现错误或不一致，请指出并修正。

            输入:
            - 原句: {original}
            - 纠正后: {corrected}

            输出格式（JSON）:
            {{
                "is_valid": bool,               # 是否有效
                "corrections": [                # 修正建议
                    {{
                        "position": int,        # 位置
                        "original": str,        # 原内容
                        "corrected": str,       # 修正后
                        "reason": str           # 修正原因
                    }},
                    ...
                ],
                "summary": str                  # 总结
            }}
        """)
        
        return reviewed_solution['response']
