import evoagentx.workflow.operators as operator
import aflow.optimized.round_14.prompt as prompt_custom
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
    
    async def __call__(self, problem: str):
        # Step 1: Use custom to generate initial correction and explanation
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

                Problem: 
            """
        )
        
        # Step 2: Validate and clean the response
        response = solution['response']
        try:
            import json
            data = json.loads(response)
            for edit in data.get('edits', []):
                if 'error_severity' in edit and not (1 <= edit['error_severity'] <= 5):
                    edit['error_severity'] = 3  # Default to medium severity if invalid
            response = json.dumps(data)
        except:
            response = '{"target": "", "edits": []}'
        
        # Step 3: Use a second custom call to review and refine the output
        refined_solution = await self.custom(
            input=response,
            instruction=r"""
                请对以下语法纠错结果进行审查和优化，确保格式正确、内容完整、语言自然。

                输入格式：
                {{
                    "target": str,                       # 纠正后的句子
                    "edits": [                           # 纠错解释
                        {{
                            "src_interval": List[int],   # 原句中的错误位置
                            "tgt_interval": List[int],   # 修正后的位置
                            "src_content": str,          # 原错误内容
                            "tgt_content": str,          # 修正后的内容
                            "src_tokens": List[str],     # 原句分词
                            "tgt_tokens": List[str],     # 修正后分词
                            "error_type": str,           # 错误类型
                            "error_severity": int,       # 严重程度（1-5）
                            "error_description": str     # 详细说明
                        }},
                        ...
                    ]
                }}

                输出要求：
                - 保持原始结构不变
                - 修正可能存在的格式错误或语义不清
                - 保证所有字段都存在且非空
                - 若发现无效数据，请补充合理默认值

                请返回优化后的 JSON 内容。
            """
        )
        
        # Step 4: Final validation and return
        final_response = refined_solution['response']
        try:
            import json
            final_data = json.loads(final_response)
            if not final_data.get('target'):
                final_data['target'] = "无法识别错误"
            if not final_data.get('edits'):
                final_data['edits'] = []
            final_response = json.dumps(final_data)
        except:
            final_response = '{"target": "无法识别错误", "edits": []}'
        
        return final_response
