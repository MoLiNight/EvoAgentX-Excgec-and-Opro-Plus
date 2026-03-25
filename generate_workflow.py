import os
from dotenv import load_dotenv
from evoagentx.models import AliyunLLM, AliyunLLMConfig
from evoagentx.agents import AgentManager
from evoagentx.workflow import WorkFlowGenerator, WorkFlowGraph, WorkFlow

load_dotenv()  
my_api_key = os.getenv("DASHSCOPE_API_KEY")

def main():
    target_directory = "./output"
    module_save_path = os.path.join(target_directory, "original_workflow.json")
    os.makedirs(target_directory, exist_ok=True)

    llm_config = AliyunLLMConfig(
        model="qwen-turbo",  # qwen-turbo、qwen-plus、qwen-max and so on
        aliyun_api_key=my_api_key,
        temperature=0.1,
        max_tokens=16000,
        stream=False,
        output_response=True
    )
    llm = AliyunLLM(llm_config)
    goal = f"""
        创建一个能够检测出中文句子中的语法错误，纠正错误并给出错误解释的工作流。

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
    """

    wf_generator = WorkFlowGenerator(llm=llm)
    workflow_graph: WorkFlowGraph = wf_generator.generate_workflow(goal=goal)

    workflow_graph.save_module(module_save_path)
    workflow_graph.display()

    print(f"✅ The workflow_graph has been saved to：{module_save_path}")

if __name__ == "__main__":
    main()