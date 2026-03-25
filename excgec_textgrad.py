import os
import json
from dotenv import load_dotenv

from evoagentx.models import AliyunLLM, AliyunLLMConfig
from excgec import EXCGEC
from evoagentx.core.logging import logger
from evoagentx.core.callbacks import suppress_logger_info
from evoagentx.workflow import WorkFlowGraph
from evoagentx.prompts import StringTemplate
from evoagentx.agents.agent_manager import AgentManager
from evoagentx.workflow.workflow_graph import SequentialWorkFlowGraph

from my_evaluator import MyEvaluator
from my_score_calculator import MyScoreCalculator
from my_textgrad_optimizer import MyTextGradOptimizer

load_dotenv()
my_api_key = os.getenv("DASHSCOPE_API_KEY")

class ExcgecForTextgrad(EXCGEC):

    def _load_data(self):
        # load the original test data 
        super()._load_data()
        # split the data into train, dev and test
        import numpy as np 
        np.random.seed(42)
        permutation = np.random.permutation(len(self._test_data))
        full_test_data = self._test_data

        self._train_data = [full_test_data[idx] for idx in permutation[:100]]
        self._dev_data = [full_test_data[idx] for idx in permutation[100:200]]
        self._test_data = [full_test_data[idx] for idx in permutation[200:300]]

# 评估器的数据输入处理函数
def collate_func(example):
    return {"source": example["source"]}

# 评估器的数据输出处理函数
def output_postprocess_func(output: str):
    output = json.loads(output)
    return output

def main():

    llm_config = AliyunLLMConfig(
        model="qwen-turbo", 
        aliyun_api_key=my_api_key, 
        temperature=0.1,
        max_tokens=16000
    )
    executor_llm = AliyunLLM(config=llm_config)
    optimizer_llm = AliyunLLM(config=llm_config)

    benchmark = ExcgecForTextgrad()
    workflow_graph = SequentialWorkFlowGraph(
        goal=f"""
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
        """,
        tasks = [
            {
                "name": "error_detection_agent",
                "description": "This agent identifies potential grammar errors in a Chinese sentence, including their positions, types, and severity levels.",
                "inputs": [
                    {"name": "source", "type": "string", "required": True, "description": "The original Chinese sentence with possible grammar errors."}
                ],
                "outputs": [
                    {"name": "detected_errors", "type": "object", "required": True, "description": "A list of detected errors with their positions, types, and severity levels."}
                ],
                "prompt": None,
                "prompt_template": StringTemplate(instruction="### Objective\nIdentify potential grammar errors in the given Chinese sentence, including their positions, types, and severity levels.\n\n### Instructions\n1. Read and understand the user's goal: <input>{goal}</input>\n2. Analyze the original Chinese sentence: <input>{source}</input>\n3. Review the problem analysis: <input>{problem_analysis}</input>\n4. Identify all potential grammar errors in the sentence, such as punctuation issues, word redundancy, or grammatical structure problems.\n5. For each detected error, determine its type (e.g., '标点冗余', '词语误用'), position (start and end indices), and severity level (1-5).\n6. Output a structured list of detected errors with their details.\n\n### Output Format\nYour final output should ALWAYS in the following format:\n\n## Thought\nBriefly explain the reasoning process for detecting the errors in the given sentence.\n\n## detected_errors\nA list of detected errors with their positions, types, and severity levels."),
                "parse_mode": "str"
            },
            {
                "name": "correction_generation_agent",
                "description": "This agent generates a corrected version of the Chinese sentence based on detected errors and provides detailed edit operations.",
                "inputs": [
                    {"name": "source", "type": "string", "required": True, "description": "The original Chinese sentence with possible grammar errors."},
                    {"name": "detected_errors", "type": "object", "required": True, "description": "A list of detected errors with their positions, types, and severity levels."}
                ],
                "outputs": [
                    {"name": "corrected_sentence", "type": "string", "required": True, "description": "The corrected version of the original sentence."},
                    {"name": "edit_operations", "type": "object", "required": True, "description": "A list of edit operations showing the original and corrected content, along with their positions and error details."}
                ],
                "prompt": None,
                "prompt_template": StringTemplate(instruction="### Objective\nGenerate a corrected version of the Chinese sentence based on the detected errors and provide detailed edit operations.\n\n### Instructions\n1. Read the user's goal: <input>{goal}</input>\n2. Analyze the original sentence: <input>{source}</input>\n3. Review the detected errors: <input>{detected_errors}</input>\n4. Apply corrections to the sentence according to the detected errors, ensuring the original meaning is preserved.\n5. Create a list of edit operations that describe each correction, including:\n   - The original erroneous string (src_content)\n   - The corrected string (tgt_content)\n   - The start and end positions of the error in the source (src_interval)\n   - The start and end positions of the correction in the target (tgt_interval)\n   - Tokenization results for both the original and corrected strings (src_tokens, tgt_tokens)\n   - The type of error (error_type)\n   - The severity of the error (error_severity)\n   - A detailed explanation of the error and its correction (error_description)\n6. Output the corrected sentence and the list of edit operations in the specified JSON format.\n\n### Output Format\nYour final output should ALWAYS in the following format:\n\n## Thought\nBriefly explain the reasoning process for generating the corrected sentence and edit operations.\n\n## corrected_sentence\nThe corrected version of the original sentence.\n\n## edit_operations\nA list of edit operations showing the original and corrected content, along with their positions and error details."),
                "parse_mode": "str"
            },
            {
                "name": "explanation_generation_agent",
                "description": "This agent generates detailed explanations for each detected error in a Chinese sentence, including the error type, severity, and correction explanation.",
                "inputs": [
                    {"name": "source", "type": "string", "required": True, "description": "The original Chinese sentence with possible grammar errors."},
                    {"name": "corrected_sentence", "type": "string", "required": True, "description": "The corrected version of the original sentence."},
                    {"name": "edit_operations", "type": "object", "required": True, "description": "A list of edit operations showing the original and corrected content, along with their positions and error details."}
                ],
                "outputs": [
                    {"name": "final_output", "type": "object", "required": True, "description": "A JSON object containing the corrected sentence and detailed explanations for each error."}
                ],
                "prompt": None,
                "prompt_template": StringTemplate(instruction="### Objective\nGenerate detailed explanations for each detected error in the given Chinese sentence, including the error type, severity, and correction explanation.\n\n### Instructions\n1. Read the user's goal: <input>{goal}</input>\n2. Analyze the original sentence: <input>{source}</input>\n3. Review the corrected sentence: <input>{corrected_sentence}</input>\n4. Examine the list of edit operations: <input>{edit_operations}</input>\n5. For each edit operation, identify the error type, severity, and provide a clear explanation of the correction.\n6. Format the final output as a JSON object with the corrected sentence and a list of explanations for each error.\n\n### Output Format\nYour final output should ALWAYS in the following format:\n\n## Thought\nBriefly explain the reasoning process for generating the explanations.\n\n## final_output\nA JSON object containing the corrected sentence and detailed explanations for each error."),
                "parse_mode": "str"
            }
        ]
    )
    agent_manager = AgentManager()
    agent_manager.add_agents_from_workflow(workflow_graph, executor_llm.config)

    evaluator = MyEvaluator(
        llm=executor_llm, 
        agent_manager=agent_manager, 
        collate_func=collate_func,
        output_postprocess_func=output_postprocess_func,
        verbose=False,
        num_workers=20
    )

    textgrad_optimizer = MyTextGradOptimizer(
        graph=workflow_graph, 
        optimize_mode="all",
        executor_llm=executor_llm, 
        optimizer_llm=optimizer_llm,
        batch_size=5,
        max_steps=10,
        evaluator=evaluator,
        eval_every_n_steps=1,
        eval_rounds=3,
        save_interval=1,
        save_path="./textgrad",
        rollback=True,
        constraints=[]
    )

    score_calculator = MyScoreCalculator()

    logger.info("Evaluating workflow on test set...")
    with suppress_logger_info():
        results = textgrad_optimizer.evaluate(dataset=benchmark, eval_mode="test")
    logger.info(f"Test results: {score_calculator.calculate(results)}\n\n{results}")

    logger.info("Optimizing workflow...")
    textgrad_optimizer.optimize(benchmark, seed=8)
    textgrad_optimizer.restore_best_graph()

    logger.info("Evaluating workflow on test set...")
    with suppress_logger_info():
        results = textgrad_optimizer.evaluate(dataset=benchmark, eval_mode="test")
    logger.info(f"Test results: {score_calculator.calculate(results)}\n\n{results}")

if __name__ == "__main__":
    main()