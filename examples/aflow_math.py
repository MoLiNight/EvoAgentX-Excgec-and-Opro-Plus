import os 
from dotenv import load_dotenv
from typing import Any, Callable 

from evoagentx.benchmark import MATH
from evoagentx.optimizers import AFlowOptimizer
from evoagentx.models import AliyunLLM, AliyunLLMConfig

load_dotenv()
my_api_key = os.getenv("DASHSCOPE_API_KEY")

EXPERIMENTAL_CONFIG = {
    "humaneval": {
        "question_type": "code", 
        "operators": ["Custom", "CustomCodeGenerate", "Test", "ScEnsemble"] 
    }, 
    "mbpp": {
        "question_type": "code", 
        "operators": ["Custom", "CustomCodeGenerate", "Test", "ScEnsemble"] 
    },
    "hotpotqa": {
        "question_type": "qa", 
        "operators": ["Custom", "AnswerGenerate", "QAScEnsemble"]
    },
    "gsm8k": {
        "question_type": "math", 
        "operators": ["Custom", "ScEnsemble", "Programmer"]
    },
    "math": {
        "question_type": "math", 
        "operators": ["Custom", "ScEnsemble", "Programmer"]
    }
    
}

class MathSplits(MATH):

    def _load_data(self):
        # load the original test data 
        super()._load_data()
        # split the data into dev and test
        import numpy as np 
        np.random.seed(42)
        permutation = np.random.permutation(len(self._test_data))
        full_test_data = self._test_data
        # radnomly select 50 samples for dev and 100 samples for test
        self._dev_data = [full_test_data[idx] for idx in permutation[:20]]
        self._test_data = [full_test_data[idx] for idx in permutation[20:50]]
    
    async def async_evaluate(self, graph: Callable, example: Any) -> float:

        problem = example["problem"]
        label = self._get_label(example)
        output = await graph(problem)
        metrics = await super().async_evaluate(prediction=output, label=label)
        return metrics["solve_rate"]

def main():

    claude_config = AliyunLLMConfig(model="qwen-turbo", aliyun_api_key=my_api_key)
    optimizer_llm = AliyunLLM(config=claude_config)
    openai_config = AliyunLLMConfig(model="qwen-turbo", aliyun_api_key=my_api_key)
    executor_llm = AliyunLLM(config=openai_config)

    # load benchmark
    math = MathSplits()

    # create optimizer
    optimizer = AFlowOptimizer(
        graph_path = "aflow",
        optimized_path = "aflow/optimized",
        optimizer_llm=optimizer_llm,
        executor_llm=executor_llm,
        validation_rounds=2,
        eval_rounds=2,
        max_rounds=10,
        **EXPERIMENTAL_CONFIG["math"]
    )

    # run optimization
    optimizer.optimize(math)

    # run test 
    optimizer.test(math) # use `test_rounds: List[int]` to specify the rounds to test 

if __name__ == "__main__":
    main() 