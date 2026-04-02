import os 
from dotenv import load_dotenv
from typing import Any, Callable 

from evoagentx.core.logging import logger
from evoagentx.models import AliyunLLM, AliyunLLMConfig

from eval.excgec import EXCGEC
from eval.my_evaluator import MyEvaluator
from optimizers.my_aflow_optimizer import MyAFlowOptimizer

load_dotenv()
my_api_key = os.getenv("DASHSCOPE_API_KEY")

EXPERIMENTAL_CONFIG = {
    "excgec": {
        "question_type": "qa", 
        "operators": ["Custom"]
    }
}

class ExcgecForAFlow(EXCGEC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _load_data(self):
        # load the original test data 
        super()._load_data()
        # split the data into train, dev and test
        import numpy as np 
        np.random.seed(42)
        permutation = np.random.permutation(len(self._test_data))
        full_test_data = self._test_data

        self._train_data = [full_test_data[idx] for idx in permutation[:100]]
        self._dev_data = [full_test_data[idx] for idx in permutation[100:600]]
        self._test_data = [full_test_data[idx] for idx in permutation[600:1100]]

    async def async_evaluate(self, graph: Callable, example: Any) -> float:
        problem = example["source"]
        label = example
        
        try:
            # 需重点关注
            raw_output = await graph(problem)
        except Exception as e:
            logger.error(f"Graph execution failed: {e}")
            return 0.0
        
        # 确保工作流输出符合格式要求
        output = None
        if isinstance(raw_output, dict):
            output = raw_output
        elif isinstance(raw_output, str):
            try:
                import json

                output = json.loads(raw_output)
                if not isinstance(output, dict):
                    output = {"target": str(output), "edits": []}
            except json.JSONDecodeError:
                output = {"target": raw_output, "edits": []}
        else:
            output = {"target": str(raw_output), "edits": []}
        
        if "index" not in output:
            output["index"] = example.get("index", 0)
        if "domain" not in output:
            output["domain"] = example.get("domain", "")
        if "source" not in output:
            output["source"] = problem
        
        try:
            metrics = super().evaluate(prediction=output, label=label)
            print(f"metrics: {metrics}")
        except Exception as e:
            logger.error(f"Superclass evaluation failed: {e}")
            return 0.0
        
        return metrics

def main():
    import shutil
    shutil.rmtree("output/aflow/optimized", ignore_errors=True)

    llm_config = AliyunLLMConfig(
        model="qwen-turbo", 
        aliyun_api_key=my_api_key, 
        temperature=0.1,
        max_tokens=16000
    )
    optimizer_llm = AliyunLLM(config=llm_config)
    executor_llm = AliyunLLM(config=llm_config)

    benchmark = ExcgecForAFlow()
    # create optimizer
    optimizer = MyAFlowOptimizer(
        graph_path = "aflow",
        optimized_path = "output/aflow/optimized",
        optimizer_llm=optimizer_llm,
        executor_llm=executor_llm,
        validation_rounds=2,
        eval_rounds=2,
        max_rounds=10,
        **EXPERIMENTAL_CONFIG["excgec"]
    )

    # run optimization
    optimizer.optimize(benchmark)

    # run test 
    optimizer.test(benchmark) # use `test_rounds: List[int]` to specify the rounds to test 

if __name__ == "__main__":
    main() 