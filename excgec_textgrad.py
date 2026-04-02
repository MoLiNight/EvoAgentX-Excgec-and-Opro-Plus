import os
import json
from dotenv import load_dotenv

from evoagentx.models import AliyunLLM, AliyunLLMConfig
from evoagentx.core.logging import logger
from evoagentx.core.callbacks import suppress_logger_info
from evoagentx.workflow import WorkFlowGraph
from evoagentx.agents.agent_manager import AgentManager

from my_graph import MY_WORKFLOW_GRAPH

from eval.excgec import EXCGEC
from eval.my_evaluator import MyEvaluator
from eval.my_score_calculator import MyScoreCalculator
from optimizers.my_textgrad_optimizer import MyTextGradOptimizer

load_dotenv()
my_api_key = os.getenv("DASHSCOPE_API_KEY")

class ExcgecForTextgrad(EXCGEC):
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

    benchmark = ExcgecForTextgrad(eval_mode="all")
    workflow_graph = MY_WORKFLOW_GRAPH
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
        batch_size=10,
        max_steps=10,
        evaluator=evaluator,
        eval_every_n_steps=1,
        eval_rounds=2,
        save_interval=1,
        save_path="output/textgrad",
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