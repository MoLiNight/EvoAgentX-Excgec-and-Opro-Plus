import json
import asyncio

from typing import Tuple
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
from typing import Tuple, Callable

from evoagentx.benchmark.benchmark import Benchmark
from evoagentx.optimizers import AFlowOptimizer
from evoagentx.core.logging import logger
from evoagentx.models.model_utils import cost_manager
from evoagentx.core.callbacks import suppress_logger_info
from evoagentx.evaluators.aflow_evaluator import AFlowEvaluator
from evoagentx.utils.aflow_utils.evaluation_utils import EvaluationUtils

from eval.my_evaluator import MyEvaluator
from eval.my_score_calculator import MyScoreCalculator

class MyAFlowEvaluator(AFlowEvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    async def graph_evaluate_async(self, benchmark: Benchmark, graph: Callable, is_test: bool = False, max_concurrent_tasks: int = 20) -> Tuple[float, float, float]:
        configured_graph = self._configure_graph(graph=graph, benchmark=benchmark)

        # Get evaluation data
        data = benchmark.get_test_data() if is_test else benchmark.get_dev_data()
        if not data:
            logger.warning("No data to evaluate. Returning zeros.")
            return (0.0, 0.0, 0.0, True)
        
        # get total cost before evaluation
        cost_before = cost_manager.get_total_cost()
        
        # Create a shared semaphore
        semaphore = asyncio.Semaphore(max_concurrent_tasks)
        
        async def evaluate_with_semaphore(example):
            async with semaphore:
                try:
                    return await benchmark.async_evaluate(configured_graph, example)
                except Exception as e:
                    logger.warning(f"Evaluation failed: {str(e)}")
                    return None
        
        # 代码修改处
        tasks = [evaluate_with_semaphore(example) for example in data]

        # Wait for all tasks to complete
        metrics = await tqdm_asyncio.gather(
            *tasks,
            desc=f"Evaluating {benchmark.name} problems",
            total=len(data)
        )

        # 代码修改处
        score_calculator = MyScoreCalculator()
        results = [score_calculator.calculate(metric) for metric in metrics]

        my_evaluator = MyEvaluator(llm=self.llm, num_workers=20)
        print(f"Average Metrics: {my_evaluator._calculate_average_score(metrics)}")

        # Replace failed evaluations (None results) with 0
        valid_results = [0.0 if r is None else r for r in results]
        all_failed = all(r is None for r in results)

        # get total cost after evaluation
        total_cost = cost_manager.get_total_cost() - cost_before
        avg_cost = total_cost / len(data)

        if not valid_results:
            logger.warning("No valid results. Returning zeros.")
            avg_metrics = 0.0
        else:
            avg_metrics = sum(valid_results) / len(valid_results)
        
        return avg_metrics, avg_cost, total_cost, all_failed 

class MyEvaluationUtils(EvaluationUtils):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    async def evaluate_graph_async(self, optimizer, validation_n, data, initial=False):
        # 代码修改处
        evaluator = MyAFlowEvaluator(llm=optimizer.executor_llm)
        sum_score = 0
        
        for _ in range(validation_n):

            with suppress_logger_info():
                score, avg_cost, total_cost, all_failed = await evaluator.graph_evaluate_async(optimizer.benchmark, optimizer.graph, is_test=False)
            cur_round = optimizer.round + 1 if initial is False else optimizer.round 
            new_data = optimizer.data_utils.create_result_data(cur_round, score, avg_cost, total_cost)
            data.append(new_data)

            result_path = optimizer.data_utils.get_results_file_path(self.root_path)
            optimizer.data_utils.save_results(result_path, data)
            
            sum_score += score

            if all_failed:
                logger.warning(f"All test cases failed in round {cur_round}. Stopping evaluation for this round.")
                break 
            
        return sum_score / validation_n

    async def evaluate_graph_test_async(self, optimizer):
        # 代码修改处
        evaluator = MyAFlowEvaluator(llm=optimizer.executor_llm)
        with suppress_logger_info():
            score, avg_cost, total_cost, all_failed = await evaluator.graph_evaluate_async(optimizer.benchmark, optimizer.graph, is_test=True)
        return score, avg_cost, total_cost

class MyAFlowOptimizer(AFlowOptimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.evaluation_utils = MyEvaluationUtils(self.root_path)
    