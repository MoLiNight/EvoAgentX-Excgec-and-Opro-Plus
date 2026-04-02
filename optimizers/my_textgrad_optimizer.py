import os
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import hashlib
from typing import Iterator, List, Optional, Tuple, Union, Any
import numpy as np
from copy import deepcopy

from tqdm import tqdm
import textgrad as tg
from textgrad import Variable

from evoagentx.workflow import WorkFlowGraph
from evoagentx.optimizers import TextGradOptimizer
from evoagentx.core.logging import logger
from evoagentx.benchmark.benchmark import Benchmark
from evoagentx.core.callbacks import suppress_logger_info

from eval.my_score_calculator import MyScoreCalculator

class MyTextGradOptimizer(TextGradOptimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache = {}
        self.score_calculator = MyScoreCalculator()   

    def init_module(self, **kwargs):
        super().init_module(**kwargs)
        self._max_workers = multiprocessing.cpu_count()
        logger.info(f"MyTextGradOptimizer initialized with {self._max_workers} workers")
    
    def _forward_with_cache(self, input: dict[str, str], label: Optional[Union[str, dict[str, str]]], 
                            use_answers: bool, dataset: Benchmark = None) -> Any:
        """带缓存的 forward 调用，避免重复计算相同输入"""
        cache_key = hashlib.md5(str(input).encode()).hexdigest()
        
        if cache_key in self._cache:
            logger.debug(f"Cache hit for input: {cache_key}")
            output = self._cache[cache_key]
        else:
            output = self.forward(input)
            self._cache[cache_key] = output
        
        if use_answers:
            if isinstance(label, str):
                label_var = Variable(label, requires_grad=False, 
                                   role_description="correct answer for the query")
            elif isinstance(label, dict):
                label_str = json.dumps(label, ensure_ascii=False)
                label_var = Variable(label_str, requires_grad=False, 
                                   role_description="correct answer for the query (converted from dict)")
            else:
                label_var = label
            
            loss = self.loss_fn([output, label_var])
        else:
            loss = self.loss_fn(output)
        
        return loss
    
    def step(self, inputs, labels, dataset, use_answers):
        """重写 step 方法，添加并行 forward 调用"""
        losses = []
        
        if use_answers:
            if labels is None:
                raise ValueError("Labels must be provided if `use_answers` is True.")

            # 并行处理 forward 调用
            with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
                futures = []
                for input, label in zip(inputs, labels, strict=True):
                    future = executor.submit(self._forward_with_cache, input, label, use_answers, dataset)
                    futures.append(future)
                
                for future in futures:
                    loss = future.result()
                    losses.append(loss)
        else:
            with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
                futures = []
                for input in inputs:
                    future = executor.submit(self._forward_with_cache, input, None, use_answers, dataset)
                    futures.append(future)
                
                for future in futures:
                    loss = future.result()
                    losses.append(loss)

        total_loss = tg.sum(losses)
        total_loss.backward(self.optimizer_engine)
        self.textgrad_optimizer.step()
        self.textgrad_optimizer.zero_grad()
        self._update_workflow_graph()
    
    async def _forward_async(self, input: dict[str, str], label: Optional[Union[str, dict[str, str]]], 
                             use_answers: bool, dataset: Benchmark = None) -> Any:
        """异步版本的 forward 调用"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._forward_with_cache, input, label, use_answers, dataset)

    async def step_async(self, inputs, labels, dataset, use_answers):
        """异步版本的 step 方法"""
        tasks = [self._forward_async(input, label, use_answers, dataset) 
                 for input, label in zip(inputs, labels)]
        losses = await asyncio.gather(*tasks)
        
        total_loss = tg.sum(losses)
        total_loss.backward(self.optimizer_engine)
        self.textgrad_optimizer.step()
        self.textgrad_optimizer.zero_grad()
        self._update_workflow_graph()
    
    def evaluate(
        self, 
        dataset: Benchmark, 
        eval_mode: str = "dev", 
        graph: Optional[WorkFlowGraph] = None,
        indices: Optional[List[int]] = None,
        sample_k: Optional[int] = None,
        **kwargs
    ) -> dict:
        """重写 evaluate 方法，添加并行评估"""
        if graph is None:
            graph = self.graph

        def evaluate_single(round_num: int) -> dict:
            """单个评估任务"""
            eval_info = [
                f"[{type(graph).__name__}]", 
                f"Evaluation round {round_num+1}/{self.eval_rounds}", 
                f"Mode: {eval_mode}"
            ]
            if indices is not None:
                eval_info.append(f"Indices: {len(indices)} samples")
            if sample_k is not None:
                eval_info.append(f"Sample size: {sample_k}")
            logger.info(" | ".join(eval_info))
            
            metrics = self.evaluator.evaluate(
                graph=graph, 
                benchmark=dataset, 
                eval_mode=eval_mode, 
                indices=indices, 
                sample_k=sample_k,
                update_agents=True, 
                **kwargs
            )
            return metrics

        metrics_list = []
        # 并行执行多个评估轮次
        with ThreadPoolExecutor(max_workers=min(self._max_workers, self.eval_rounds)) as executor:
            futures = [executor.submit(evaluate_single, i) for i in range(self.eval_rounds)]
            for future in futures:
                metrics_list.append(future.result())

        avg_metrics = self.evaluator._calculate_average_score(metrics_list)
        return avg_metrics
    
    def optimize(self, dataset: Benchmark, use_answers: bool = True, seed: Optional[int] = None) -> None:
        """重写 optimize 方法，使用加权分数进行回滚"""
        self._init_textgrad(dataset, use_answers)
 
        def iterator() -> Iterator[Tuple[List[dict[str, str]], Optional[List[Union[str, dict[str, str]]]]]]:
            epoch = 0
            while True:
                effective_seed = seed + epoch if seed is not None else None
                train_data = dataset.get_train_data(sample_k=len(dataset._train_data), seed=effective_seed)
                for i in range(0, len(train_data), self.batch_size):
                    batch = train_data[i:i + self.batch_size]
                    inputs = [self.evaluator.collate_func(x) for x in batch]
                    if use_answers:
                        labels = dataset.get_labels(batch)
                    else:
                        labels = None
                    yield inputs, labels
                epoch += 1

        data_iterator = iterator()

        # 记录最佳快照
        best_score = -float('inf')
        best_graph = None
        best_snapshot = None

        for step in tqdm(range(self.max_steps)):
            inputs, labels = next(data_iterator)
            
            # 并行处理
            self._process_batch_parallel(inputs, labels, dataset, use_answers)

            # 评估
            if self.eval_every_n_steps is not None and (step + 1) % self.eval_every_n_steps == 0:
                logger.info(f"Evaluating the workflow at step {step+1} ...")
                with suppress_logger_info():
                    metrics = self.evaluate(dataset, **self.eval_config)
                self.log_snapshot(self.graph, metrics)
                
                # 计算加权分数
                current_score = self.score_calculator.calculate(metrics)
                logger.info(f"Step {step+1} metrics: {metrics}")
                logger.info(f"Step {step+1} weighted score: {current_score:.6f}")
                
                # 更新最佳快照
                if current_score > best_score:
                    best_score = current_score
                    best_graph_config = deepcopy(self.graph.get_config())
                    best_graph = WorkFlowGraph.from_dict(best_graph_config)
                    best_snapshot = self._snapshot[-1] if self._snapshot else None
                    logger.info(f"New best score: {best_score:.6f}")

                # 回滚优化：性能下降时回滚
                if self.rollback and best_graph is not None and current_score < best_score * 0.95:
                    logger.info(f"Performance degraded from {best_score:.6f} to {current_score:.6f}. Rolling back.")
                    self.graph = best_graph
                    self._create_textgrad_agents()

            # 保存中间结果
            if self.save_interval is not None and (step + 1) % self.save_interval == 0:
                logger.info(f"Saving the workflow at step {step+1} ...")
                self.save(os.path.join(self.save_path, f"{dataset.name}_textgrad_step_{step+1}.json"))

        logger.info(f"Reached maximum steps {self.max_steps}. Optimization finished.")
        self.save(os.path.join(self.save_path, f"{dataset.name}_textgrad_final.json"))

        # 保存最佳快照
        if best_graph is not None:
            self.save(os.path.join(self.save_path, f"{dataset.name}_textgrad_best.json"), graph=best_graph)
            logger.info(f"Best score: {best_score:.6f}")
    
    def _process_batch_parallel(self, inputs, labels, dataset, use_answers):
        """并行处理多个 batch"""
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures = []
            for i in range(0, len(inputs), self.batch_size):
                batch_inputs = inputs[i:i+self.batch_size]
                batch_labels = labels[i:i+self.batch_size] if labels else None
                future = executor.submit(self.step, batch_inputs, batch_labels, dataset, use_answers)
                futures.append(future)
            for future in futures:
                future.result()
    
    def _select_graph_with_highest_score(self, return_metrics: bool = False):
        """重写，使用加权分数选择最佳快照"""
        if len(self._snapshot) == 0:
            if return_metrics:
                return self.graph, None
            return self.graph
        
        best_score = -float('inf')
        best_graph = None
        best_metrics = None
        
        for snapshot in self._snapshot:
            metrics = snapshot["metrics"]
            score = self.score_calculator.calculate(metrics)
            
            if score > best_score:
                best_score = score
                best_graph = WorkFlowGraph.from_dict(snapshot["graph"])
                best_metrics = metrics
        
        logger.info(f"Selected best graph with score: {best_score:.6f}")
        
        if return_metrics:
            return best_graph, best_metrics
        return best_graph
    
    def clear_cache(self):
        self._cache.clear()
        logger.info("Cache cleared")

    def get_cache_stats(self) -> dict:
        return {"cache_size": len(self._cache)}