import os
import json
from typing import Iterator, List, Optional, Tuple, Union

from tqdm import tqdm
import textgrad as tg
from textgrad import Variable

from evoagentx.workflow import WorkFlowGraph
from evoagentx.optimizers import TextGradOptimizer
from evoagentx.core.logging import logger
from evoagentx.core.callbacks import suppress_logger_info

from my_score_calculator import MyScoreCalculator

class MyTextGradOptimizer(TextGradOptimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.score_calculator = MyScoreCalculator()
    
    def optimize(self, dataset, use_answers: bool = True, seed: Optional[int] = None) -> None:
        """修改，现使用加权分数"""
        self._init_textgrad(dataset, use_answers)
 
        def iterator() -> Iterator[Tuple[List[dict[str, str]],  Optional[List[Union[str, dict[str, str]]]]]]:
            epoch = 0
            while True:
                # Shuffle train data every epoch
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

        for step in tqdm(range(self.max_steps)):
            inputs, labels = next(data_iterator)
            self.step(inputs, labels, dataset, use_answers)

            if self.eval_every_n_steps is not None and (step + 1) % self.eval_every_n_steps == 0:
                logger.info(f"Evaluating the workflow at step {step+1} ...")
                with suppress_logger_info():
                    metrics = self.evaluate(dataset, **self.eval_config)
                self.log_snapshot(self.graph, metrics)
                logger.info(f"Step {step+1} metrics: {metrics}")

                current_score = self.score_calculator.calculate(metrics)
                logger.info(f"Step {step+1} score: {current_score}")

                # 代码修改处
                if self.rollback:
                    if len(self._snapshot) == 1:
                        best_snapshot = self._snapshot[-1]
                        best_score = current_score
                    else:
                        if current_score >= best_score:
                            best_snapshot = self._snapshot[-1]
                            best_score = current_score
                        else:
                            logger.info(f"Metrics are worse than the best snapshot. Rolling back.")
                            best_graph = WorkFlowGraph.from_dict(best_snapshot["graph"])
                            self.graph = best_graph
                            self._create_textgrad_agents()

            if self.save_interval is not None and (step + 1) % self.save_interval == 0:
                logger.info(f"Saving the workflow at step {step+1} ...")
                self.save(os.path.join(self.save_path, f"{dataset.name}_textgrad_step_{step+1}.json"))

        logger.info(f"Reached the maximum number of steps {self.max_steps}. Optimization has finished.")
        self.save(os.path.join(self.save_path, f"{dataset.name}_textgrad_final.json"))

        # Saves the best graph
        if len(self._snapshot) > 0:
            best_graph = self._select_graph_with_highest_score()
            self.save(os.path.join(self.save_path, f"{dataset.name}_textgrad_best.json"), graph=best_graph)
    
    def _select_graph_with_highest_score(self, return_metrics: bool = False):
        """修改，现使用加权分数"""
        if len(self._snapshot) == 0:
            if return_metrics:
                return self.graph, None
            return self.graph
        
        best_score = -float('inf')
        best_graph = None
        best_metrics = None
        
        for snapshot in self._snapshot:
            metrics = snapshot["metrics"]
            # 代码修改处
            score = self.score_calculator.calculate(metrics)
            
            if score > best_score:
                best_score = score
                best_graph = WorkFlowGraph.from_dict(snapshot["graph"])
                best_metrics = metrics
        
        if return_metrics:
            return best_graph, best_metrics
        return best_graph

    def step(self, inputs, labels, dataset, use_answers):
        """对于非编程类的任务，源代码仅支持字符串格式的 label，故修改"""
        losses = []
        if use_answers:
            if labels is None:
                raise ValueError("Labels must be provided if `use_answers` is True.")

            for input, label in zip(inputs, labels, strict=True):
                output = self.forward(input)
                
                # 代码修改处
                if isinstance(label, str):
                    label_var = Variable(label, requires_grad=False, role_description="correct answer for the query")
                elif isinstance(label, dict):
                    label_str = json.dumps(label, ensure_ascii=False)
                    label_var = Variable(label_str, requires_grad=False, role_description="correct answer for the query (converted from dict)")
                
                loss = self.loss_fn([output, label_var])
                losses.append(loss)
        else:
            for input in inputs:
                output = self.forward(input)
                loss = self.loss_fn(output)
                losses.append(loss)

        total_loss = tg.sum(losses)
        total_loss.backward(self.optimizer_engine)
        self.textgrad_optimizer.step()
        self.textgrad_optimizer.zero_grad()
        self._update_workflow_graph()