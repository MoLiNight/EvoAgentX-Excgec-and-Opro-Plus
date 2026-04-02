import collections
import json
import os
import re
from typing import List, Dict, Any, Callable, Optional, Iterator, Tuple, Union
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from pydantic import Field, PositiveInt

from evoagentx.core.module import BaseModule
from evoagentx.core.logging import logger
from evoagentx.core.callbacks import suppress_logger_info
from evoagentx.models.base_model import BaseLLM
from evoagentx.evaluators import Evaluator
from evoagentx.benchmark.benchmark import Benchmark
from evoagentx.workflow import WorkFlowGraph
from evoagentx.prompts import StringTemplate
from evoagentx.agents.agent_manager import AgentManager
from evoagentx.workflow.workflow_graph import SequentialWorkFlowGraph

from eval.my_score_calculator import MyScoreCalculator

def _bucketize_float(num: float, n_buckets: int = 20) -> int:
    """Convert a float in [0,1] to an integer bucket."""
    assert 0 <= num <= 1, "Number must be between 0 and 1."
    return round(num * n_buckets)

def gen_ins_and_score_pairs_substr(
    old_instructions_and_scores: List[tuple],
    old_instruction_score_threshold: float = 0.1,
    max_num_instructions: int = 1000,
    return_str_only: bool = False,
    num_score_buckets: float = np.inf,
) -> Union[str, tuple]:
    """Generate string containing instruction-score pairs for meta-prompt."""
    assert num_score_buckets == np.inf or isinstance(num_score_buckets, int)
    
    old_instructions_and_scores_str = ""
    sorted_instructions = sorted(old_instructions_and_scores, key=lambda x: x[1])[-max_num_instructions:]
    instructions_in_meta_prompt = []
    
    for instruction, score, i_step in sorted_instructions:
        if not old_instruction_score_threshold or score >= old_instruction_score_threshold:
            instructions_in_meta_prompt.append((instruction, score, i_step))
            if num_score_buckets == np.inf:
                score_to_show = round(score, 3)
            else:
                score_to_show = _bucketize_float(score, num_score_buckets)
            old_instructions_and_scores_str += f"\ntext:\n{instruction}\nscore:\n{score_to_show}\n"
    
    if return_str_only:
        return old_instructions_and_scores_str
    return old_instructions_and_scores_str, instructions_in_meta_prompt

def polish_sentence(sentence: str, add_ending_punc: bool = False) -> str:
    """Standardize sentence format."""
    sentence = sentence.strip()
    if sentence:
        sentence = sentence.replace("**", "")
        if len(sentence) > 1:
            sentence = sentence[0].upper() + sentence[1:]
        if add_ending_punc and not sentence[-1] in {".", "?", "!"}:
            sentence += "."
    return sentence

def instruction_to_filename(instruction: str, md5_hashing: bool = True) -> str:
    """Convert instruction to filename using MD5 hash."""
    if md5_hashing:
        import hashlib
        return hashlib.md5(instruction.encode("utf-8")).hexdigest()
    
    filename = re.sub(r'[^\w\s]', '', instruction).replace(' ', '_')
    return filename[:100] if filename else "empty_instruction"

def gen_meta_prompt(
    old_instructions_and_scores: List[tuple],
    instruction_pos: str,
    old_instruction_score_threshold: float = 0.1,
    max_num_instructions: int = 1000,
    meta_prompt_type: str = "both_instructions_and_exemplars",
    few_shot_qa_pairs: bool = False,
    include_qa: bool = True,
    data: Optional[list] = None,
    few_shot_index_list: Optional[List[int]] = None,
    instructions_before_exemplars: bool = True,
    num_score_buckets: float = np.inf,
    dataset_name: str = "",
    task_name: str = "",
) -> str:
    """Generate meta-prompt for instruction rewriting."""
    assert instruction_pos in {"before_Q", "Q_begin", "Q_end", "A_begin"}
    assert meta_prompt_type in {"both_instructions_and_exemplars", "instructions_only"}
    assert num_score_buckets == np.inf or isinstance(num_score_buckets, int)

    meta_prompt = ""
    
    if meta_prompt_type == "both_instructions_and_exemplars":
        # Build old instructions section
        if instruction_pos == "A_begin":
            meta_prompt_old_instruction_part = (
                "Your task is to generate the answer starting sentence <Start>."
                " Below are some previous starting sentences with their scores."
                " The score ranges from 0 to 100.\n"
            )
        else:
            meta_prompt_old_instruction_part = (
                "Your task is to generate the instruction <INS>."
                " Below are some previous instructions with their scores."
                " The score ranges from 0 to 100.\n"
            )
        
        old_instructions_str = gen_ins_and_score_pairs_substr(
            old_instructions_and_scores=old_instructions_and_scores,
            old_instruction_score_threshold=old_instruction_score_threshold,
            max_num_instructions=max_num_instructions,
            return_str_only=True,
            num_score_buckets=num_score_buckets,
        )
        meta_prompt_old_instruction_part += old_instructions_str
        
        # Build few-shot examples section
        meta_prompt_exemplar_part = ""
        if few_shot_qa_pairs and few_shot_index_list and data:
            meta_prompt_exemplar_part += "Below are some problems.\n"
            
            for idx in few_shot_index_list:
                if dataset_name == "excgec":
                    if isinstance(data, list):
                        question = data[idx].get("source", "")
                        true_answer = data[idx].get("target", "")
                    else:
                        question = data.iloc[idx]["source"]
                        true_answer = data.iloc[idx]["target"]
                
                if include_qa:
                    if instruction_pos == "before_Q":
                        meta_prompt_exemplar_part += f"\ninput:\n<INS>\nQ: {question}\nA:"
                    elif instruction_pos == "Q_begin":
                        meta_prompt_exemplar_part += f"\ninput:\nQ: <INS>\n{question}\nA:"
                    elif instruction_pos == "Q_end":
                        meta_prompt_exemplar_part += f"\ninput:\nQ: {question}\n<INS>\nA:"
                    else:  # instruction_pos == "A_begin"
                        meta_prompt_exemplar_part += f"\nQ: {question}\nA: <Start>"
                else:
                    assert instruction_pos in {"Q_begin", "Q_end"}
                    if instruction_pos == "Q_begin":
                        meta_prompt_exemplar_part += f"\nProblem:\n<INS>\n{question}\n"
                    else:
                        meta_prompt_exemplar_part += f"\nProblem:\n{question}\n<INS>\n"
                
                meta_prompt_exemplar_part += f"\nGround truth answer:\n{true_answer}\n"
        
        # Combine sections
        if few_shot_qa_pairs:
            if instructions_before_exemplars:
                meta_prompt = meta_prompt_old_instruction_part + "\n\n" + meta_prompt_exemplar_part
            else:
                meta_prompt = meta_prompt_exemplar_part + "\n\n" + meta_prompt_old_instruction_part
        else:
            meta_prompt = meta_prompt_old_instruction_part
        
        # Add generation instruction
        if instruction_pos == "A_begin":
            meta_prompt += (
                "\n\nGenerate a starting sentence that is different from all the"
                " <Start> sentences above, and has a higher score than all the"
                " <Start> sentences above. The starting sentence should be a direct"
                " instruction to the model. It MUST include:"
                "\n- The input format (source sentence)"
                "\n- The output format (JSON with target and edits fields)"
                "\n- The list of error types to choose from"
                "\nThe instruction should be concise but complete, containing all the"
                " necessary format specifications."
                " It should begin with <Start> and end with </Start>."
            )
        else:
            meta_prompt += (
                "\n\nGenerate an instruction that is different from all the"
                " instructions <INS> above, and has a higher score than all the"
                " instructions <INS> above. The instruction should be a direct"
                " instruction to the model. It MUST include:"
                "\n- The input format (source sentence)"
                "\n- The output format (JSON with target and edits fields)"
                "\n- The list of error types to choose from"
                "\nThe instruction should be concise but complete, containing all the"
                " necessary format specifications."
                " It should begin with <INS> and end with </INS>."
            )
    else:
        # instructions_only mode
        assert meta_prompt_type == "instructions_only"
        assert instruction_pos in {"Q_begin", "Q_end", "A_begin"}
        
        pos_desc = {
            "Q_begin": "at the beginning of the question",
            "Q_end": "at the end of the question",
            "A_begin": "at the beginning of the answer",
        }
        instruction_pos_description = pos_desc[instruction_pos]
        instruction_task_description = "Chinese grammatical error correction"
        
        meta_instruction = (
            f"Create a piece of text {instruction_pos_description} to"
            " enhance the precision in solving diverse"
            f" {instruction_task_description} problems."
        )
        
        sorted_instructions = sorted(old_instructions_and_scores, key=lambda x: x[1])
        old_instructions_str = ""
        for instruction, score, _ in sorted_instructions:
            if num_score_buckets == np.inf:
                score_to_show = round(score, 2)
            else:
                score_to_show = _bucketize_float(score, num_score_buckets)
            old_instructions_str += f"\n\nPrecision: {score_to_show} <TEXT>{instruction}</TEXT>"
        
        meta_prompt = meta_instruction + old_instructions_str
    
    return meta_prompt

class MyOPROOptimizer(BaseModule):
    graph: WorkFlowGraph = Field(description="The workflow to optimize.")
    executor_llm: BaseLLM = Field(default=None, description="The LLM to use for execution.")
    optimizer_llm: BaseLLM = Field(default=None, description="The LLM to use for optimization.")
    batch_size: PositiveInt = Field(default=1, description="The batch size for optimization.")
    max_steps: PositiveInt = Field(default=10, description="The maximum number of optimization steps.")
    evaluator: Evaluator = Field(default=None, description="The evaluator to perform evaluation during optimization.")
    eval_every_n_steps: Optional[PositiveInt] = Field(default=None, description="Evaluate the workflow every `eval_every_n_steps` steps.")
    eval_rounds: PositiveInt = Field(default=1, description="The number of times to evaluate the performance.")
    eval_config: dict = Field(default={}, description="The configuration for evaluation.")
    save_interval: Optional[PositiveInt] = Field(default=None, description="Save the workflow every `save_interval` steps.")
    save_path: str = Field(default="./", description="The path to save the optimized workflow.")
    rollback: bool = Field(default=True, description="Whether to rollback to the best graph after each evaluation during optimization.")
    
    num_generated_instructions_in_each_step: PositiveInt = Field(default=6, description="Number of instructions to generate per step.")
    optimizer_llm_temperature: float = Field(default=0.7, description="Temperature for optimizer LLM.")
    old_instruction_score_threshold: float = Field(default=0.0, description="Threshold for keeping old instructions.")
    max_num_instructions: int = Field(default=20, description="Max instructions to keep in meta-prompt.")
    num_score_buckets: int = Field(default=np.inf, description="Number of score buckets for discretization.")
    
    instruction_pos: str = Field(default="before_Q", description="Position to insert instruction.")
    few_shot_qa_pairs: bool = Field(default=False, description="Whether to include few-shot examples.")
    few_shot_selection_criteria: str = Field(default="random", description="Strategy for few-shot selection.")
    num_few_shot_questions_for_instruction_refinement: int = Field(default=3, description="Number of few-shot examples.")
    meta_prompt_type: str = Field(default="both_instructions_and_exemplars", description="Type of meta-prompt: 'both_instructions_and_exemplars' or 'instructions_only'.")
    
    def init_module(self, **kwargs):
        self.score_calculator = MyScoreCalculator()

        os.makedirs(self.save_path, exist_ok=True)
        self._snapshot: List[dict] = []
        self._best_score: float = -float('inf')
        self._best_instruction: Optional[str] = None
        self._best_graph_config: Optional[dict] = None
        self._initial_instruction: str = self.graph.get_config()["goal"]
    
    def _call_optimizer_llm(self, prompt: str, **kwargs) -> List[str]:
        """Call the optimizer LLM to generate new instructions."""
        import time
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.optimizer_llm.generate(prompt=prompt, **kwargs)
                print("--------------------generate response--------------------")
                print(response)
                return [response.content if hasattr(response, 'content') else str(response)]
            except Exception as e:
                logger.warning(f"Optimizer LLM call failed (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    wait_time = 5 * (attempt + 1)
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Optimizer LLM call failed after {max_retries} attempts")
                    return []
        
        return []
    
    def create_sequential_workflow_graph(self, instruction: str):
        workflow_graph = SequentialWorkFlowGraph(
            goal=instruction,
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
        return workflow_graph

    def evaluate(self, dataset: Benchmark, instruction: str = None, eval_mode: str = "dev", sample_k: Optional[int] = None) -> float:
        
        if instruction is None:
            workflow_graph = self.graph
        else:
            workflow_graph = self.create_sequential_workflow_graph(instruction)

        agent_manager = AgentManager()
        agent_manager.add_agents_from_workflow(workflow_graph, self.executor_llm.config)

        all_metrics = []
        for round_idx in range(self.eval_rounds):
            with suppress_logger_info():
                metrics = self.evaluator.evaluate(
                    graph=workflow_graph,
                    benchmark=dataset,
                    eval_mode=eval_mode,
                    sample_k=sample_k,
                    update_agents=True,
                    **self.eval_config
                )
            all_metrics.append(metrics)

        with suppress_logger_info():
            metrics = self.evaluator.evaluate(
                graph=workflow_graph,
                benchmark=dataset,
                eval_mode=eval_mode,
                sample_k=sample_k,
                update_agents=True,
                **self.eval_config
            )

        avg_metrics = self.evaluator._calculate_average_score(all_metrics)
        return avg_metrics
    
    def log_snapshot(self, graph: WorkFlowGraph, metrics: dict) -> None:
        """Log the snapshot of the workflow."""
        self._snapshot.append(
            {
                "index": len(self._snapshot),
                "graph": deepcopy(graph.get_config()),
                "metrics": metrics,
            }
        )
    
    def optimize(self, dataset: Benchmark, use_answers: bool = True, seed: Optional[int] = None) -> dict:
        # Get training and validation data (used for sampling few-shot examples)
        train_data = dataset.get_train_data()
        val_data = dataset.get_dev_data()
        
        # Track instructions and scores
        old_instructions_and_scores: List[tuple] = []
        old_instruction_md5_set = set()
        
        # Evaluate initial instructions
        print("============== Evaluating initial instructions ===============")
        print(f'Computing score for initial_instruction')
        metrics = self.evaluate(dataset=dataset, eval_mode="test")
        score = self.score_calculator.calculate(metrics)
        logger.info(f"Test Score: {score:.4f}")
        print(metrics)

        metrics = self.evaluate(dataset=dataset, eval_mode="dev")
        score = self.score_calculator.calculate(metrics)
        old_instructions_and_scores.append((self._initial_instruction, score, -1))
        
        # Main optimization loop
        for step in tqdm(range(self.max_steps), desc="OPRO Optimization"):
            logger.info(f"\n================== Step {step} =====================")
            
            # Select few-shot samples (based on training data)
            few_shot_index_list = []
            if self.few_shot_qa_pairs and len(train_data) > 0:
                np.random.seed(seed + step if seed else step)
                k = min(self.num_few_shot_questions_for_instruction_refinement, len(train_data))
                few_shot_index_list = np.sort(np.random.choice(len(train_data), k, replace=False)).tolist()
            
            # Generate meta-prompt
            meta_prompt = gen_meta_prompt(
                old_instructions_and_scores=old_instructions_and_scores,
                instruction_pos=self.instruction_pos,
                old_instruction_score_threshold=self.old_instruction_score_threshold,
                max_num_instructions=self.max_num_instructions,
                meta_prompt_type=self.meta_prompt_type,
                few_shot_qa_pairs=self.few_shot_qa_pairs,
                include_qa=True,
                data=train_data,
                few_shot_index_list=few_shot_index_list,
                instructions_before_exemplars=True,
                num_score_buckets=self.num_score_buckets,
                dataset_name="excgec",
                task_name="",
            )
            
            # Generate new instructions
            generated_instructions_raw = []
            for _ in range(self.num_generated_instructions_in_each_step):
                raw_outputs = self._call_optimizer_llm(meta_prompt)
                generated_instructions_raw.extend(raw_outputs)
            
            # Clean and deduplicate
            generated_instructions_raw = [polish_sentence(ins) for ins in generated_instructions_raw]
            new_instructions = []
            for ins in generated_instructions_raw:
                ins_hash = instruction_to_filename(ins, md5_hashing=True)
                if ins_hash not in old_instruction_md5_set:
                    new_instructions.append(ins)
                    old_instruction_md5_set.add(ins_hash)
            
            # Filter invalid instructions
            valid_instructions = []
            for instruction in new_instructions:
                if "INS" in instruction:
                    continue
                valid_instructions.append(instruction)
            
            # Evaluate new instructions
            for instruction in valid_instructions:
                logger.info(f'Computing score for "{instruction[:50]}..."')
                metrics = self.evaluate(
                    instruction=instruction, 
                    dataset=dataset, 
                    eval_mode="dev"
                )
                score = self.score_calculator.calculate(metrics)
                logger.info(f"Step {step} score: {score:.4f}")
                print(metrics)
                old_instructions_and_scores.append((instruction, score, step))
            
            # Periodic validation on validation set
            if self.eval_every_n_steps and (step + 1) % self.eval_every_n_steps == 0 and val_data:
                logger.info(f"\n=== Evaluating on validation set at step {step+1} ===")
                best_instruction = max(old_instructions_and_scores, key=lambda x: x[1])[0]

                self.graph = self.create_sequential_workflow_graph(best_instruction)
                metrics = self.evaluate(dataset=dataset, eval_mode="dev")
                val_score = self.score_calculator.calculate(metrics)
                logger.info(f"Best instruction validation score: {val_score:.4f}")
                
                self.log_snapshot(self.graph, metrics)
                
                if self.rollback and val_score > self._best_score:
                    self._best_score = val_score
                    self._best_instruction = best_instruction
                    # 保存最佳图配置
                    if hasattr(self.graph, 'get_config'):
                        self._best_graph_config = deepcopy(self.graph.get_config())
                    logger.info(f"New best score: {self._best_score:.4f}")
                elif self.rollback and self._best_instruction:
                    logger.info(f"Current score {val_score:.4f} is worse than best {self._best_score:.4f}")
            
            # Save checkpoint
            if self.save_interval and (step + 1) % self.save_interval == 0:
                checkpoint_path = os.path.join(self.save_path, f"checkpoint_step_{step}.json")
                with open(checkpoint_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        "step": step,
                        "instructions_and_scores": old_instructions_and_scores,
                        "best_instruction": getattr(self, '_best_instruction', None),
                        "best_score": getattr(self, '_best_score', None),
                    }, f, indent=4, ensure_ascii=False)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Optimization finished
        logger.info(f"\nOptimization finished. Total instructions evaluated: {len(old_instructions_and_scores)}")
        
        # Restore best instruction
        if hasattr(self, '_best_instruction') and self._best_instruction:
            logger.info(f"Restoring best instruction: {self._best_instruction[:100]}...")
            if hasattr(self.graph, 'prompt_template'):
                self.graph.prompt_template = self._best_instruction
        
        # Save final results
        final_results = {
            "instructions_and_scores": old_instructions_and_scores,
            "best_instruction": getattr(self, '_best_instruction', None),
            "best_score": getattr(self, '_best_score', None),
        }
        final_path = os.path.join(self.save_path, "final_results.json")
        with open(final_path, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=4, ensure_ascii=False)
        logger.info(f"Saved final results to {final_path}")
        
        return final_results

    def _select_graph_with_highest_score(self, return_metrics: bool = False):
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
        
        if return_metrics:
            return best_graph, best_metrics
        return best_graph

    def restore_best_graph(self) -> None:
        if len(self._snapshot) == 0:
            logger.info("No snapshot found. No graph to restore.")
            return

        best_graph, best_metrics = self._select_graph_with_highest_score(return_metrics=True)
        self.graph = best_graph
        logger.info(f"Restored the best graph from snapshot with metrics {best_metrics}")