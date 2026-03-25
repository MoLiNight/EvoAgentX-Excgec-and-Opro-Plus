import os
from typing import Any
import numpy as np
from tabulate import tabulate

from EvoAgentX.evoagentx.benchmark.benchmark import Benchmark
from EvoAgentX.evoagentx.core.logging import logger
from EvoAgentX.evoagentx.core.module_utils import load_json

import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "EXCGEC"))
from data import Dataset, Sample
from utils import get_logger, remove_space
LOGGER = get_logger(__name__)
from benchmarks.xcgec.objects import XDataset, XEdit, XSample

from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_chinese import Rouge
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    precision_recall_fscore_support,
)

from benchmarks.xcgec.evaluate import check_dataset
from evaluation import DependentCLEME, ScorerType, WeigherType
from benchmarks.xcgec.objects_eval import BaseExplanationMetricResult, SampleExplanationMetricResult

EXCGEC_FILES_MAP = {"train": "train.json", "dev": "valid.json", "test": "test.json"}
VALIDE_RAW_EXCGEC_FILES = [file for file in list(EXCGEC_FILES_MAP.values()) if file is not None]

class EXCGEC(Benchmark):

    # Override
    def __init__(self, path: str = None, mode: str = "all", **kwargs):
        path = os.path.expanduser(path or "EXCGEC/benchmarks/xcgec/data/splits")
        super().__init__(name=type(self).__name__, path=path, mode=mode, **kwargs)

    def _load_data_from_file(self, file_name: str):
        file_path = os.path.join(self.path, file_name)
        return load_json(path = file_path, type = "json")

    def _load_data(self):
        if self.mode == "train" or self.mode == "all":
            self._train_data = self._load_data_from_file(file_name=EXCGEC_FILES_MAP["train"])["samples"]
        if self.mode == "dev" or self.mode == "all":
            self._dev_data = self._load_data_from_file(file_name=EXCGEC_FILES_MAP["dev"])["samples"]
        if self.mode == "test" or self.mode == "all":
            self._test_data = self._load_data_from_file(file_name=EXCGEC_FILES_MAP["test"])["samples"]
    
    def _get_label(self, example: Any) -> Any:
        return example
    
    def _get_id(self, example: Any) -> Any:
        return example['index']
    
    # EXCGEC
    def convert_json_to_xsample(self, data: Any) -> Any:
        """将模型返回的 json 数据封装为 XSample 对象，方便后续评估"""
        xedits = []
        for edit in data['edits']:
            xedit = XEdit(
                src_interval=edit['src_interval'],
                tgt_interval=edit['tgt_interval'],
                src_content=edit['src_content'],
                tgt_content=edit['tgt_content'],
                error_type=edit['error_type'],
                error_severity=edit['error_severity'],
                error_description=edit['error_description']
            )
            xedits.append(xedit)
        
        xsample = XSample(
            index=data['index'],
            domain=data['domain'],
            source=data['source'],
            target=data['target'],
            edits=xedits
        )
        
        return xsample

    # GEC
    def convert_dataset(self, dataset):
        """将 EXCGEC 的 XDataset 对象转换为 GEC 评估所需的 Dataset 对象"""
        gec_dataset = Dataset()
        for exp_sample in dataset:
            gec_sample = Sample(
                index=exp_sample.index,
                source=[exp_sample.source],
                target=[exp_sample.target],
            )
            gec_dataset.append(gec_sample)
        return gec_dataset

    def get_chunked_dataset(self, dataset, merge_distance: int = 1, output_visualize: str = None):
        """构建含 chunk 的数据集，用于 GEC 评估。"""
        # Convert XDataset into conventional Dataset
        gec_dataset = self.convert_dataset(dataset=dataset)

        metric = DependentCLEME(
            lang="zho",
            scorer_type=ScorerType.PRF,
            weigher_type=WeigherType.LENGTH,
            output_visualize=output_visualize,
            merge_distance=merge_distance,
        )
        metric.prepare_dataset(gec_dataset)

        # Chunk partition
        chunk_dataset = metric.chunk_partition(
            dataset=gec_dataset, merge_distance=merge_distance
        )
        for sample_chunk, gec_sample in zip(chunk_dataset, gec_dataset):
            gec_sample.chunks = [sample_chunk]

        return gec_dataset

    def evaluate_gec(self, dataset_hyp, dataset_ref,
        lang: str = "zho", merge_distance: int = 1, output_visualize: str = None, output_evaluation: str = None):
        """GEC 评估函数"""
        metric = DependentCLEME(
            lang=lang,
            scorer_type=ScorerType.PRF,
            weigher_type=WeigherType.LENGTH,
            output_visualize=output_visualize,
            merge_distance=merge_distance,
        )
        
        scorer_results, metric_results = metric.evaluate(
            dataset_hyp=dataset_hyp,
            dataset_ref=dataset_ref,
            persist_path=output_evaluation,
        )
        return scorer_results
    
    # EXP
    def match_edits(self, sample_hyp, sample_ref):
        """计算参考编辑与模型返回的编辑的 src_interval:[begin, end] 的位置重叠度，为每个返回的编辑找到重叠最多的参考编辑，用于后续 EXP 评估"""
        results = []
        for edit_hyp in sample_hyp.edits:
            src_pos_hyp = set(range(edit_hyp.src_interval[0], edit_hyp.src_interval[1]+1))
            best_edit_ref, max_overlap = None, 0
            for edit_ref in sample_ref.edits:
                src_pos_ref = set(range(edit_ref.src_interval[0], edit_ref.src_interval[1]+1))
                curr_overlap = len(src_pos_hyp & src_pos_ref)
                if curr_overlap > max_overlap:
                    best_edit_ref = edit_ref
                    max_overlap = curr_overlap

            result = BaseExplanationMetricResult(edit_hyp=edit_hyp, edit_ref=best_edit_ref)
            results.append(result)
            LOGGER.debug(f"Match Edit: {result}")

        return results
    
    def evaluate_exp(self, dataset_hyp, dataset_ref, verbose: bool = False):
        """EXP 评估函数"""
        num_pred, num_true, hit = 0, 0, 0
        # Match edits
        sample_results = []
        for sample_hyp, sample_ref in zip(dataset_hyp, dataset_ref):
            edit_results = self.match_edits(sample_hyp=sample_hyp, sample_ref=sample_ref)
            sample_result = SampleExplanationMetricResult(bases=edit_results)
            sample_results.append(sample_result)

            num_pred += len(sample_hyp.edits)
            num_true += len(sample_ref.edits)
            hit += len(list(filter(lambda x: x.edit_ref, edit_results)))

        # 检查是否有匹配的编辑
        has_matched = hit > 0
        
        # Evaluate different parts of explanations
        if has_matched:
            eval_error_type = self.evaluate_exp_error_type_simple(sample_results, verbose=verbose)
            eval_error_severity = self.evaluate_exp_error_severity_simple(sample_results)
            eval_error_description = self.evaluate_exp_error_description_simple(sample_results)
        else:
            # 没有匹配时返回默认值
            eval_error_type = {
                "accuracy": 0.0,
                "precision_micro": 0.0,
                "recall_micro": 0.0,
                "f1_micro": 0.0,
            }
            eval_error_severity = {"mae": 0.0}
            eval_error_description = {
                "bleu": 0.0,
                "meteor": 0.0,
                "rouge-1": 0.0,
                "rouge-2": 0.0,
                "rouge-L": 0.0,
            }
        
        """Visualize results as a table."""
        print("Error Type:")
        print(tabulate(eval_error_type.items(), headers=["Metric", "Value"], tablefmt="grid"))
        print()

        print("Error Severity:")
        print(tabulate(eval_error_severity.items(), headers=["Metric", "Value"], tablefmt="grid"))
        print()

        print("Error Description:")
        print(tabulate(eval_error_description.items(), headers=["Metric", "Value"], tablefmt="grid"))
        
        return {
            "num_pred": num_pred,
            "num_true": num_true,
            "hit": hit,
            "hit_ratio": round(hit / num_pred, 4) if num_pred > 0 else 0.0,
            "error_type": eval_error_type,
            "error_severity": eval_error_severity,
            "error_description": eval_error_description,
        }

    def evaluate_exp_error_type_simple(self, sample_results, verbose: bool = True):
        """简化版错误类型评估 - 仅计算准确率和微平均指标"""
        y_true, y_pred = [], []
        for sample_result in sample_results:
            for edit_result in sample_result.bases:
                if edit_result.edit_ref is not None:
                    y_true.append(edit_result.edit_ref.error_type)
                    y_pred.append(edit_result.edit_hyp.error_type)
        
        if not y_true or not y_pred:
            return {
                "accuracy": 0.0,
                "precision_micro": 0.0,
                "recall_micro": 0.0,
                "f1_micro": 0.0,
            }

        # 计算准确率和微平均指标
        acc = accuracy_score(y_true=y_true, y_pred=y_pred)
        
        precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
            y_true=y_true, y_pred=y_pred, average="micro"
        )
        
        if verbose:
            print(f"Accuracy: {acc:.4f}")
            print(f"Micro Precision: {precision_micro:.4f}")
            print(f"Micro Recall: {recall_micro:.4f}")
            print(f"Micro F1: {f1_micro:.4f}")

        return {
            "accuracy": round(acc, 4),
            "precision_micro": round(precision_micro, 4),
            "recall_micro": round(recall_micro, 4),
            "f1_micro": round(f1_micro, 4),
        }

    def evaluate_exp_error_severity_simple(self, sample_results):
        """简化版错误严重程度评估 - 计算平均绝对误差"""
        y_true, y_pred = [], []
        for sample_result in sample_results:
            for edit_result in sample_result.bases:
                if edit_result.edit_ref is not None:
                    y_true.append(edit_result.edit_ref.error_severity)
                    y_pred.append(edit_result.edit_hyp.error_severity)
        
        if not y_true or not y_pred:
            return {"mae": 0.0}

        mae = mean_absolute_error(y_true=y_true, y_pred=y_pred)
        return {"mae": round(mae, 4)}

    def evaluate_exp_error_description_simple(self, sample_results):
        """简化版错误描述评估"""
        y_true, y_pred = [], []
        for sample_result in sample_results:
            for edit_result in sample_result.bases:
                if edit_result.edit_ref is not None:
                    y_true.append(edit_result.edit_ref.error_description)
                    y_pred.append(edit_result.edit_hyp.error_description)
        
        if not y_true or not y_pred:
            return {
                "bleu": 0.0,
                "meteor": 0.0,
                "rouge-1": 0.0,
                "rouge-2": 0.0,
                "rouge-L": 0.0
            }

        rouge = Rouge()
        bleu_reults, meteor_results = [], []
        rouge1_results, rouge2_results, rouge_long_results = [], [], []
        
        for hyp, ref in zip(y_pred, y_true):
            hyp_tokens = self.tokenize(hyp)
            ref_tokens = self.tokenize(ref)
            
            # BLEU
            bleu = sentence_bleu(references=[ref_tokens], hypothesis=hyp_tokens)
            bleu_reults.append(bleu)

            # METEOR
            meteor = meteor_score(references=[ref_tokens], hypothesis=hyp_tokens)
            meteor_results.append(meteor)
            
            # ROUGE
            rouge_tmp = rouge.get_scores(
                hyps=[" ".join(hyp_tokens)], refs=[" ".join(ref_tokens)]
            )
            rouge1_results.append(rouge_tmp[0]["rouge-1"]["f"])
            rouge2_results.append(rouge_tmp[0]["rouge-2"]["f"])
            rouge_long_results.append(rouge_tmp[0]["rouge-l"]["f"])
        
        return {
            "bleu": round(np.average(bleu_reults), 4),
            "meteor": round(np.average(meteor_results), 4),
            "rouge-1": round(np.average(rouge1_results), 4),
            "rouge-2": round(np.average(rouge2_results), 4),
            "rouge-L": round(np.average(rouge_long_results), 4),
        }

    def tokenize(self, content: str):
        return [x for x in remove_space(content.strip())]

    def evaluate(self, prediction: Any, label: Any) -> dict:
        """最终评估函数，供其他文件调用"""
        dataset_ref = XDataset()
        dataset_ref.append(self.convert_json_to_xsample(label))

        dataset_hyp = XDataset()
        # 难以保证模型返回的数据格式正确，故对模型返回的数据的格式要求降低，改为人为处理
        prediction['index'] = label['index']
        prediction['domain'] = label['domain']
        prediction['source'] = label['source']
        dataset_hyp.append(self.convert_json_to_xsample(prediction))

        scores = {}

        print("--------------------Begin Check--------------------")
        try:
            check_dataset(dataset_hyp)
            check_dataset(dataset_ref)
        except Exception as e:
            print(f"Check dataset failed: {e}")
            return {
                'exp': {
                    'num_pred': 0,
                    'num_true': 0,
                    'hit': 0,
                    'hit_ratio': 0.0,
                    'error_type': {
                        'accuracy': 0.0,
                        'precision_micro': 0.0,
                        'recall_micro': 0.0,
                        'f1_micro': 0.0
                    },
                    'error_severity': {'mae': 0.0},
                    'error_description': {
                        'bleu': 0.0,
                        'meteor': 0.0,
                        'rouge-1': 0.0,
                        'rouge-2': 0.0,
                        'rouge-L': 0.0
                    }
                },
                'gec': {
                    'prf_corpus_unweighted': {
                        'tp': 0.0, 'fp': 0.0, 'fn': 0.0, 'tn': 0.0,
                        'p': 0.0, 'r': 0.0, 'f': 0.0, 'acc': 0.0
                    },
                    'prf_corpus_weighted': {
                        'tp': 0.0, 'fp': 0.0, 'fn': 0.0, 'tn': 0.0,
                        'p': 0.0, 'r': 0.0, 'f': 0.0, 'acc': 0.0
                    },
                    'prf_sentence_unweighted': {
                        'tp': 0.0, 'fp': 0.0, 'fn': 0.0, 'tn': 0.0,
                        'p': 0.0, 'r': 0.0, 'f': 0.0, 'acc': 0.0
                    },
                    'prf_sentence_weighted': {
                        'tp': 0.0, 'fp': 0.0, 'fn': 0.0, 'tn': 0.0,
                        'p': 0.0, 'r': 0.0, 'f': 0.0, 'acc': 0.0
                    },
                    'num_sample': 1
                }
            }
        
        print("--------------------Begin EXP--------------------")
        try:
            scores["exp"] = self.evaluate_exp(dataset_ref=dataset_ref, dataset_hyp=dataset_hyp)
        except Exception as e:
            print(f"EXP evaluation failed: {e}")
            scores["exp"] = {"error": str(e)}
        
        print("--------------------Begin Chunked--------------------")
        try:
            dataset_ref = self.get_chunked_dataset(dataset_ref)
            dataset_hyp = self.get_chunked_dataset(dataset_hyp)
        except Exception as e:
            print(f"Chunked dataset creation failed: {e}")
            return {"error": f"Chunked dataset creation failed: {str(e)}"}

        print("--------------------Begin GEC--------------------")
        try:
            scores["gec"] = self.evaluate_gec(dataset_ref=dataset_ref, dataset_hyp=dataset_hyp)
        except Exception as e:
            print(f"GEC evaluation failed: {e}")
            scores["gec"] = {"error": str(e)}

        print(scores)
        return scores