import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from dotenv import load_dotenv
from copy import deepcopy

from evoagentx.agents import CustomizeAgent
from evoagentx.models import AliyunLLM, AliyunLLMConfig
from evoagentx.core.callbacks import suppress_logger_info
from evoagentx.core.logging import logger

from eval.excgec import EXCGEC
from eval.my_evaluator import MyEvaluator
from eval.my_score_calculator import MyScoreCalculator

load_dotenv()
my_api_key = os.getenv("DASHSCOPE_API_KEY")

class ExcgecPlus(EXCGEC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _load_data(self):
        super()._load_data()
        import numpy as np
        np.random.seed(42)
        permutation = np.random.permutation(len(self._test_data))
        full_test_data = self._test_data

        self._train_data = [full_test_data[idx] for idx in permutation[:100]]
        self._dev_data = [full_test_data[idx] for idx in permutation[100:600]]
        self._test_data = [full_test_data[idx] for idx in permutation[600:1100]]

def process_single_sample(grammar_corrector, sample, benchmark, eval_rounds=3):
    """处理单个样本，支持多轮评估取平均"""
    source = sample["source"]
    label = sample
    
    round_results = []
    
    for round_idx in range(eval_rounds):
        with suppress_logger_info():
            response = grammar_corrector(inputs={"source": source})
            
            prediction = {
                "index": label["index"],
                "domain": label["domain"],
                "source": label["source"],
                "target": response.content.target if hasattr(response.content, 'target') else "",
                "edits": response.content.edits if hasattr(response.content, 'edits') else []
            }
            
            eval_result = benchmark.evaluate(prediction=prediction, label=label)
            round_results.append(eval_result)
    
    # 计算多轮评估的平均值
    avg_result = _average_eval_results(round_results)
    return avg_result

def _average_eval_results(results):
    """计算多轮评估结果的平均值"""
    # 初始化聚合字典（使用第一个结果的键结构）
    avg_result = deepcopy(results[0])
    
    # 递归计算平均值
    def _average_dict(dict_list, target_dict):
        for key in target_dict:
            if isinstance(target_dict[key], dict):
                _average_dict([d[key] for d in dict_list], target_dict[key])
            elif isinstance(target_dict[key], (int, float)):
                values = [d[key] for d in dict_list if key in d]
                if values:
                    target_dict[key] = sum(values) / len(values)
    
    _average_dict(results, avg_result)
    
    return avg_result

def process_batch(grammar_corrector, samples, benchmark, batch_size=10, max_workers=10, eval_rounds=3):
    """批量处理样本"""
    results = []
    total_samples = len(samples)
    
    # 使用 tqdm 显示进度
    with tqdm(total=total_samples, desc=f"Processing samples (eval_rounds={eval_rounds})") as pbar:
        # 分批处理
        for i in range(0, total_samples, batch_size):
            batch = samples[i:i + batch_size]
            
            # 并行处理当前批次
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 提交所有任务
                futures = {
                    executor.submit(process_single_sample, grammar_corrector, sample, benchmark, eval_rounds): sample
                    for sample in batch
                }
                
                # 收集结果
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        sample = futures[future]
                        logger.error(f"Error processing sample {sample.get('index', 'unknown')}: {e}")
                        # 添加空结果
                        results.append({
                            'exp': {'hit_ratio': 0.0, 'num_pred': 0, 'num_true': 0, 'hit': 0,
                                    'error_type': {'accuracy': 0.0, 'precision_micro': 0.0, 
                                                'recall_micro': 0.0, 'f1_micro': 0.0},
                                    'error_severity': {'mae': 0.0},
                                    'error_description': {'bleu': 0.0, 'meteor': 0.0, 
                                                        'rouge-1': 0.0, 'rouge-2': 0.0, 'rouge-L': 0.0}},
                            'gec': {'prf_corpus_unweighted': {'tp': 0.0, 'fp': 0.0, 'fn': 0.0, 'tn': 0.0,
                                                            'p': 0.0, 'r': 0.0, 'f': 0.0, 'acc': 0.0},
                                    'prf_corpus_weighted': {'tp': 0.0, 'fp': 0.0, 'fn': 0.0, 'tn': 0.0,
                                                            'p': 0.0, 'r': 0.0, 'f': 0.0, 'acc': 0.0},
                                    'prf_sentence_unweighted': {'tp': 0.0, 'fp': 0.0, 'fn': 0.0, 'tn': 0.0,
                                                                'p': 0.0, 'r': 0.0, 'f': 0.0, 'acc': 0.0},
                                    'prf_sentence_weighted': {'tp': 0.0, 'fp': 0.0, 'fn': 0.0, 'tn': 0.0,
                                                            'p': 0.0, 'r': 0.0, 'f': 0.0, 'acc': 0.0},
                                    'num_sample': 1}
                        })
                    pbar.update(1)
    
    return results

def main():
    EVAL_ROUNDS = 2          # 评估轮数，对每个样本进行多次评估取平均
    BATCH_SIZE = 10          # 批次大小
    MAX_WORKERS = 10         # 并行线程数
    
    llm_config = AliyunLLMConfig(
        model="qwen-turbo",
        aliyun_api_key=my_api_key,
        temperature=0.1,
        max_tokens=16000,
        stream=False,
        output_response=True
    )
    llm = AliyunLLM(llm_config)

    grammar_corrector = CustomizeAgent(
        name="GrammarCorrector",
        description="检测并纠正中文句子中的语法错误，并给出错误解释",
        prompt="""
        请仔细分析输入的中文句子，检测其中的语法错误，并按照要求的 JSON 格式输出纠正结果和错误解释。

        输入句子：{source}

        输出格式（必须严格遵循，不得添加任何 Markdown 格式）：
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
                    "error_type": str,           # Error type
                    "error_severity": int,       # Error severity (1-5, higher number indicates more severe)
                    "error_description": str     # Detailed error description and correction explanation
                }},
                ...  # Multiple edit operations can exist
            ]
        }}

        错误类型范围：(必须从以下的错误中选择进行回答，不得进行任何修改)
        标点冗余，标点丢失，标点误用；字音混淆错误，字形混淆错误，词内部字符异位错误，命名实体拼写错误；
        词语冗余，词语丢失，词语误用；词序不当，逻辑不通，句式杂糅；照应错误，歧义错误，语气不协调；其他错误。
        """,
        llm_config=llm_config,
        inputs=[
            {"name": "source", "type": "string", "description": "待纠错的中文句子"}
        ],
        outputs=[
            {"name": "target", "type": "string", "description": "纠正后的句子"},
            {"name": "edits", "type": "list", "description": "错误编辑列表"}
        ],
        parse_mode="json"
    )

    benchmark = ExcgecPlus(eval_mode="all", mode="test")
    test_data = benchmark._test_data

    # 批量处理
    results = process_batch(
        grammar_corrector=grammar_corrector,
        samples=test_data,
        benchmark=benchmark,
        batch_size=BATCH_SIZE,
        max_workers=MAX_WORKERS,
        eval_rounds=EVAL_ROUNDS
    )

    # 聚合结果
    evaluator = MyEvaluator(llm=llm)
    avg_result = evaluator._calculate_average_score(results)

    score_calculator = MyScoreCalculator()
    final_score = score_calculator.calculate(avg_result)
    
    logger.info(f"Final score: {final_score}")
    logger.info(f"Detailed results:\n{avg_result}")
    
if __name__ == "__main__":
    main()