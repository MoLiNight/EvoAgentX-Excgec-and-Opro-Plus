class MyScoreCalculator:
    """对评估结果（json 数据）进行加权计算，将其转换为单一分数"""
    def __init__(self, positive_metrics: dict = None, negative_metrics: dict = None, mode: str = "all"):
        self.mode = mode
        
        # 默认正向指标（越大越好）
        self.default_positive_metrics = {
            # EXP 指标
            'exp.hit_ratio': 0.15,                      # 编辑匹配率
            'exp.error_type.accuracy': 0.10,            # 错误类型分类准确率
            'exp.error_type.precision_micro': 0.10,     # 错误类型微平均精确率
            'exp.error_type.recall_micro': 0.10,        # 错误类型微平均召回率
            'exp.error_type.f1_micro': 0.15,            # 错误类型微平均F1
            'exp.error_description.bleu': 0.05,         # 错误描述BLEU分数
            'exp.error_description.meteor': 0.05,       # 错误描述METEOR分数
            'exp.error_description.rouge-1': 0.05,      # ROUGE-1
            'exp.error_description.rouge-2': 0.05,      # ROUGE-2
            'exp.error_description.rouge-L': 0.05,      # ROUGE-L
            
            # GEC 指标
            'gec.prf_corpus_unweighted.p': 0.10,        # 精确率
            'gec.prf_corpus_unweighted.r': 0.10,        # 召回率
            'gec.prf_corpus_unweighted.f': 0.15,        # F1分数
            'gec.prf_corpus_unweighted.acc': 0.10,      # 准确率
            'gec.prf_corpus_weighted.f': 0.10,          # 加权F1
            'gec.prf_sentence_unweighted.f': 0.10,      # 句子级F1
            'gec.prf_sentence_weighted.f': 0.05,        # 句子级加权F1
        }

        # 负向指标（越小越好，用负权重）
        self.default_negative_metrics = {
            'exp.error_severity.mae': -0.10,            # 严重程度平均绝对误差
            'gec.prf_corpus_unweighted.fp': -0.05,      # 假正例（误报）
            'gec.prf_corpus_unweighted.fn': -0.05,      # 假负例（漏报）
        }

        self.positive_metrics = positive_metrics or self.default_positive_metrics
        self.negative_metrics = negative_metrics or self.default_negative_metrics

        # 根据模式过滤指标
        self.metric_weights = self._filter_metrics_by_mode()

    def _filter_metrics_by_mode(self) -> dict:
        """根据模式过滤指标"""
        all_weights = {**self.positive_metrics, **self.negative_metrics}
        
        if self.mode == "gec":
            # 只保留 gec 相关的指标
            filtered = {k: v for k, v in all_weights.items() if k.startswith('gec.')}
        elif self.mode == "exp":
            # 只保留 exp 相关的指标
            filtered = {k: v for k, v in all_weights.items() if k.startswith('exp.')}
        elif self.mode == "all":
            # 保留所有指标
            filtered = all_weights
        else:
            raise ValueError(f"Invalid mode: {self.mode}. Choose from 'gec', 'exp', or 'all'.")
        
        # 如果没有匹配的指标，返回空字典
        if not filtered:
            print(f"Warning: No metrics found for mode '{self.mode}'. Using all metrics.")
            filtered = all_weights
        
        return filtered

    def _get_metric_value(self, metrics: dict, path: str) -> float:
        """通过点号分隔的路径（'exp.error_type.f1_micro'）从 json 数据中提取 value"""
        parts = path.split('.')
        value = metrics
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return 0.0
        return value if isinstance(value, (int, float)) else 0.0

    def calculate(self, metrics: dict) -> float:
        """计算加权总分"""
        total_score = 0.0
        total_weight = 0.0

        # 计算加权总分
        for path, weight in self.metric_weights.items():
            value = self._get_metric_value(metrics, path)
            total_score += value * weight
            total_weight += abs(weight)
        
        # 无有效权重时返回 0
        if total_weight == 0:
            return 0.0
        
        # 计算最终分数
        final_score = total_score / total_weight
        
        return final_score