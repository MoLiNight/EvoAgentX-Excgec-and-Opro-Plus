class MyScoreCalculator:
    """对评估结果（json 数据）进行加权计算，将其转换为单一分数"""

    def __init__(self, positive_metrics: dict = None, negative_metrics: dict = None):
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

        # 合并所有权重
        self.metric_weights = {**self.positive_metrics, **self.negative_metrics}

        # 必要指标列表（缺失时会扣分）
        self.required_metrics = ['exp.hit_ratio', 'exp.error_type.f1_micro']

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

    def _calculate_penalty(self, metrics: dict) -> float:
        """计算缺失必要指标的惩罚分数"""
        penalty = 0.0
        for req in self.required_metrics:
            if self._get_metric_value(metrics, req) == 0.0:
                penalty += 0.05
        return min(penalty, 0.5)  # 最多扣 50%

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

        # 计算基础分数
        base_score = total_score / total_weight

        # 应用惩罚
        penalty = self._calculate_penalty(metrics)
        final_score = base_score * (1 - penalty)
        
        return final_score