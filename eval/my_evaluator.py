from evoagentx.evaluators import Evaluator
from evoagentx.core.logging import logger

class MyEvaluator(Evaluator):
    """
        原评估器的平均分数计算代码无法处理模型返回的 json 数据，故进行函数的代码重写
        原代码：return {k: sum(d[k] for d in scores if d is not None) / num_total_items for k in first_valid_score}
    """
    def _calculate_average_score(self, scores):
        if not scores:
            logger.warning("No scores found. Return an empty dictionary.")
            return {}
        
        valid_scores = [s for s in scores if s is not None]
        if not valid_scores:
            logger.warning("No valid scores found. Return an empty dictionary.")
            return {}
        
        num_total_items = len(scores)
        result = {}
        
        all_keys = set()
        for score in valid_scores:
            all_keys.update(score.keys())
        
        for key in all_keys:
            values = [score.get(key) for score in valid_scores if key in score]

            if not values: continue
            
            first_val = values[0]
            
            # 情况1：数值类型
            if isinstance(first_val, (int, float)):
                numeric_values = [v for v in values if isinstance(v, (int, float))]
                if numeric_values:
                    result[key] = sum(numeric_values) / num_total_items
            
            # 情况2：OverallScorerResult 对象
            elif hasattr(first_val, 'scores') and hasattr(first_val, 'num_sample'):
                result[key] = {}
                
                all_scores = {}
                for obj in values:
                    if hasattr(obj, 'scores'):
                        for score_key, scorer in obj.scores.items():
                            if score_key not in all_scores:
                                all_scores[score_key] = []
                            all_scores[score_key].append(scorer)
                
                for score_key, scorers in all_scores.items():
                    result[key][score_key] = {}
                    
                    metrics = {}
                    for scorer in scorers:
                        if hasattr(scorer, '__dict__'):
                            for attr, val in scorer.__dict__.items():
                                if isinstance(val, (int, float)):
                                    if attr not in metrics:
                                        metrics[attr] = []
                                    metrics[attr].append(val)
                    
                    for attr, vals in metrics.items():
                        if vals:
                            result[key][score_key][attr] = sum(vals) / len(vals)
                
                total_samples = sum(obj.num_sample for obj in values if hasattr(obj, 'num_sample'))
                result[key]['num_sample'] = total_samples / len(values)
            
            # 情况3：json 数据
            elif isinstance(first_val, dict):
                result[key] = self._average_dict_values(values, num_total_items)
            
            # 其他类型
            else:
                result[key] = first_val
        
        return result

    def _average_dict_values(self, dict_list, total_items):
        """递归计算 json 数据的平均值"""
        if not dict_list:
            return {}
        
        result = {}
        
        # 获取所有键
        all_keys = set()
        for d in dict_list:
            if isinstance(d, dict):
                all_keys.update(d.keys())
        
        for key in all_keys:
            values = [d.get(key) for d in dict_list if isinstance(d, dict) and key in d]
            
            if not values:
                continue
            
            first_val = values[0]
            
            # 数值类型
            if isinstance(first_val, (int, float)):
                numeric_values = [v for v in values if isinstance(v, (int, float))]
                if numeric_values:
                    result[key] = sum(numeric_values) / total_items
            
            # 嵌套字典
            elif isinstance(first_val, dict):
                result[key] = self._average_dict_values(values, total_items)
            
            # 列表类型
            elif isinstance(first_val, list):
                # 处理列表，假设是数值列表
                if all(isinstance(v, list) for v in values):
                    # 找到最大长度
                    max_len = max(len(lst) for lst in values)
                    avg_list = []
                    for i in range(max_len):
                        pos_values = [lst[i] for lst in values if i < len(lst) and isinstance(lst[i], (int, float))]
                        if pos_values:
                            avg_list.append(sum(pos_values) / len(pos_values))
                        else:
                            avg_list.append(None)
                    result[key] = avg_list
                else:
                    result[key] = first_val
            
            # 其他类型
            else:
                result[key] = first_val
        
        return result