SOLVE_MATH_PROBLEM_PROMPT = r"""
Answer the math question. The answer should be in box format, e.g., \boxed{123}

Problem: """

SOLVE_MATH_PROBLEM_PROMPT_V2 = r"""
Solve the math problem step by step. Provide a detailed explanation and then put the final answer in box format, e.g., \boxed{123}

Problem: """

SOLVE_MATH_PROBLEM_PROMPT_V3 = r"""
Solve the following math problem and provide the final answer in box format. Think carefully and show your reasoning.

Problem: """

REVIEW_SOLUTION_PROMPT = r"""
Review the following solution to the math problem and improve it if necessary. Ensure that the final answer is in box format, e.g., \boxed{123}.

Problem: {problem}
Solution: {solution}
Improvement: """