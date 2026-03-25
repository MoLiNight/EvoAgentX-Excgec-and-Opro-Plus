import evoagentx.workflow.operators as operator
import aflow.optimized.round_10.prompt as prompt_custom
from evoagentx.models.model_configs import LLMConfig
from evoagentx.benchmark.benchmark import Benchmark
from evoagentx.models.model_utils import create_llm_instance

class Workflow:
    
    def __init__(
        self,
        name: str,
        llm_config: LLMConfig,
        benchmark: Benchmark
    ):
        self.name = name
        self.llm = create_llm_instance(llm_config)
        self.custom = operator.Custom(self.llm)
        self.sc_ensemble = operator.ScEnsemble(self.llm)
    
    async def __call__(self, problem: str):
        """
        Implementation of the workflow with self-consistency and review
        """
        # Generate multiple solutions using different prompts
        solution1 = await self.custom(input=problem, instruction=prompt_custom.SOLVE_MATH_PROBLEM_PROMPT)
        solution2 = await self.custom(input=problem, instruction=prompt_custom.SOLVE_MATH_PROBLEM_PROMPT_V2)
        solution3 = await self.custom(input=problem, instruction=prompt_custom.SOLVE_MATH_PROBLEM_PROMPT_V3)
        
        # Review each solution to improve quality
        reviewed_solution1 = await self.custom(
            input=f"Problem: {problem}\nSolution: {solution1['response']}",
            instruction=prompt_custom.REVIEW_SOLUTION_PROMPT
        )
        reviewed_solution2 = await self.custom(
            input=f"Problem: {problem}\nSolution: {solution2['response']}",
            instruction=prompt_custom.REVIEW_SOLUTION_PROMPT
        )
        reviewed_solution3 = await self.custom(
            input=f"Problem: {problem}\nSolution: {solution3['response']}",
            instruction=prompt_custom.REVIEW_SOLUTION_PROMPT
        )
        
        # Use ScEnsemble to select the most consistent solution
        solutions = [
            reviewed_solution1['response'],
            reviewed_solution2['response'],
            reviewed_solution3['response']
        ]
        final_solution = await self.sc_ensemble(solutions=solutions, problem=problem)
        
        return final_solution['response']
