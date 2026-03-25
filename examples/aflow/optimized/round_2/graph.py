import evoagentx.workflow.operators as operator
import aflow.optimized.round_2.prompt as prompt_custom
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
        self.programmer = operator.Programmer(self.llm)

    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        """
        # Generate multiple solutions using custom method
        solutions = []
        for _ in range(3):
            solution = await self.custom(input=problem, instruction=prompt_custom.SOLVE_MATH_PROBLEM_PROMPT)
            solutions.append(solution['response'])

        # Use ScEnsemble to select the most consistent solution
        final_solution = await self.sc_ensemble(solutions=solutions, problem=problem)

        # Verify the final solution with Programmer
        verification = await self.programmer(problem=problem, analysis=final_solution['response'])
        
        # Return the verified solution
        return verification['output']
