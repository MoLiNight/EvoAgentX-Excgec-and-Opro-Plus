import os
from dotenv import load_dotenv
from evoagentx.models import AliyunLLM, AliyunLLMConfig
from evoagentx.agents import AgentManager
from evoagentx.workflow import WorkFlowGenerator, WorkFlowGraph, WorkFlow

from my_graph import GOAL

load_dotenv()  
my_api_key = os.getenv("DASHSCOPE_API_KEY")

def main():
    target_directory = "./output"
    module_save_path = os.path.join(target_directory, "original_workflow.json")
    os.makedirs(target_directory, exist_ok=True)

    llm_config = AliyunLLMConfig(
        model="qwen-turbo",  # qwen-turbo、qwen-plus、qwen-max and so on
        aliyun_api_key=my_api_key,
        temperature=0.1,
        max_tokens=16000,
        stream=False,
        output_response=True
    )
    llm = AliyunLLM(llm_config)
    goal = GOAL

    wf_generator = WorkFlowGenerator(llm=llm)
    workflow_graph: WorkFlowGraph = wf_generator.generate_workflow(goal=goal)

    workflow_graph.save_module(module_save_path)
    workflow_graph.display()

    print(f"✅ The workflow_graph has been saved to：{module_save_path}")

if __name__ == "__main__":
    main()