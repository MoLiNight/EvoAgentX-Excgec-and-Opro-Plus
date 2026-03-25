import os
from dotenv import load_dotenv
from evoagentx.models import AliyunLLM, AliyunLLMConfig
from evoagentx.agents import AgentManager
from evoagentx.workflow import WorkFlowGraph, WorkFlow

load_dotenv()  
my_api_key = os.getenv("DASHSCOPE_API_KEY")

def main():
    target_directory = "./output"
    result_path = os.path.join(target_directory, "results.md")
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

    workflow_graph = WorkFlowGraph.from_file("./output/original_workflow.json")
    
    agent_manager = AgentManager()
    agent_manager.add_agents_from_workflow(workflow_graph, llm_config=llm_config)

    workflow = WorkFlow(graph=workflow_graph, agent_manager=agent_manager, llm=llm)
    output = workflow.execute(
        {
            "source": "作为古希腊哲学家，亚里士多德在本体论问题的论述中充满着辩证法，因此被誉为“古代世界的黑格尔”。"
        }
    )

    with open(result_path, "w", encoding="utf-8") as f:
        f.write(output)

    print(f"✅ Your file has been saved to：{result_path}")
    print("📬 You can run this script everyday to obtain daily recommendation")

if __name__ == "__main__":
    main()