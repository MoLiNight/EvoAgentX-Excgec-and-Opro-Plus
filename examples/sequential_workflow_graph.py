from evoagentx.workflow.workflow_graph import SequentialWorkFlowGraph

# 定义任务及其输入、输出和提示
tasks = [
    {
        "name": "DataExtraction",
        "description": "Extract data from the specified source",
        "inputs": [
            {"name": "data_source", "type": "string", "required": True, "description": "Source data location"}
        ],
        "outputs": [
            {"name": "extracted_data", "type": "string", "required": True, "description": "Extracted data"}
        ],
        "prompt": "Extract data from the following source: {data_source}", 
        "parse_mode": "str"
    },
    {
        "name": "DataTransformation",
        "description": "Transform the extracted data",
        "inputs": [
            {"name": "extracted_data", "type": "string", "required": True, "description": "Data to transform"}
        ],
        "outputs": [
            {"name": "transformed_data", "type": "string", "required": True, "description": "Transformed data"}
        ],
        "prompt": "Transform the following data: {extracted_data}", 
        "parse_mode": "str"
    },
    {
        "name": "DataAnalysis",
        "description": "Analyze data and generate insights",
        "inputs": [
            {"name": "transformed_data", "type": "string", "required": True, "description": "Data to analyze"}
        ],
        "outputs": [
            {"name": "insights", "type": "string", "required": True, "description": "Generated insights"}
        ],
        "prompt": "Analyze the following data and generate insights: {transformed_data}", 
        "parse_mode": "str"
    }
]

# 创建顺序工作流图
sequential_workflow_graph = SequentialWorkFlowGraph(
    goal="Extract, transform, and analyze data to generate insights",
    tasks=tasks
)

# 保存顺序工作流图
sequential_workflow_graph.save_module("output/my_sequential_workflow.json")