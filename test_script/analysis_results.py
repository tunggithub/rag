import json
from typing import Dict

def conversation_results_analysis(results: Dict):
    similarity_scores = [float(sample['similarity']) for sample in results]
    process_time = [float(sample['process_time']) for sample in results]
    similarity_score_avg = sum(similarity_scores) / len(similarity_scores)
    process_time_avg = sum(process_time) / len(process_time)
    print("---------------CONVERSATION---------------")
    print(f"Average similarity score: {similarity_score_avg}")
    print(f"Average process time: {process_time_avg} s")

def qa_results_analysis(results: Dict):
    process_time = [sample['process_time'] for sample in results]
    citation_rate = [sample['citation_rate'] for sample in results]
    correct_response = [1 if sample['correct_response'] == True else 0 for sample in results]
    relevent_content = [1 if sample['relevent_content'] == True else 0 for sample in results]
    
    process_time_avg = sum(process_time) / len(process_time)
    citation_rate_avg = sum(citation_rate) / len(citation_rate)
    correct_response_avg = sum(correct_response) / len(correct_response)
    relevent_content_avg = sum(relevent_content) / len(relevent_content)
   
    print("---------------QnA---------------")
    print(f"Average process time: {process_time_avg} s")
    print(f"Average citation rate: {citation_rate_avg}")
    print(f"Average correct response: {correct_response_avg}")
    print(f"Average relevent content: {relevent_content_avg}")

def tool_call_results_analysis(results: Dict):
    process_time = [sample['process_time'] for sample in results]
    correct_tool_percent = [sample['correct_tool_percentage'] for sample in results]
    correct_assistant_percent = [sample['correct_assistant_percentage'] for sample in results]

    process_time_avg = sum(process_time) / len(process_time)
    correct_tool_avg = sum(correct_tool_percent) / len(correct_tool_percent)
    correct_assistant_avg = sum(correct_assistant_percent) / len(correct_assistant_percent)

    print("---------------Tool Call---------------")
    print(f"Average process time: {process_time_avg} s")
    print(f"Average correct tool: {correct_tool_avg}")
    print(f"Average correct assist: {correct_assistant_avg}")

def tool_gen_results_analysis(results: Dict):
    process_time = [sample['process_time'] for sample in results]
    scores = [sample['scores'] for sample in results]
    name_similarity_scores = [score['name_similarity'] for score in scores]
    description_similarity_scores = [score['description_similarity'] for score in scores]
    match_arg_rate = [score['match_arg_rate'] for score in scores]

    process_time_avg = sum(process_time) / len(process_time)
    name_similarity_avg = sum(name_similarity_scores) / len(name_similarity_scores)
    description_similarity_avg = sum(description_similarity_scores) / len(description_similarity_scores)
    match_arg_avg = sum(match_arg_rate) / len(match_arg_rate)

    print("---------------Tool Generation---------------")
    print(f"Average process time: {process_time_avg} s")
    print(f"Average name similarity: {name_similarity_avg}")
    print(f"Average description similarity: {description_similarity_avg}")
    print(f"Average match args rate: {match_arg_avg}")


if __name__ == "__main__":
    conversation_res_path = "../test_results/test_conversation_result.json"
    conversation_result = json.load(open(conversation_res_path))
    conversation_results_analysis(conversation_result)

    qa_res_path = "../test_results/test_qa_result_with_scores.json"
    qa_result = json.load(open(qa_res_path))
    qa_results_analysis(qa_result)

    tool_gen_res_path = "../test_results/test_toolgen_result_with_scores.json"
    tool_gen_result = json.load(open(tool_gen_res_path))
    tool_gen_results_analysis(tool_gen_result)

    tool_call_res_path = "../test_results/test_toolcall_result_with_scores.json"
    tool_call_res = json.load(open(tool_call_res_path))
    tool_call_results_analysis(tool_call_res)