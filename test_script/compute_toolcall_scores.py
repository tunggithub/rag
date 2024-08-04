import json
import ast
from tqdm import tqdm
import difflib 
from sentence_transformers import SentenceTransformer, util
from typing import Dict, List
from pydantic import BaseModel, Field
from test_script.test_toolcall import messages_from_list, find_first_tool_call, validate_tool_call, messages_to_list

sentence_transformer = SentenceTransformer('BAAI/bge-small-en-v1.5')
def measure_relevance_of_texts(text1, text2): 
    if type(text2) == list:
        embeddings = sentence_transformer.encode([text1,*text2], convert_to_tensor=True, show_progress_bar=False)
    else:
        embeddings = sentence_transformer.encode([text1,text2], convert_to_tensor=True, show_progress_bar=False)
    # Compute the cosine similarity between the embeddings
    if type(text2) == list:
        return util.pytorch_cos_sim(embeddings[0], embeddings[1:])[0]
    else:
        return float(util.pytorch_cos_sim(embeddings[0], embeddings[1:])[0][0])


def find_last_assistant(messages):
    for d in reversed(messages_to_list(messages)):
        if d.role == 'assistant':
            return d.content

def find_assistant_after_tool_call(dicts):
    found_tool_call = False
    for d in dicts:
        if not found_tool_call:
            if d.role == 'tool call':
                found_tool_call = True
        elif d.role == 'assistant':
            return d.content
    return None  # If no matching dictionary is found after the 'tool call'

def correct_tool_use_and_response(expected_convo, llm_response, tools):
    expected_convo = messages_from_list(expected_convo)
    miner_convo = messages_from_list(llm_response)
    
    expect_tool_call = True

    if not any([msg.role == 'tool call' for msg in expected_convo]):
        expect_tool_call = False
        if any([msg.role == 'tool call' for msg in miner_convo]):
            return 0, 0
    
    # Check to ensure that it goes `tool call` then `assistant`
    if any([msg.role == 'tool call' for msg in expected_convo]):
        if len(miner_convo) != 2:
            return 0, 0
        if miner_convo[0].role != 'tool call' or miner_convo[1].role != 'assistant':
            return 0, 0
    else:
        if len(miner_convo) != 1:
            return 0, 0
    

    try:
        if isinstance(find_first_tool_call(expected_convo).content, str):
            expected_tool_call = ast.literal_eval(find_first_tool_call(expected_convo).content)
        else:
            expected_tool_call = find_first_tool_call(expected_convo).content
    except Exception as e:
        return 1, 1

    
    try: 
        miner_tool_call = [msg for msg in miner_convo if msg.role == 'tool call'][0].content
        if isinstance(miner_tool_call, str):
            # miner_tool_call = ast.literal_eval(miner_tool_call)
            miner_tool_call = json.loads(miner_tool_call)
    except:
        return 0, 0
    # try:
    #     for tool in tools:
    #         if tool.name == miner_tool_call['name']:
    #             if not validate_tool_call(tool, miner_tool_call):
    #                 raise Exception("Miner failed to return a valid tool call.")    
    # except:
    #     return 0, 0
    
    # Compare arguments
    num_expected_keys = 2 * len(expected_tool_call['arguments'].keys()) + 1 # extra 1 is for the name
    num_gotten_keys = 0
    num_gotten_values = 0
    for miner_key, miner_value in miner_tool_call['arguments'].items():
        if miner_key in expected_tool_call['arguments']:
            num_gotten_keys += 1
            
            if difflib.SequenceMatcher(None, str(miner_value), str(expected_tool_call['arguments'][miner_key])).ratio() > 0.75:
                num_gotten_values += 1
    if difflib.SequenceMatcher(None, str(miner_tool_call['name']), str(expected_tool_call['name'])).ratio() > 0.75:
        num_gotten_values += 1
    correct_tool_percentage = (num_gotten_values+num_gotten_keys)/(num_expected_keys)
    try:
        if expect_tool_call:
            expected_assistant = find_assistant_after_tool_call(expected_convo)
        else:
            expected_assistant = find_last_assistant(expected_convo)
    except Exception as e:
        return 1, 1
    correct_assistant_percentage = 0
    
    try:
        if expect_tool_call:
            miner_assistant = find_assistant_after_tool_call(miner_convo)
        else:
            miner_assistant = find_last_assistant(miner_convo)
        sim = measure_relevance_of_texts(expected_assistant, miner_assistant)
        if sim>0.90:
            correct_assistant_percentage = 1
        elif sim>0.85:
            correct_assistant_percentage = 0.75
        elif sim>0.50:
            correct_assistant_percentage = 0.25
    except Exception as e:
        return 0, 0
            
    return correct_tool_percentage, correct_assistant_percentage


if __name__ == '__main__':
    test_file = 'test_results/test_toolcall_result.json'
    output_file = 'test_results/test_toolcall_result_with_scores.json'
    with open(test_file, 'r') as file:
        data = json.load(file)

    samples = []

    for sample in tqdm(data):
        correct_tool_use_and_response(sample['gt'], sample['llm_response'], sample['tools'])