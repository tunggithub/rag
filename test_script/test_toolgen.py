from test_toolcall import ToolDataset, ToolCallData, find_first_tool_call
from tqdm import tqdm
from langchain_openai import ChatOpenAI
import json
import requests
import time

import os
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"]="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"

class Validator():
    def __init__(self):
        self.api_url = "http://localhost:8000/tool-generation"
        self.validation_llm_url = "https://polite-partly-sunbird.ngrok-free.app/v1"
        self.test_sets = ToolDataset()

    def run(self):
        num_sample = 500

        samples = []
        error_samples = []
        for i in tqdm(range(num_sample)):
            data: ToolCallData = next(self.test_sets)

            if not any(msg for msg in data.messages if msg.role == 'tool call'):
                continue
            query = data.messages[0].content
            
            rewrite_prompt = f"""Please rewrite the following text, ensuring to maintain the original meaning and nuances but altering the sentence structures, vocabulary, and overall presentation. 
            The goal is to produce a version of the text that conveys the same information and sentiments as the original, but in a fresh and distinct manner. 
            Avoid summarizing or omitting any details; instead, focus on presenting the same concepts and messages in a new light.
            
            Rewrite this text: {query}
            
            Rewritten text: """
            
            rewritten_query = self.validation_llm(rewrite_prompt, max_new_tokens=200)
            try:
                first_tool_call = json.loads(find_first_tool_call(data.messages).content)
            except Exception as e:
                # bt.logging.error(f"first tool call error {e}")
                continue

            tool = [tool for tool in data.tools if tool.name == first_tool_call['name']][0]

            try:
                start = time.time()
                llm_response = self.call_api(rewritten_query)
                process_time = time.time()-start
                sample = {
                    "rewritten_query": rewritten_query,
                    "tool": tool.to_dict(),
                    "llm_response": eval(llm_response),
                    "process_time": process_time
                }
                samples.append(sample)
                with open("./test_results/test_toolgen_result.json", "w") as file:
                        # Write the data to the file
                        json.dump(samples, file)
            except:
                sample = {
                    "rewritten_query": rewritten_query,
                    "tool": tool.to_dict(),
                }
                error_samples.append(sample)
                with open("./test_results/test_toolgen_error_samples.json", "w") as file:
                        # Write the data to the file
                        json.dump(error_samples, file)




    def call_api(self, prompt):
        payload = {
            "prompt": prompt
        }
        headers = {
            'Content-Type': 'application/json'
        }
        json_payload = json.dumps(payload)
        response = requests.post(self.api_url, data=json_payload, headers=headers)
        llm_response = eval(response.text)["response"]
        return llm_response

    def validation_llm(self, messages, max_new_tokens = 160, temperature=0.7):
        if isinstance(messages, str):
            messages = [{"role":"user","content":messages}]
        llm = ChatOpenAI(
            openai_api_key="EMPTY",
            openai_api_base=self.validation_llm_url,
            model_name="thesven/Mistral-7B-Instruct-v0.3-GPTQ",
            max_tokens = max_new_tokens,
            temperature = temperature,
        )
        return llm.invoke(messages).content.strip()
    
if __name__ == "__main__":
    Validator().run()