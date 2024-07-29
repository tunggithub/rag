import random
import logging
import json
import requests
import time
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from typing import Dict, List
from strenum import StrEnum
from pydantic import BaseModel, Field
from tqdm import tqdm
from langchain_openai import ChatOpenAI


class ChatRole(StrEnum):
    """One of ASSISTANT|USER to identify who the message is coming from."""

    ASSISTANT = "assistant"
    USER = "user"
    TOOL_CALL = "tool call"
    TOOL_RESPONSE = "tool response"

class ChatMessage(BaseModel):
    """A list of previous messages between the user and the model, meant to give the model conversational context for responding to the user's message."""

    role: ChatRole = Field(
        title="One of the ChatRole's to identify who the message is coming from.",
    )
    content: str | dict | list = Field( # TODO the dict/list was added to support json loading the function calls. this should maybe be done inside  a ToolMessage type
        title="Contents of the chat message.",
    )

    @classmethod
    def from_dict(cls, data: Dict[str, str]):
        """Create a ChatMessage object from a dictionary."""
        return cls(role=ChatRole(data['role']), content=data['content'])
    
    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role.value, "content": self.content}

def messages_from_list(data_list: List[Dict[str, str]]):
    messages = [ChatMessage.from_dict(item) for item in data_list]
    return messages

def messages_to_list(messages: List[ChatMessage]):
    return [msg.to_dict() for msg in messages]


class ChatDataset():
    def __init__(self, seed=0):
        super().__init__()
        # countering the effect of setting seed for task orchestration from validators
        
        lmsys = load_dataset('lmsys/lmsys-chat-1m', token="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")
        wizardlm = load_dataset('cognitivecomputations/WizardLM_alpaca_evol_instruct_70k_unfiltered', split="train")
        wizard_vic = load_dataset('cognitivecomputations/wizard_vicuna_70k_unfiltered', split="train")
        safe_ds = load_dataset("PKU-Alignment/PKU-SafeRLHF", split="train")
        beaver_ds = load_dataset("PKU-Alignment/BeaverTails", split="330k_train")

        self.datasets = { 
            "lmsys": iter(lmsys.shuffle(seed=seed)),
            "wizardlm": iter(wizardlm.shuffle(seed=seed)),
            "wizard_vic": iter(wizard_vic.shuffle(seed=seed)),
            "safe": iter(safe_ds.shuffle(seed=seed)),
            "beaver": iter(beaver_ds.shuffle(seed=seed)) 
        }
    
    def wizardlm_formatter(self,row):
        messages = [{'role': 'user', 'content': row['instruction']}, {'role': 'assistant', 'content': row['output']}]
        return messages_from_list(messages)
    
    def wizard_vic_formatter(self,row):
        convos = row['conversations']
        for convo in convos:
            convo['role'] = convo.pop('from')
            convo['content'] = convo.pop('value')
            if convo['role'] == 'human':
                convo['role'] = 'user'
            if convo['role'] == 'gpt':
                convo['role'] = 'assistant'
        return messages_from_list(convos)
    
    def safe_formatter(self, row):
        user = row['prompt']
        if not row['is_response_0_safe']:
            assistant = row['response_0']
        elif not row['is_response_1_safe']:
            assistant = row['response_1']
        
        return messages_from_list([{"role": "user","content":user},{"role":"assistant","content":assistant}])
    
    def beaver_formatter(self, row):
        user = row['prompt']
        assistant = row['response']
        return messages_from_list([{"role": "user","content":user},{"role":"assistant","content":assistant}])
     
    def __next__(self) -> List[ChatMessage]:
        logging.debug("Retrieving chat data from dataset...")
        # countering the effect of setting seed for task orchestration from validators
        random.seed(None)
        for _ in range(5):
            try:
                # dname, ds = random.choices(list(self.datasets.items()), [1,1,1,2,2])[0]
                dname, ds = random.choices(list(self.datasets.items()), [0,1,0,0,0])[0]
                row = next(ds)
                
                if dname == "wizardlm":
                    return self.wizardlm_formatter(row)
                if dname == "wizard_vic":
                    return self.wizard_vic_formatter(row)
                if dname == "safe":
                    return self.safe_formatter(row)
                if dname == "beaver":
                    return self.beaver_formatter(row)
                
                if row['language'] != "English":
                    continue
                conversation = row["conversation"]
                conversation = [{'role': msg['role'], 'content': msg['content']} for msg in conversation]
                
                return messages_from_list(conversation)
            except Exception as e:
                logging.debug(f"Issue getting chat history {e}")

class Validator():
    def __init__(self):
        self.sentence_transformer = SentenceTransformer('BAAI/bge-small-en-v1.5')
        self.api_url = "http://localhost:8000/conversation"
        self.test_sets = ChatDataset()

    def run(self):
        num_sample = 500

        samples = []
        for i in tqdm(range(num_sample)):
            data = next(self.test_sets)
            
            if data[0].role == "user" and data[1].role == "assistant":
                sample = {
                    "query": data[0].content,
                    "assistant": data[1].content,
                    "rewrite_query": self.rewrite(data[0].content),
                    "rewrite_assistant": self.rewrite(data[1].content)
                }

                start = time.time()
                llm_response = self.call_api(sample["rewrite_query"])
                process_time = time.time() - start

                embeddings = self.sentence_transformer.encode([llm_response, sample["rewrite_assistant"]], normalize_embeddings=True)
                similarity = embeddings[0] @ embeddings[1].T
                sample["llm_response"] = llm_response
                sample["similarity"] = str(similarity)
                sample["process_time"] = str(process_time)
                samples.append(sample)
        
        with open("./test_results/test_conversation_result.json", "w") as file:
            # Write the data to the file
            json.dump(samples, file)

    def call_api(self, question):
        payload = {
            "message_history": [
                {
                    "role": "user",
                    "content": question
                }
            ]
        }
        headers = {
            'Content-Type': 'application/json'
        }
        json_payload = json.dumps(payload)
        response = requests.post(self.api_url, data=json_payload, headers=headers)
        llm_response = eval(response.text)["response"]
        return llm_response

    def rewrite(self, text):
        REWRITE_PROMPT = """Please rewrite the following text, ensuring to maintain the original meaning and nuances but altering the sentence structures, vocabulary, and overall presentation. 
The goal is to produce a version of the text that conveys the same information and sentiments as the original, but in a fresh and distinct manner. 
Avoid summarizing or omitting any details; instead, focus on presenting the same concepts and messages in a new light.
        
Rewrite this text: {text}"""
        messages = [{"role":"user","content":REWRITE_PROMPT.format(text=text)}]

        llm = ChatOpenAI(
            openai_api_key="EMPTY",
            openai_api_base="https://polite-partly-sunbird.ngrok-free.app/v1",
            model_name="thesven/Mistral-7B-Instruct-v0.3-GPTQ",
            max_tokens = 160,
            temperature = 0.7,
        )
        return llm.invoke(messages).content.strip()

if __name__ == "__main__":
    Validator().run()
