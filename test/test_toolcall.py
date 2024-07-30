import random
from datasets import load_dataset
import json
import re
from typing import List, Dict, Any
from strenum import StrEnum
from typing import Dict, List
from pydantic import BaseModel, Field
from tqdm import tqdm
from langchain_openai import ChatOpenAI
import ast
from pydantic import ValidationError
from sentence_transformers import SentenceTransformer


REWRITE_PROPMT = """Please rewrite the following text, ensuring to maintain the original meaning and nuances but altering the sentence structures, vocabulary, and overall presentation. 
The goal is to produce a version of the text that conveys the same information and sentiments as the original, but in a fresh and distinct manner. 
Avoid summarizing or omitting any details; instead, focus on presenting the same concepts and messages in a new light.
            
Rewrite this text: {query}
            
Rewritten text: """

REWRITE_TOOL_PROMPT = "Modify the function call to have different arguments. Your response should only be the modified function. You should not use the same argument values. The arguments should be valid in reference to the other argument values\n Given the function call:\n{tool_call}. Modified function call: "

REWRITE_TOOL_USER_PROMPT = "You rewrite questions to make sense when paired with a function call. The rewritten question will need to be changed to match the arguments of the function call. You should change the phrasing of the question up. Your response should be the rewritten question.\nFunction call:\n{tool_call} \n Question: {user}\n Question:"

REWRITE_TOOL_ASSISTANT_PROMPT = """Input:
User Question: {user}
Tool Call: {tool_call}
Incorrect Answer: {assistant}
Task: Rewrite the incorrect answer to accurately reflect the result of the given Tool call. Also, modify the wording to ensure it is different from the original, it should be concise. Output only the revised answer."""


def clean_text(text):
    text = text.replace("<|endoftext|>", "")
    text = text.replace("ASSISTANT: <functioncall>", "TOOL CALL: ")
    text = text.replace("FUNCTION RESPONSE", "TOOL RESPONSE")
    text = text.replace("  ", " ")
    return text.strip()


type_mapping = {
    "string": str,
    "integer": int,
    "number": (int, float),  # Allow both int and float for 'number'
    "boolean": bool,
    "array": List,
    "dictionary": Dict,
    "object": Dict,  # Handle nested objects as dictionaries
}


class ChatRole(StrEnum):
    """One of ASSISTANT|USER to identify who the message is coming from."""

    ASSISTANT = "assistant"
    USER = "user"
    TOOL_CALL = "tool call"
    TOOL_RESPONSE = "tool response"

class Tool(BaseModel):
    """
    Attributes:
    - name: str
    - description: str
    - arguments: dict where the key is the name of the argument and the value is a dict containing the keys (required:bool, type:str, description:str)
    """
    name: str
    description: str
    arguments: Dict[str, Any]
    
    def to_dict(self):
        return self.dict()
    
def validate_tool_call(tool: Tool, tool_call: Dict[str, Any]) -> bool:
    try:
        # Validate the tool call structure
        tool_call_validated = ToolCall(**tool_call)
        
        # Check if the tool call name matches the tool name
        if tool_call_validated.name != tool.name:
            # bt.logging.warning(f"Tool name mismatch: {tool_call_validated.name} != {tool.name}")
            return False
        
        # Check arguments
        for arg_name, arg_schema in tool.arguments.items():
            if arg_schema['required'] and arg_name not in tool_call_validated.arguments:
                # bt.logging.warning(f"Missing required argument: {arg_name}")
                return False
            if arg_name in tool_call_validated.arguments:
                expected_type = type_mapping.get(arg_schema['type'])
                if expected_type is None:
                    # bt.logging.warning(f"Unknown type for argument {arg_name}: {arg_schema['type']}")
                    return False
                
                # Handle nested objects
                if expected_type == dict:
                    if not isinstance(tool_call_validated.arguments[arg_name], dict):
                        # bt.logging.warning(f"Argument {arg_name} has incorrect type. Expected {expected_type}, got {type(tool_call_validated.arguments[arg_name])}")
                        return False
                else:
                    if not isinstance(tool_call_validated.arguments[arg_name], expected_type):
                        # bt.logging.warning(f"Argument {arg_name} has incorrect type. Expected {expected_type}, got {type(tool_call_validated.arguments[arg_name])}")
                        return False
        
        # All checks passed
        return True
    except ValidationError as e:
        # bt.logging.warning(f"Validation error: {e}")
        return False
    
    
class ToolCall(BaseModel):
    name: str
    arguments: Dict[str, Any]


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
    
def find_msgs_before_tool_call(messages: List[ChatMessage]):
    result = []
    for msg in messages:
        if msg.role == 'tool call':
            break
        result.append(msg)
    return result

def messages_from_list(data_list: List[Dict[str, str]]):
    messages = [ChatMessage.from_dict(item) for item in data_list]
    return messages

def split_dialogue(text) -> List[ChatMessage]:
    # Define a pattern to match the roles and capture messages
    pattern = r"(USER|ASSISTANT|TOOL CALL|TOOl RESPONSE): (.*?)(?=\s*(USER|ASSISTANT|TOOL CALL|TOOL RESPONSE):|$)"

    # Find all matches in the text using the pattern
    matches = re.findall(pattern, text, re.DOTALL)

    # Create a list of dictionaries based on the matches
    dialogue_list = [{"role": role.lower(), "content": message.strip().replace('\'','')} for role, message, _ in matches]
    
    for message in dialogue_list:
        if not message['role']:
            raise ValueError("There is a message with no role.")
     
    return messages_from_list(dialogue_list)

def parse_multiple_space_sep_json(json_str):
    """
    Parses a string containing multiple JSON objects separated by whitespace.
    
    {} {} -> [{},{}]
    """
    results = []
    start = 0
    json_str = json_str.strip()  # Remove leading and trailing whitespace
    while start < len(json_str):
        # Find the start of a JSON object
        start = json_str.find('{', start)
        if start == -1:  # No more JSON object
            break
        try:
            obj, index = json.JSONDecoder().raw_decode(json_str[start:])
            results.append(obj)
            start += index
            while start < len(json_str) and json_str[start] in ' \t\n\r':  # Skip whitespace
                start += 1
        except json.JSONDecodeError:
            # Move start forward and try again
            start += 1
    return results

def detect_type(value: Any) -> str:
    type_mapping = {
        int: 'integer',
        float: 'number',
        str: 'string',
        bool: 'boolean',
        list: 'array',
        dict: 'object'
    }
    return type_mapping.get(type(value), 'string')

def add_extra_arguments(tool_call: Dict[str, Any], tools: List[Tool]):
    # Find the tool in the list
    tool_name = tool_call['name']
    arguments = tool_call.get('arguments', {})
    
    for tool in tools:
        if tool.name == tool_name:
            for arg_name, arg_value in arguments.items():
                if arg_name not in tool.arguments:
                    # Detect the type of the argument
                    arg_type = detect_type(arg_value)
                    # Add the new argument to the tool's schema
                    tool.arguments[arg_name] = {
                        'required': False, # assume false
                        'type': arg_type,
                        'description': arg_name
                    }
            break

def json_schema_to_pydantic_tool(schema: dict) -> Tool:
    tool_name = schema.get("name", "")
    tool_description = schema.get("description", "")

    schema_parameters = schema.get("parameters", {})
    if not schema_parameters:
        schema_parameters = {}
    properties = schema_parameters.get("properties", {})
    required_params = schema_parameters.get("required", [])
    if isinstance(required_params, bool):
        required_params = list(properties.keys()) if required_params else []
    elif not isinstance(required_params, list):
        required_params = []
    parameters = {}
    for param_name, param_info in properties.items():
        if param_name == "required":
            continue
        parameters[param_name] = {
            "required": param_name in required_params,
            "type": param_info.get("type", ""),
            "description": param_info.get("description", ""),
        }
    return Tool(name=tool_name, description=tool_description, arguments=parameters)

def find_first_tool_call(messages: List[ChatMessage]):
    for msg in messages:
        if msg.role == 'tool call':
            return msg

class ToolCallData(BaseModel):
    messages: List[ChatMessage]
    tools: list[Tool]

class ToolDataset():
    def __init__(self):
        super().__init__()
        # countering the effect of setting seed for task orchestration from validators
        random.seed(None)
        seed = random.randint(0, 1000)
        glaive_ds = load_dataset("glaiveai/glaive-function-calling-v2", split="train")

        self.datasets = {
            "glaive": iter(glaive_ds.shuffle(seed=seed)),
        }

    def __next__(self):
        # countering the effect of setting seed for task orchestration from validators
        count = 0
        while count < 25:
            count += 1
            try:
                random.seed(None)
                dname, ds = random.choices(list(self.datasets.items()), [10])[0]
                data = next(ds)
                if dname == "glaive":
                    system_prompt = data["system"].replace("SYSTEM: ", "")
                    if "following functions" not in system_prompt:
                        continue

                    chat_history = clean_text(data["chat"])
                    tools = parse_multiple_space_sep_json(
                        system_prompt.replace(
                            "You are a helpful assistant with access to the following functions. Use them if required - ",
                            "",
                        )
                    )
                    tools = [json_schema_to_pydantic_tool(tool) for tool in tools]
                    messages = split_dialogue(chat_history)

                    # Add arguments that werent defined in schema to the tool
                    for msg in messages:
                        if msg.role == "tool call":
                            tool_call = None
                            if isinstance(msg.content, str):
                                tool_call = json.loads(msg.content)
                            else:
                                tool_call = msg.content
                            
                            add_extra_arguments(tool_call, tools) 

                    
                    return ToolCallData(messages=messages, tools=tools)
            except Exception as e:
                print(f"Issue getting tool call from dataset ... {e}")

class Validator():
    def __init__(self):
        self.api_url = "http://localhost:8000/conversation"
        self.validation_llm_url = "https://polite-partly-sunbird.ngrok-free.app/v1"
        self.test_sets = ToolDataset()
        self.sentence_transformer = SentenceTransformer('BAAI/bge-small-en-v1.5')

    def run(self):
        num_sample = 500

        samples = []
        for i in tqdm(range(num_sample)):
            data: ToolCallData = next(self.test_sets)
            messages = data.messages
            filtered_msgs = []
            seen_tool_call = False
            for msg in messages:
                filtered_msgs.append(msg)
                if seen_tool_call: # want to do break after to include the assistant response
                    break
                if msg.role == 'tool call':
                    seen_tool_call = True
            data.messages = filtered_msgs

            user = data.messages[0].content
            assistant = data.messages[-1].content
            count = 0

            while count < 10:
                count += 1
                if find_first_tool_call(data.messages):
                    tool_call = find_first_tool_call(data.messages).content
                    rewritten_tool_call = self.validation_llm([{"role": "user", "content": REWRITE_TOOL_PROMPT.format(tool_call=tool_call)}], max_new_tokens=1000, temperature=1.2)
                    try: # check that the tool call can be loaded, and that it's valid
                        try:
                            new_tool_call = json.dumps(json.loads(rewritten_tool_call))
                            tool_call_dict = json.loads(rewritten_tool_call)
                        except:
                            new_tool_call = json.dumps(ast.literal_eval(rewritten_tool_call))
                            tool_call_dict = ast.literal_eval(rewritten_tool_call)
                        for tool in data.tools:
                            if tool.name == tool_call_dict['name']:
                                if not validate_tool_call(tool, tool_call_dict):
                                    raise Exception('The tool call is not valid')
                    except Exception as e:
                        # bt.logging.warning(f'An error occured while rewriting the tool call {e}')
                        count = 11
                        continue
                    
                    new_user = self.validation_llm([{"role": "user", "content": REWRITE_TOOL_USER_PROMPT.format(tool_call=new_tool_call, user=user)}], max_new_tokens=1000, temperature=1)
                    if not self.check_rewrite_alignment(new_user, user):
                        raise Exception(f"User rewrite is not in alignment\nOriginal: {user}\n Rewrite: {new_user}")
                    
                    new_assistant = self.validation_llm([{"role": "user", "content": REWRITE_TOOL_ASSISTANT_PROMPT.format(tool_call=new_tool_call, user=new_user, assistant=assistant)}], max_new_tokens=1000, temperature=1).split("(")[0] # sometimes it adds an explanation in paranthesis
                    if not self.check_rewrite_alignment(new_assistant, assistant):
                        raise Exception(f"Assistant rewrite is not in alignment\nOriginal: {assistant}\n Rewrite: {new_assistant}")
                    
                    data.messages[0].content = new_user
                    data.messages[-1].content = new_assistant
                    
                    data = ToolCallData(messages=data.messages, tools=data.tools)
                    messages_before_call = find_msgs_before_tool_call(data.messages)
                    if messages_before_call[-1].role == "assistant":
                        messages_before_call = messages_before_call[:-1]
                    
                    return messages_before_call, data.tools, data
                else:
                    new_user = self.validation_llm(REWRITE_PROPMT.format(query=user))
                    if not self.check_rewrite_alignment(new_user, user):
                        raise Exception(f"User rewrite is not in alignment\nOriginal: {user}\n Rewrite: {new_user}")
                    
                    new_assistant = self.validation_llm(REWRITE_PROPMT.format(query=assistant))
                    if not self.check_rewrite_alignment(new_assistant, assistant):
                        raise Exception(f"Assistant rewrite is not in alignment\nOriginal: {assistant}\n Rewrite: {new_assistant}")
                    
                    data.messages[0].content = new_user
                    data.messages[-1].content = new_assistant
                    if messages_before_call[-1].role == "assistant":
                        messages_before_call = messages_before_call[:-1]
                    
                    return messages_before_call, data.tools, data

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

    def check_rewrite_alignment(self, original: str, rewrite: str) -> bool:
        embeddings = self.sentence_transformer.encode([original, rewrite], normalize_embeddings=True)
        score = embeddings[0] @ embeddings[1].T
        
        if score > 0.98:
            return False
        
        if score < 0.2:
            return False

        if len(rewrite) > 2 * len(rewrite):
            return False
        
        if len(rewrite) < 0.25 * len(rewrite):
            return False
        
        return True

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

if __name__=="__main__":
    a =  Validator().run()