system_prompt = "You are an excellent assistant in building function in programming. Users will provide you with a task context where your job is to assist them by constructing a function to solve their problem. Just provide the function name and its parameters, without needing to write the actual function code."

user_prompt_template = """This is an example that you can refer to:
Context: Greetings! I find myself in a situation where I require the exchange of 1000 USD into Euros. Could you lend a hand in facilitating this conversion? I appreciate your assistance.
Answer:
```json
{{
    "name": "convert_usd_to_eur",
    "arguments": {{
        "amount": {{
            "description": "Amount in USD to be converted to Euros",
            "required": true,
            "type": "number"
        }},
        "exchange_rate": {{
            "description": "Current exchange rate from USD to Euros",
            "required": false,
            "type": "number"
        }}
    }},
    "description": "Tool to convert money from usd to eur"
}}
```
Create a function based on the context below. 

Context: {context}
Answer:

The output will be returned in JSON format with the 3 fields name, description and arguments as below:
name: A string represented function, separated by underscore and started by verb. For example, 'calculate_tip' is a good function name
description: Brief description about function
arguments: A dictionary with keys as argument names, Each arguments should be lowercase, noun and separated by underscore if neccessary. Value for each field in arguments must be a dict contains 3 keys required, type and description.

    description: A string explaining the meaning of the argument
    required: A boolean indicating whether it's necessary for the user to provide the argument or if a default value can be set
    type: A string indicate the type of argument. It should be integer, float, boolean.
"""

