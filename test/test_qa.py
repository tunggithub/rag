import json
import random
import time
import requests
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from datasets import load_dataset
from langchain_openai import ChatOpenAI
from tqdm import tqdm


def random_subsequence(input_list, n):
    # Check if the length of the list is less than n
    if len(input_list) < n:
        raise ValueError("Length of the input list must be at least n.")

    # Choose a random starting index such that a subsequence of length n can be obtained
    start_index = random.randint(0, len(input_list) - n)

    # Calculate a random length for the subsequence that is at least n

    # Extract the subsequence
    return input_list[start_index : start_index + n]

class QnADataset():
    def __init__(self):
        super().__init__()
        # countering the effect of setting seed for task orchestration from validators
        random.seed(None)
        seed = random.randint(0, 10000)
        wiki = load_dataset("wikipedia", name="20220301.en", split="train", trust_remote_code=True)
        
        self.datasets = {"wiki": iter(wiki.shuffle(seed=seed))}
        

    def __next__(self):
        # print("Retrieving Q&A data from dataset...")
        # countering the effect of setting seed for task orchestration from validators
        random.seed(None)
        while True:
            try:
                dname, ds = random.choice(list(self.datasets.items()))
                text = next(ds)["text"]

                # Check if the text is not empty or does not consist only of newline characters
                if text.strip():
                    return {"text": text}

            except Exception as e:
                print(f"HuggingFace issue ... {e}")
                # time.sleep(15)

class Validator():
    def __init__(self):
        self.sentence_transformer = SentenceTransformer('BAAI/bge-small-en-v1.5')
        self.api_url = "http://localhost:8000/qa-task"
        self.validation_llm_url = "https://polite-partly-sunbird.ngrok-free.app/v1"
        self.test_sets = QnADataset()

    def get_next_sample(self):
        n_texts = random.choice([2, 3, 5])
        all_chunks, correct_chunks = self.generate_random_texts(n_texts=n_texts)
        texts = [d["context"] for d in correct_chunks]
        satisfied_with_question = False
        loop_count = 0
        while not satisfied_with_question and loop_count < 11:
            loop_count += 1
            selected_text = '\n'.join(texts) 
            question = self.get_question_for_text(text=selected_text)
            satisfied_with_question = self.check_question_for_alignment_with_text(
                question,
                text=selected_text
            )
        
        return question, all_chunks, correct_chunks
    
    def run(self):
        num_sample = 500

        samples = []
        for i in tqdm(range(num_sample)):
            try:
                question, all_chunks, correct_chunks = self.get_next_sample()
                texts = [d["context"] for d in correct_chunks]
                print("-----1")
                selected_text = '\n'.join(texts) 
                query_text = f"""Given the following CONTEXT:

                    ```{selected_text}```
            
                    Please provide the user with an answer to their question: {question}.
                    Response: """
                response_gen = self.validation_llm(
                    query_text, temperature=0.8, max_new_tokens=2000
                )
                print("-----2")
                start = time.time()
                answer, citations = self.call_api(question, all_chunks)
                process_time = time.time() - start
                print("-----3")
            except Exception as e: 
                print(e)
            sample = {
                "write_question": question,
                "all_chunks": all_chunks,
                "correct_chunks": [chunk["source"] for chunk in correct_chunks],
                "write_answer": response_gen,
                "answer": answer,
                "citations": citations,
                "process_time": process_time
            }
            samples.append(sample)
            with open("./test_results/test_qa_result.json", "w") as file:
                # Write the data to the file
                json.dump(samples, file)
    
    
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
    
    def get_question_for_text(self, text: str) -> str:
        input_text = f"""
            Ask a question that is unique to this text, making sure to contain key phrases from the text such that the question can be answered by using this text and not other texts. Here is the text:
            ``` 
                {text}
            ```
            Best Question:
        """
        question = self.validation_llm(input_text, temperature=0.8)
        return question
    
    def generate_random_texts(self, n_texts: int = 3) -> [List[str], List[str]]:
        text_splitter = RecursiveCharacterTextSplitter(
            # Set a really small chunk size, just to show.
            chunk_size=random.choices([256, 512, 1024, 2048])[0],
            chunk_overlap=0,
            length_function=len,
            is_separator_regex=False,
        )

        # get n random data
        # countering the effect of setting seed for task orchestration from validators
        random.seed(None)
        output = []
        n_chunks_to_use = random.randint(1, 4)
        chunks_to_use = []
        
        text_count = 0
        for _ in range(200):
            text = next(self.test_sets)["text"]
            split_text = text_splitter.split_text(text)
            if len(split_text) < 4:
                continue
            else:
                text_count += 1
                
            for stext in split_text:
                h = random.getrandbits(128)
                source = f"bitagent.source.{h}"
                output.append({"source": source, "context": stext})
            if not chunks_to_use:
                chunks_to_use = random_subsequence(output, n_chunks_to_use)
            if len(output) >= n_texts:
                break
        return output, chunks_to_use
    
    def check_question_for_alignment_with_text(self, question: str, text: str):
        embeddings = self.sentence_transformer.encode([question, text], normalize_embeddings=True)
        score = embeddings[0] @ embeddings[1].T
        if score < 0.4:
            return False

        # bunch of common things we have seen that are too vague of questions
        if "What is the main " in question:
            return False

        if "What are the main " in question:
            return False

        if "author's" in question:
            return False

        if len(question) < 20:
            return False

        input_text = f"""
            Given this Question:
            ```
            {question}
            ```
            And this Context: 
            ```
            {text}
            ```
            Is the provided Question a strongly relevant question for the provided Context that is not a general question that can be ambiguous when selecting from a long list of text options.  In other words, are there enough details in the Question linked explicitly to the Context? Only respond with yes or no, no other words:
        """
        yes_or_no = self.validation_llm(input_text)
        if yes_or_no.strip().lower() == "yes":
            return True
        return False
    
    def call_api(self, prompt, datas):
        payload = {
            "prompt": prompt,
            "datas": datas
        }
        headers = {
            'Content-Type': 'application/json'
        }
        json_payload = json.dumps(payload)
        with open('data.json', 'w') as f:
            json.dump(payload, f)

        response = requests.post(self.api_url, data=json_payload, headers=headers)
        answer = eval(response.text)["response"]
        citations = eval(response.text)["citations"]
        return answer, citations
    
if __name__=="__main__":
    Validator().run()