import json
from tqdm import tqdm
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer, util
from langchain_text_splitters import CharacterTextSplitter
import torch

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


def compute_correct_citation_rate(llm_citations, selected_sources):
    identified_sources = 0
    for source in llm_citations:
        if source in selected_sources:
            identified_sources += 1
    return identified_sources / len(selected_sources)

def check_relevant_to_provided_content(prompt, completion, selected_datas):
    context = selected_datas[0]['context']
    
    rel_prompt = measure_relevance_of_texts(completion, prompt)
    rel_context = measure_relevance_of_texts(completion, context)

    # check relevance of response to prompt and context
    if rel_context < 0.55 or rel_prompt < 0.55:
       return False
    else:
        return True

def check_ensure_unique_response(prompt, completion, selected_datas, response_gen):
    max_reward = 1.0

    context = selected_datas[0]['context']
    
    
    def _sliding_window_score(completion, context):
        length = torch.tensor([len(c) for c in completion.split("\n")]).max()
        text_splitter = CharacterTextSplitter(
            separator=" ",
            chunk_size=length,
            chunk_overlap=length-1, 
            length_function=len,
            is_separator_regex=False,
        )
        context_windows = [split.page_content for split in text_splitter.create_documents([context])]
        split_completion = completion.split("\n")
        context_and_completions_embeddings = sentence_transformer.encode(context_windows+ split_completion, show_progress_bar=False)
        context_embeddings = context_and_completions_embeddings[:len(context_windows)]
        completions_embeddings = context_and_completions_embeddings[len(context_windows):]
        
        all_scores = []
        for c_embedding in completions_embeddings:
            all_scores.append(
                util.pytorch_cos_sim(c_embedding, context_embeddings)[0].tolist()
            )
        return all_scores
    
    max_cos_score = torch.tensor(_sliding_window_score(completion, context)).max()

    match_prompt = SequenceMatcher(None, completion, prompt).ratio()
    match_context = SequenceMatcher(None, completion, context).ratio()
    match_prompt_gen = SequenceMatcher(None, response_gen, prompt).ratio()
    match_context_gen = SequenceMatcher(None, response_gen, context).ratio()

    max_response_cos_score = torch.tensor(_sliding_window_score(response_gen, context)).max()
    
    if max_response_cos_score > 0.90 or match_prompt_gen > 0.90 or match_context_gen > 0.90:
        return None
    if (match_context > 0.8 and match_context < 1.05*match_context_gen) or (match_prompt > 0.9 and match_prompt < 1.05*match_prompt_gen):
        return None
        
    # check relevance of response to prompt and context
    if completion in context or max_cos_score > 0.90:
        return False

    if match_context > 0.80 or match_prompt > 0.90:
        return False
    else:
        return True

def check_correct_response_provided(completion, response_gen):
    score_gen = measure_relevance_of_texts(completion, response_gen)

    if score_gen < 0.60:
        return False
    else:
        return True

if __name__ == '__main__':
    test_file = 'test_results/test_qa_result.json'
    output_file = 'test_results/test_qa_result_with_scores.json'
    with open(test_file, 'r') as file:
        data = json.load(file)

    samples = []

    for sample in tqdm(data):
        citation_rate = compute_correct_citation_rate(sample['citations'], sample['correct_chunks'])

        selected_chunks = []
        for chunk in sample['all_chunks']:
            if chunk['source'] in sample['correct_chunks']:
                selected_chunks.append(chunk)
        relevent_content = check_relevant_to_provided_content(sample['write_question'], sample['answer'], selected_chunks)
        unique_response = check_ensure_unique_response(sample['write_question'], sample['answer'], selected_chunks, sample['write_answer'])
        correct_response = check_correct_response_provided(sample['answer'], sample['write_answer'])

        sample['citation_rate'] = citation_rate
        sample['relevent_content'] = relevent_content
        sample['unique_response'] = unique_response
        sample['correct_response'] = correct_response
        samples.append(sample)

    with open(output_file, 'w') as file:
        json.dump(samples, file,)