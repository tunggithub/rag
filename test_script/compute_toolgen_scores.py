import difflib
from sentence_transformers import SentenceTransformer, util
import json
from tqdm import tqdm

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


def compute_scores(expected_tool, miner_tool):
    name_similarity = difflib.SequenceMatcher(None, expected_tool['name'], miner_tool['name']).ratio()

    description_similarity = measure_relevance_of_texts(expected_tool['description'], miner_tool['description'])

    # Score for argument matching
    expected_args = expected_tool['arguments']
    miner_args = miner_tool['arguments']
    total_arg_comparisons = 0
    match_count = 0

    for exp_arg, exp_details in expected_args.items():
        # Find the closest match in miner_tool arguments
        possible_matches = difflib.get_close_matches(exp_arg, miner_args.keys(), n=1, cutoff=0.6)
        if possible_matches:
            gen_arg = possible_matches[0]
            gen_details = miner_args[gen_arg]
            total_arg_comparisons += 3  # For required, type, and description comparisons

            # Check for required match
            if gen_details.get('required', False) == exp_details['required']:
                match_count += 1
            # Check type similarity
            if gen_details.get('type', '') == exp_details['type']:
                match_count += 1
            # Check description similarity
            arg_description_similarity = measure_relevance_of_texts(exp_details['description'], gen_details.get('description', ''))
            if arg_description_similarity > 0.5:  # Threshold for considering the descriptions similar
                match_count += 1

    if total_arg_comparisons > 0:
        match_arg_rate = match_count / total_arg_comparisons
    else:
        match_arg_rate = 0
    return name_similarity, description_similarity, match_arg_rate

if __name__ == "__main__":
    test_file = "test_results/test_toolgen_result.json"
    output_file = "test_results/test_toolgen_result_with_score.json"
    with open(test_file, 'r') as file:
        data = json.load(file)
    samples = []
    for e in tqdm(data):
        for k, v in e['llm_response']['arguments'].items():
            v['required'] = v['required'] == 'True'
        name_similarity, description_similarity, match_arg_rate = compute_scores(e['tool'], e['llm_response'])
        e['scores'] = {
            'name_similarity': name_similarity,
            'description_similarity': description_similarity,
            'match_arg_rate': match_arg_rate
        }
        samples.append(e)
    
    with open(output_file, 'w') as file:
        json.dump(samples, file,)
