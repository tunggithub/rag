import json 
import os 
from datetime import datetime
from tqdm import tqdm

def calculate_float_points(diff):
        if diff == 0:
            return 5
        elif diff <= 2:
            return 4.5
        elif diff <= 3:
            return 4.0
        elif diff <= 4:
            return 3.5
        elif diff <= 5:
            return 3.0
        elif diff <= 6:
            return 2.5
        elif diff <= 7:
            return 2.0
        elif diff <= 8:
            return 1.5
        elif diff <= 9:
            return 1.0
        else:
            return 0.0
    
def calculate_date_points(diff):
    if diff == 0:
        return 5
    elif diff <= 2:
        return 4.5
    elif diff <= 5:
        return 4.0
    elif diff <= 10:
        return 3.5
    elif diff <= 15:
        return 3.0
    elif diff <= 20:
        return 2.5
    elif diff <= 25:
        return 2.0
    elif diff <= 30:
        return 1.5
    else:
        return 1.0

if __name__ == "__main__":
    vision_qa_result_path = "../visual-qa/test_result.json"
    vision_qa_result = json.load(open(vision_qa_result_path))
    max_reward = 5.0
    rewards = []
    error_case = 0
    for result in vision_qa_result:
        answer = result['answer']
        groundtruth = result['groundtruth']
        is_date = False
        try:
            groundtruth = float(groundtruth)
        except ValueError:
            groundtruth = datetime.strptime(groundtruth, '%m/%d/%Y')
            is_date = True
        
        if is_date:
            response_date = datetime.strptime(answer, '%m/%d/%Y')
            reward = calculate_date_points(abs((response_date - groundtruth).days))
        else:
            try:
                reward = (calculate_float_points(abs(float(answer) - groundtruth)) / 5) * max_reward
            except:
                error_case += 1
                continue
        rewards.append(reward)
    
    reward_avg = sum(rewards) / len(rewards)

    print(f"Average reward: {reward_avg}")
    print(f"Number of error cases: {error_case}")