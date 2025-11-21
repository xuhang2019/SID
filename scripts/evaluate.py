import json
import pandas as pd
from utils import batch_evaluate, parse_answer_mmlupro
from utils import find_latest_result



def evaluate_from_json(gt_path, pred_path, task_name, gt_key="answer", pred_key="output", return_detail=False):
    with open(gt_path, 'r') as f:
        gt_data = json.load(f)
    with open(pred_path, 'r') as f:
        pred_data = json.load(f)

    if isinstance(gt_data, dict):
        gt_list = [v[gt_key] for v in gt_data.values()]
    else:
        gt_list = [v[gt_key] for v in gt_data]

    if isinstance(pred_data, dict):
        pred_list = [v[pred_key] for v in pred_data.values()]
    else:
        pred_list = [v[pred_key] for v in pred_data]

    acc, results = batch_evaluate(gt_list, pred_list, task_name)
    if return_detail:
        return acc, results, gt_list, pred_list
    return acc


def evaluate_mmlupro_result(result_file, task_name='mmlupro', agent_idx=0, round_idx=-1):

    with open(result_file, 'r') as f:
        data = json.load(f)
    
    gt_list = []
    pred_list = []
    
    for question, (agent_contexts, answer) in data.items():
        gt_list.append(answer)
        last_response = agent_contexts[agent_idx][round_idx]['content']
        pred_list.append(last_response)
    
    acc, results = batch_evaluate(gt_list, pred_list, task_name)
    return acc, results, gt_list, pred_list


def evaluate_commonsenseqa_result(result_file, task_name='commonsenseqa'):

    with open(result_file, 'r') as f:
        data = json.load(f)
    
    gt_list = []
    pred_list = []
    
    for question, (agent_contexts, answer) in data.items():
        gt_list.append(answer)
        # Take the last response from the last agent
        last_response = agent_contexts[0][-1]['content']
        pred_list.append(last_response)
    
    acc, results = batch_evaluate(gt_list, pred_list, task_name)
    return acc, results, gt_list, pred_list


def evaluate_alpacaeval_result(result_file):

    with open(result_file, 'r') as f:
        data = json.load(f)
    
    print(f"AlpacaEval results loaded from {result_file}")
    print(f"Total questions: {len(data)}")
    

    return {
        'total_questions': len(data),
        'file_path': result_file
    }



def evaluate_latest_result(task_name):

    if task_name == 'mmlupro':
        pattern = "mmlupro_*.json"
        return evaluate_mmlupro_result(find_latest_result(pattern))
    elif task_name == 'commonsenseqa':
        pattern = "commonsenseqa_*.json"
        return evaluate_commonsenseqa_result(find_latest_result(pattern))
    elif task_name == 'alpacaeval':
        pattern = "alpacaeval_*.json"
        return evaluate_alpacaeval_result(find_latest_result(pattern))
    else:
        raise ValueError(f"Unknown task: {task_name}")


def evaluate_mmlupro_df(df, extra_dict={}):
    # df is a processed df from batch_load_json_to_csv
    # df has the following columns:
    # ['question', 'gt_answer', 'agt1_r1_resp', 'agt1_r1_answer', 'agt1_r1_correct', 'agt1_r2_resp', 'agt1_r2_answer', 'agt1_r2_correct', 'agt2_r1_resp', 'agt2_r1_answer', 'agt2_r1_correct', 'agt2_r2_resp', 'agt2_r2_answer', 'agt2_r2_correct']
    # we need to evaluate the accuracy of the first round and the last round
    # the result should be a dict with the following keys:
    result_dict = {}
    
    df_last_colname = str(df.columns[-1])
    n_agents, n_rounds = int(df_last_colname[3]), int(df_last_colname[6])
    
    # 1. Calculate accuracy of the first round and its standard deviation use f"agt{i}_r1_correct"
    for i in range(1, n_agents + 1):
        ...
        
    result_dict.update(extra_dict)
    return result_dict


def evaluate_gpqa(predictions, references):

    assert len(predictions) == len(references), "same number of predictions and references"
    correct = 0
    for pred, ref in zip(predictions, references):
        pred = pred.strip().upper()
        ref = ref.strip().upper()
        if pred == ref:
            correct += 1
    accuracy = correct / len(predictions)
    return {"accuracy": accuracy, "total": len(predictions), "correct": correct}