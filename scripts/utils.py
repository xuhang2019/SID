import ast
import re
import numpy as np
from collections import Counter
import torch
import torch.nn.functional as F
import pickle
import json
import pandas as pd
import os
import shutil
from wrapt_timeout_decorator import timeout
import logging

from config import RESULTS_DIR,RESULTS_TBL_DIR, LOGS_DIR
from evaluate_math import simplify_math_nips, is_equiv

# ================== debug settings =====================
from dataclasses import dataclass
@dataclass
class DebugSettings:
    strategy: str = ''
    dmad_skip_wrong_sbp_idx_2: bool = False
    first_round_res: bool = False
    
    return_df:bool = False


# ================== helper function for running in slurm system =====================

def set_timeout_for_run(timeout_sec, run_func):
    @timeout(timeout_sec)
    def wrapper(*args, **kwargs):
        return run_func(*args, **kwargs)
    print(f"Set timeout for {run_func.__name__} to {timeout_sec} seconds")
    return wrapper

# ================== Physical Move Files =====================
def move_file(src, dst_folder):
    """Move file src to dst_folder, creating dst_folder if needed."""
    if not os.path.exists(src):
        print(f"File not found: {src}")
        return
    os.makedirs(dst_folder, exist_ok=True)
    dst = os.path.join(dst_folder, os.path.basename(src))
    try:
        shutil.move(src, dst)
        print(f">>> Moved {src} -> {dst}")
    except Exception as e:
        print(f">>> Failed to move {src} -> {dst}: {e}")

def archive_experiment_files(archive_paths:list[str], archive_subfolder = "archive"):
    """
    archive_paths: list of json paths (.json suffix only)
    
    For each file in archive_paths:
      - Move the .json file to ../results/archive/
      - Move the corresponding .log file to ../logs/archive/
      - If a .pkl file with the same base name exists in ../results/, move it to ../results/archive/
    """
    for json_path in archive_paths:
        # Move .json file
        move_file(json_path, os.path.join(RESULTS_DIR, archive_subfolder))

        # Move .log file
        json_base = os.path.basename(json_path)
        log_name = os.path.splitext(json_base)[0] + ".log"
        log_path = os.path.join(LOGS_DIR, log_name)
        if os.path.exists(log_path):
            move_file(log_path, os.path.join(LOGS_DIR, archive_subfolder))
        else:
            logging.info(f">>> Log file not found: {log_path}")

        # Move any .pkl file in ../results/ that starts with the base name
        pkl_base = os.path.splitext(json_base)[0]
        pkl_candidates = [f for f in os.listdir(RESULTS_DIR) if f.startswith(pkl_base) and f.endswith(".pkl")]
        for pkl_name in pkl_candidates:
            print(f">>> Find pkl file related to json: {pkl_name}")
            pkl_path = os.path.join(RESULTS_DIR, pkl_name)
            move_file(pkl_path, os.path.join(RESULTS_DIR, archive_subfolder))


# ================== Data Validation =====================

def data_validation(big_df):
    big_df = big_df.copy()
    # ================ model name validation ================
    unique_model_names = big_df['model_name'].unique()

    model_name_map = {}
    for model_name in unique_model_names:
        if 'Llama' in model_name:
            model_name_map[model_name] = 'Llama3.1-8B'
        elif 'Qwen' in model_name:
            model_name_map[model_name] = 'Qwen2.5-7B'
        elif 'llava' in model_name.lower():
            model_name_map[model_name] = 'LlaVA1.6V-13B'
        elif 'GPTOSS' in model_name:
            model_name_map[model_name] = 'GPTOSS-20B'
        elif 'GLM4.1V-9B' in model_name:
            model_name_map[model_name] = 'GLM4.1V-9B'    
        elif 'dryrun' in model_name.lower():
            model_name_map[model_name] = 'dryrun'
        else:
            raise ValueError(f"Unknown model name: {model_name}")
    
    big_df['model_name'] = big_df['model_name'].map(model_name_map)
    logging.info("Model name checked successfully")
    
    # ================ dataset subset splitting ================
    # def extract_dataset_subset(row):
    #     dataset = row['dataset']
    #     # Handle mmluproNone -> mmlupro, subset None
    #     if dataset == 'mmluproNone':
    #         return pd.Series({'dataset': 'mmlupro', 'subset': None})
    #     # Handle mathxx -> math, subset=xx
    #     # All others: subset None
    #     return pd.Series({'dataset': dataset, 'subset': None})
    
    # big_df[['dataset', 'subset']] = big_df.apply(extract_dataset_subset, axis=1)
    logging.info("Dataset subset splitting checked successfully")
    
    return big_df


# ================== Preview Json =====================
def get_r0_contexts(json_path, question:str|int, n_agents=3, r0_end_idx=3, RESP_IDX = 0, pdir = RESULTS_DIR, image=None, debug=False):
    """
        question: int or str
    """
    json_data = load_json(json_path, pdir=pdir)
    
    qs = list(json_data.keys())
    if isinstance(question, int):
        question = qs[question]
    
    r0_contexts = []
    for agt_idx in range(n_agents):
        agt_resp = json_data[question][RESP_IDX][agt_idx][:r0_end_idx]
        if image is not None:
            agt_resp = replace_image_in_context(agt_resp, image)
        r0_contexts.append(agt_resp)
    if debug:
        return r0_contexts, json_data
    else:
        return r0_contexts

# ================== load and pre-process related code =====================

def load_pkl(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)
    
    
def load_json(json_path, pdir = RESULTS_DIR):
    with open(os.path.join(pdir, json_path), 'r') as f:
        return json.load(f)

def count_json_tokens(json_path, pdir = RESULTS_DIR, mode='simple', agt_end_idx=None, q_range=(0, None)):
    # TODO: add a function of count early-exit df token. need to load_df, recommend to  use df to generate a dropped json list.
    
    RESP_IDX = 0
    json_data = load_json(json_path, pdir=pdir)
    keys = list(json_data.keys())
    if mode == 'simple':
        STR_TOKEN_RATIO = 4
        total_tokens = 0
        for key in keys[q_range[0]:q_range[1]]:
            for agt_resp in json_data[key][RESP_IDX]:
                total_tokens += len(str(agt_resp[:agt_end_idx])) / STR_TOKEN_RATIO
        return round(total_tokens)
    elif mode == 'advanced':
        STR_TOKEN_RATIO = 4
        total_tokens = 0
        for key in keys[q_range[0]:q_range[1]]:
            for agt_resp in json_data[key][RESP_IDX]:
                total_tokens += len(str(agt_resp)) / STR_TOKEN_RATIO
            total_tokens += len(str(json_data[key][RESP_IDX])) / STR_TOKEN_RATIO
        return round(total_tokens)
    else:
        raise ValueError(f"Unknown mode: {mode}")

def has_uncertainty_file(json_path):
    uncertainty_path = json_path.replace('.json', '_uncertainty.pkl')
    return os.path.exists(uncertainty_path)

def load_json_to_df(json_pattern: str = "mmlupro_3_2_50*_mtokens-8192*.json", use_system_prompt=True, results_dir=RESULTS_DIR, dataset_name = 'mmlu', load_mode='both', verbose=True, debugs=DebugSettings()):
    """
    load_mode:
        - both: load both json and uncertainty pkl
        - json: load only json
    """
    target_json = find_latest_result(json_pattern, results_dir=results_dir)
    if verbose:
        print(f'Loading json data from {target_json}')
    assert target_json is not None, f"No result found for {json_pattern}"
    
    uncertainty_path = None
    if load_mode == 'both':
        uncertainty_path = target_json.replace('.json', '_uncertainty.pkl')
        
        if not os.path.exists(uncertainty_path):
            if verbose:
                print(f"No uncertainty data found!")
            uncertainty_path = None
        else:
            if verbose:
                print(f"Loading uncertainty data from {uncertainty_path}")
       
    df_target = load_mmlupro_to_df(target_json, uncertainty_path=uncertainty_path, use_system_prompt=use_system_prompt, dataset_name=dataset_name, debugs=debugs)
    return df_target    

def load_mmlupro_to_df(json_path, uncertainty_path=None, use_system_prompt=True, dataset_name = 'mmlu', debugs=DebugSettings()):
    """
    # Unified loading of mmlupro format JSON (and optional uncertainty pkl), returns DataFrame
    """
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    json_keys = list(json_data.keys())
    if uncertainty_path:
        with open(uncertainty_path, 'rb') as f:
            uncertainty_data = pickle.load(f)
        uncertainty_keys = list(uncertainty_data.keys())
        assert json_keys == uncertainty_keys, "json and pkl keys are not the same"
    
    rows = []
    for key in json_keys:
        json_data_answer = json_data[key]
        agents_responses = json_data_answer[0]
        gt_choice = json_data_answer[1]
        
        # here, dmad skip wrong sbp idx 2
        if debugs.strategy == 'dmad' and debugs.dmad_skip_wrong_sbp_idx_2:
            agents_responses = agents_responses[:1] + agents_responses[2:]
        
        if dataset_name in ['mmlu', 'mmlupro']:
            row = {
                "question": key,
                "gt_answer": gt_choice,
            }
        elif dataset_name == 'gpqa':
            row = {
                "question": key,
                "gt_answer": gt_choice,
                "category": json_data_answer[2],
            }
        elif dataset_name == 'math':
            gt_answer = simplify_math_nips(gt_choice)
            row = {
                "question": key,
                "gt_answer": gt_answer,
                "category": json_data_answer[2],
                # "level": json_data_answer[3],
            }
        elif dataset_name == 'scienceqa':
            row = {
                "question": key,
                "gt_answer": gt_choice,
            }
        elif dataset_name == 'mmstar':
            row = {
                "question": key,
                "gt_answer": gt_choice,
            }
        else:
            raise ValueError(f"Unknown dataset name: {dataset_name}")
        
        if uncertainty_path:
            agents_uncertainty = uncertainty_data[key][0]
        
        # automatically infer # of agents and # of rounds
        for agt_idx, agent_resps in enumerate(agents_responses):
            k = len(agent_resps) // 2
            for r in range(k):
                resp = agent_resps[2 * r + 2] if use_system_prompt else agent_resps[2 * r + 1]
                prefix = f"agt{agt_idx+1}_r{r+1}"
                row[f"{prefix}_resp"] = resp
                # TODO: The coupling here is too tight, should be properly separated.
                if dataset_name in ['mmlu', 'gpqa', 'mmlupro']:
                    row[f"{prefix}_answer"] = parse_answer_mmlupro(resp['content'])
                    row[f"{prefix}_correct"] = int(row[f"{prefix}_answer"] == gt_choice)
                elif dataset_name in ['scienceqa', 'mmstar']:
                    row[f"{prefix}_answer"] = parse_answer_mmlu(resp['content'][0]['text'], reverse=True)
                    row[f"{prefix}_correct"] = int(row[f"{prefix}_answer"] == gt_choice)
                elif dataset_name == 'math':
                    row[f"{prefix}_answer"] = simplify_math_nips(resp['content'])
                    row[f"{prefix}_correct"] = int(is_equiv(row[f"{prefix}_answer"], gt_answer))
                    
                if uncertainty_path:
                    ucty = agents_uncertainty[agt_idx][r]
                    for metric_key, metric_val in ucty.items():
                        row[f"{prefix}_{metric_key}"] = metric_val
        rows.append(row)
    return pd.DataFrame(rows) 


def batch_load_json_to_csv(patterns = ["2_2", "4_2", "5_2"],
                     json_pattern = "mmlupro_{}_50*_mtokens-1024*.json", 
                     save_pattern = "mmlupro_{}_50_model-Meta-Llama-3.1-8B-Instruct_mtokens-1024"
                     ):

    assert '{}' in json_pattern, "json_pattern must contain '{}' to format the pattern"
    assert '{}' in save_pattern, "save_pattern must contain '{}' to format the pattern"
    
    df_dict = {}
    for pattern in patterns:
        df = load_json_to_df(json_pattern.format(pattern))
        save_csv_name = save_pattern.format(pattern)
        df.to_csv(f"{RESULTS_TBL_DIR}/{save_csv_name}.csv", index=False)    
        df_dict[pattern] = df
    return df_dict
    
def merge_result_mmlupro(file_paths, save_path):
    # merge two part of results, like two dictionary, append the second part to the first part
    
    # return: merged dictionary
    is_json = file_paths[0].endswith('.json')
    is_pkl = file_paths[0].endswith('.pkl')
    if not is_json and not is_pkl:
        raise ValueError("Unknown file type: {file_paths[0]}")
    
    
    merged_dict = {}
    for file_path in file_paths:
        if is_json:
            with open(file_path, 'r') as f:
                data = json.load(f)
        elif is_pkl:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
        else:
            raise ValueError(f"Unknown file type: {file_path}")
        merged_dict.update(data)
        
    if is_json:
        with open(save_path, 'w') as f:
            json.dump(merged_dict, f)
    elif is_pkl:
        with open(save_path, 'wb') as f:
            pickle.dump(merged_dict, f)
    

def load_json_mmlupro(json_path, use_system_prompt=True):
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    
    json_keys = list(json_data.keys())
    rows = []
    
    for key in json_keys:
        json_data_answer = json_data[key]
        
        agents_responses = json_data_answer[0]
        gt_choice = json_data_answer[1]
        
        row = {
            "question": key,
            "gt_answer": gt_choice,
        }
        
        for agt_idx, agent_resps in enumerate(agents_responses):
        # Each agent has 2 * k - 1 elements in responses: system, prompt, resp1, prompt, resp2, ...
            k = len(agent_resps) // 2
            for r in range(k):
                resp = agent_resps[2 * r + 2] if use_system_prompt else agent_resps[2 * r + 1]  # skip prompt, take response
                prefix = f"agt{agt_idx+1}_r{r+1}"
                row[f"{prefix}_resp"] = resp
                row[f"{prefix}_answer"] = parse_answer_mmlupro(resp['content'])
                row[f"{prefix}_correct"] = int(row[f"{prefix}_answer"] == gt_choice)
                
        rows.append(row)
        
    return pd.DataFrame(rows)
    

def load_json_and_pkl_mmlupro(json_path, pkl_path, format="long", use_system_prompt=False):
    # question, answer, agt1_r1_resp, (agt1_r1_resp_ucty1, ...) ,agt2_r1_resp, agt2_r1_resp_ucty1, ..., agt1_r2_resp, agt2_r2_resp,
    print(json_path, pkl_path)
    
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    with open(pkl_path, 'rb') as f:
        pkl_data = pickle.load(f)
        
    json_keys = list(json_data.keys())
    pkl_keys = list(pkl_data.keys())
    assert json_keys == pkl_keys, "json and pkl keys are not the same"
    
    rows = []
    
    for key in json_keys:
        json_data_answer = json_data[key]
        pkl_data_answer = pkl_data[key]
        
        agents_responses = json_data_answer[0]
        gt_choice = json_data_answer[1]
        agents_uncertainty = pkl_data_answer[0]
        
        row = {
            "question": key,
            "gt_answer": gt_choice,
        }
        
        for agt_idx, (agent_resps, agent_ucty) in enumerate(zip(agents_responses, agents_uncertainty)):
        # Each agent has 2 * k - 1 elements in responses: prompt, resp1, prompt, resp2, ...
            k = len(agent_ucty)  # number of rounds
            for r in range(k):
                resp = agent_resps[2 * r + 2] if use_system_prompt else agent_resps[2 * r + 1]
                ucty = agent_ucty[r]  # dict with uncertainty metrics
                prefix = f"agt{agt_idx+1}_r{r+1}"
                row[f"{prefix}_resp"] = resp
                row[f"{prefix}_answer"] = parse_answer_mmlupro(resp['content'])
                row[f"{prefix}_correct"] = int(row[f"{prefix}_answer"] == gt_choice)
                
                for metric_key, metric_val in ucty.items():
                    row[f"{prefix}_{metric_key}"] = metric_val
        
        rows.append(row)
        
    return pd.DataFrame(rows)
        
        
def load_long_from_wide_mmlupro(df_wide):
    df_wide = df_wide.copy()
    id_vars = ['question', 'gt_answer']

    # All agent answer columns
    value_vars = [col for col in df_wide.columns if col.startswith('agt') and '_r' in col]

    # Step 1: Melt to flatten into one column
    df_melted = df_wide.melt(id_vars=id_vars, value_vars=value_vars,
                                var_name="full_key", value_name="value")

    # Step 2: Split into agtX, round, metric triplet
    df_melted[['agt', 'round', 'metric']] = df_melted['full_key'].str.extract(r'(agt\d+)_r(\d+)_(.+)')

    # Step 3: Pivot to restore metrics as columns (rows: question+answer+agt+round) pivot opposite to melt
    df_long = df_melted.pivot_table(
        index=['question', 'gt_answer', 'agt', 'round'],
        columns='metric',
        values='value',
        aggfunc='first'
    ).reset_index()

    return df_long
    

def find_latest_result(pattern, results_dir=RESULTS_DIR, return_mode='max'):
    """
        Find the lastest results
    """
    import glob
    files = glob.glob(os.path.join(results_dir, pattern))
    if not files:
        return None
    
    if return_mode == 'max':
        return max(files, key=os.path.getmtime)
    elif return_mode == 'min':
        return min(files, key=os.path.getmtime)
    elif return_mode == 'list':
        # return sorted by time list
        return sorted(files, key=os.path.getmtime)
    else:
        raise ValueError(f"Unknown return_mode: {return_mode}")


def parse_bullets(sentence):
    bullets_preprocess = sentence.split("\n")
    bullets = []
    for bullet in bullets_preprocess:
        try:
            idx = bullet.find(next(filter(str.isalpha, bullet)))
        except:
            continue
        bullet = bullet[idx:]
        if len(bullet) != 0:
            bullets.append(bullet)
    return bullets


def parse_yes_no(string):
    s = string.lower()
    if "uncertain" in s:
        return None
    elif "yes" in s:
        return True
    elif "no" in s:
        return False
    else:
        return None

def most_frequent(lst):
    if not lst:
        return None
    return Counter(lst).most_common(1)[0][0]

# ================== parse_answer =====================

def parse_answer_mmlu(input_str, reverse=True):
    pattern = r'\((\w)\)'
    matches = re.findall(pattern, input_str)
    if reverse:
        matches = matches[::-1]
    for match_str in matches:
        solution = match_str.upper()
        if solution:
            return solution
    return None

def parse_answer_mmlupro(input_str):
    pattern = r'\((\w)\)'
    matches = re.findall(pattern, input_str)
    if matches:
        return matches[-1].upper()
    # Also supports direct numbers
    pattern_num = r'\b([0-9])\b'
    matches_num = re.findall(pattern_num, input_str)
    if matches_num:
        return matches_num[-1]
    return None

def parse_answer_commonsenseqa(input_str):
    pattern = r'\((\w)\)'
    matches = re.findall(pattern, input_str)
    if matches:
        return matches[-1].upper()
    return None

def parse_answer_math(input_str):
    pattern = r"\{([0-9.,$]*)\}"
    matches = re.findall(pattern, input_str)
    for match_str in matches[::-1]:
        solution = re.sub(r"[^0-9.]", "", match_str)
        if solution:
            return solution
    return solve_math_problems(input_str)

def solve_math_problems(input_str):
    pattern = r"\d+\.?\d*"
    matches = re.findall(pattern, input_str)
    if matches:
        return matches[-1]
    return None


# ================== compute_accuracy =====================

def compute_accuracy_classification(gt, pred_solutions, parse_answer_func):
    if isinstance(pred_solutions, list):
        pred_answers = [parse_answer_func(p) for p in pred_solutions]
        pred_answers = [a for a in pred_answers if a is not None]
        if not pred_answers:
            return 0
        pred_answer = most_frequent(pred_answers)
    else:
        pred_answer = parse_answer_func(pred_solutions)
    if pred_answer is None:
        return 0
    return int(str(gt).strip().upper() == str(pred_answer).strip().upper())

def compute_accuracy_math(gt, pred_solutions, parse_answer_func):
    if isinstance(pred_solutions, list):
        pred_answers = [parse_answer_func(p) for p in pred_solutions]
        pred_answers = [a for a in pred_answers if a is not None]
        if not pred_answers:
            return 0
        pred_answer = most_frequent(pred_answers)
    else:
        pred_answer = parse_answer_func(pred_solutions)
    if pred_answer is None:
        return 0
    try:
        return int(float(gt) == float(pred_answer))
    except:
        return 0


# ================== Image utils =====================
def replace_image_in_context(context, image):
    for message in context:
        if message['role'] == 'user':
            for content in message['content']:
                if content['type'] == 'image':
                    content['image'] = image
    return context


def get_total_running_time(log_path):
    
    with open(log_path, 'r') as file:
        log_text = file.read()
    
    return sum_durations_from_log(log_text)

def sum_durations_from_log(log_text: str) -> dict:
    time_dict = {
        'items': 0,
        'time_hr': 0.0,
    }
    # Use regex to find all durations ending with "s" (seconds)
    durations = re.findall(r'completed in ([\d.]+)s', log_text)
    # Convert to float and sum
    total_seconds = sum(float(d) for d in durations)
    time_dict['items'] = len(durations)
    time_dict['time_hr'] = total_seconds / 3600
    return time_dict

# ================== df and text utils =====================
def df_equal(df1, df2, cols = None):
    if cols is None:
        cols = df1.columns    
    return df1[cols].equals(df2[cols])

def df_compare(df1, df2, cols = None, result_names = ('ori', 'r0_path')):
    return df1[cols].compare(df2[cols], result_names=result_names)


def parse_log_config(log_path, pdir=LOGS_DIR):
    with open(os.path.join(pdir, log_path), 'r') as file:
        line = file.readline()
        return parse_config(line)

def parse_config(line):
    config_dict = {}
    match = re.search(r"Config:\s*(\{.*\})", line)
    if match:
        dict_str = match.group(1)
        config_dict = ast.literal_eval(dict_str)
    else:
        print("No config found.")
    
    return config_dict

def parse_exp_filename(filename: str) -> dict:
    """
    Parses a log filename to extract key information.

    mmlupro_3_3_200_model-Llama3.1-8B_mtokens-1024_strategy-attn_r0d_topp-1.0_reasoning-CoT_time-081410
07.json'

    Args:
        filename: The log filename string.

    Returns:
        A dictionary containing extracted information:
    
    (?P<xxx>[]+?)_ # here ? means non-greedy   
    """
    # Define a regex pattern that captures all the desired parts of the filename
    filename = os.path.basename(filename)
    
    
    pattern = r'^'
    pattern += r'(?P<dataset>[A-Za-z0-9-_]+?)_'          
    pattern += r'(?:(?P<n_agents>\d+)_)?'               
    pattern += r'(?:(?P<n_rounds>\d+)_)?'               
    pattern += r'(?P<n_questions>\d+|None)_'                       
    pattern += r'model-(?P<model_name>[\w.-]+?)_'        
    pattern += r'mtokens-(?P<mtokens>\d+)_'             
    pattern += r'(?:strategy-(?P<strategy>[\w-]+?)_)?'        
    pattern += r'(?:topp-(?P<topp>\d+(?:\.\d+)?)_)?'      
    pattern += r'(?:reasoning-(?P<reasoning>[\w-]+?)_)?'     
    pattern += r'(?:loadr0-(?P<loadr0>True|False)_)?'     
    pattern += r'time-(?P<time>\d+)'                    
    pattern += r'(?:_semantic-(?P<semantic>True|False))?' 
    pattern += r'\.(?:json|log)$'

    # Search the filename for the pattern
    match = re.search(pattern, filename)
    
    # If a match is found, extract the named groups and return a dictionary
    if match:
        data = match.groupdict()
        data['filename'] = filename
        int_keys = ['n_agents', 'n_rounds', 'n_questions', 'mtokens']
        for key in int_keys:
            if data.get(key):
                if data[key] == 'None':
                    data[key] = None
                else:
                    data[key] = int(data[key])
        
        # special case for math
        if data['dataset'].startswith('math'):
            data['subset'] = data['dataset'][len('math'):]
            data['dataset'] = 'math'
        
        return data
    else:
        # If no match is found, return an empty dictionary
        print(f"No match found for {filename}")
        return {}
