import time
import torch
from models import LMModel
from tqdm import tqdm
from config import ExperimentConfig
from logger import ExperimentLogger, ResultSaver
from processor import VQuestionProcessor
from dataset import get_dataset_class
from utils import set_timeout_for_run

def run_mllm_experiment(llm_model: LMModel, config: ExperimentConfig) -> None:
    assert llm_model is not None, "llm_model must be provided"
    model_name = llm_model.model_name or "unknown_model"
    save_name = config.save_name
    logger = ExperimentLogger(config, save_name)
    vprocessor = VQuestionProcessor(config, llm_model)
    
    
    dataset_class = get_dataset_class(config.dataset, config.n_questions, config.start_idx, config.seed, config.subset)
    dataset = dataset_class.load()
    
    response_dict = {}
    uncertainty_dict = {} if config.record_uncertainty else None
    print("Config dict:", config.__dict__, '\n')
    
    try:
        dataset_len = len(dataset)
        for idx_q in tqdm(range(dataset_len), desc="Processing questions"):
            start_time = time.time()
            example = dataset[idx_q]
            parse_result = dataset_class.parse(example)
            
            # other_items can be answer, or other items
            image, question, *other_items = parse_result 
            agent_contexts, agent_uncertainties = vprocessor.process(image, question, qid=idx_q)

            # add question id to the key to avoid repetitive questions
            q_key = f'{question}_{idx_q:03d}'
            response_dict[q_key] = (agent_contexts, *other_items)
            if config.record_uncertainty and agent_uncertainties is not None:
                uncertainty_dict[q_key] = (agent_uncertainties, other_items[0])
            end_time = time.time()
            logger.log_question_completion(idx_q, dataset_len, end_time - start_time)
    except Exception as e:
        logger.log_error(str(e), agent_contexts, idx_q)
        raise
    finally:
        if not config.debug:
            if config.record_uncertainty:
                result = ResultSaver.save_results(response_dict, save_name, uncertainty_dict)
                if isinstance(result, tuple):
                    output_file, uncertainty_file = result
                    logger.log_uncertainty_save(uncertainty_file)
                else:
                    output_file = result
            else:
                output_file = ResultSaver.save_results(response_dict, save_name)
            logger.log_experiment_completion(output_file)


def main(model_name="LlaVA1.6V-13Bs", strategy="MAD", dataset='scienceqa', attn_topp=0.4, subset='', seed=0, n_agents=3, n_rounds=2, start_idx=0, r0_path=None,
         n_questions=10, device_map='auto', max_new_tokens=1024, reasoning_method='CoT',
         verbose=False, debug=False, save_name_prefix='', semantic_c=False, summarize_model='', record_uncertainty=False,
         slurm_run=False, job_end_time=None):
    
    torch.cuda.empty_cache()
    config = ExperimentConfig(
        model_name=model_name,
        r0_path=r0_path,
        dataset=dataset,
        subset=subset,
        attn_topp=attn_topp,
        reasoning_method=reasoning_method,
        record_uncertainty=record_uncertainty,
        n_agents=n_agents,
        n_rounds=n_rounds,
        n_questions=n_questions,
        seed=seed,
        start_idx=start_idx,
        strategy=strategy,
        max_new_tokens=max_new_tokens,
        save_name_prefix=save_name_prefix,
        semantic_c=semantic_c,
        verbose=verbose,
        debug=debug,
        slurm_run=slurm_run,
        job_end_time=job_end_time,
        summarize_model=summarize_model,
        device_map=device_map,
    )
    
    print('Save name: ', config.save_name, '\n')
    llm_model = LMModel.from_name(model_name, device_map=device_map, max_new_tokens=max_new_tokens)
    if config.slurm_run:
        assert config.job_end_time is not None, "job_end_time must be provided"
        run_func_with_timeout = set_timeout_for_run(int(config.job_end_time - time.time()), run_mllm_experiment)
        run_func_with_timeout(llm_model=llm_model, config=config)
    else:
        run_mllm_experiment(llm_model=llm_model, config=config)

if __name__ == "__main__":
    import fire
    fire.Fire(main)

    
    