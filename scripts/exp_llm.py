import time
import torch
from models import LMModel
from tqdm import tqdm
from config import ExperimentConfig
from logger import ExperimentLogger, ResultSaver
from processor import QuestionProcessor
from dataset import get_dataset_class
from utils import set_timeout_for_run

def run_llm_experiment(llm_model: LMModel, config: ExperimentConfig, with_uncertainty: bool = False) -> None:
    assert llm_model is not None, "llm_model must be provided"
    model_name = llm_model.model_name or "unknown_model"
    save_name = config.save_name
    logger = ExperimentLogger(config, save_name)
    processor = QuestionProcessor(config, llm_model)
    
    
    dataset_class = get_dataset_class(config.dataset, config.n_questions, config.start_idx, config.seed, config.subset)
    dataset = dataset_class.load()
    
    response_dict = {}
    uncertainty_dict = {} if with_uncertainty else None
    print(config.__dict__) 
    
    try:
        dataset_len = len(dataset)
        for idx_q in tqdm(range(dataset_len), desc="Processing questions"):
            start_time = time.time()
            example = dataset[idx_q]
            parse_result = dataset_class.parse(example)
            
            # other_items can be answer, or other items
            question, *other_items = parse_result 
            agent_contexts, agent_uncertainties = processor.process(question, with_uncertainty, qid=idx_q)
            response_dict[question] = (agent_contexts, *other_items)
            if with_uncertainty and agent_uncertainties is not None:
                uncertainty_dict[question] = (agent_uncertainties, other_items[0])
            end_time = time.time()
            logger.log_question_completion(idx_q, dataset_len, end_time - start_time)
    except Exception as e:
        logger.log_error(str(e), agent_contexts, idx_q)
        raise
    finally:
        if not config.debug:
            if with_uncertainty:
                result = ResultSaver.save_results(response_dict, save_name, uncertainty_dict)
                if isinstance(result, tuple):
                    output_file, uncertainty_file = result
                    logger.log_uncertainty_save(uncertainty_file)
                else:
                    output_file = result
            else:
                output_file = ResultSaver.save_results(response_dict, save_name)
            logger.log_experiment_completion(output_file)


def main(model_name="Llama3.1-8B", strategy="MAD", dataset='mmlupro', attn_topp=0.4, subset='', seed=0, n_agents=3, n_rounds=2, start_idx=0, r0_path=None,
         n_questions=10, device_map='auto', max_new_tokens=1024, with_uncertainty=False, reasoning_method='CoT',
         verbose=False, debug=False, save_name_prefix='', semantic_c=False, summarize_model='',
         slurm_run=False, job_end_time=None):
    
    torch.cuda.empty_cache()
    config = ExperimentConfig(
        model_name=model_name,
        r0_path=r0_path,
        dataset=dataset,
        subset=subset,
        attn_topp=attn_topp,
        reasoning_method=reasoning_method,
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
    
    print(config.save_name, '\n')
    update_generation_config = {"max_new_tokens": max_new_tokens}
    llm_model = LMModel.from_name(model_name, device_map=device_map, update_generation_config=update_generation_config)
    if config.slurm_run:
        assert config.job_end_time is not None, "job_end_time must be provided"
        run_func_with_timeout = set_timeout_for_run(int(config.job_end_time - time.time()), run_llm_experiment)
        run_func_with_timeout(llm_model=llm_model, config=config, with_uncertainty=with_uncertainty)
    else:
        run_llm_experiment(llm_model=llm_model, config=config, with_uncertainty=with_uncertainty)

if __name__ == "__main__":
    import fire
    fire.Fire(main)

    
    