from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional, Any, Dict


RESULTS_DIR = "../results"
LOGS_DIR = "../logs"
RESULTS_TBL_DIR = "../results_tbl"
FIGS_DIR = "../figs"

# store the original first round json path for each model
r0_path_dict = {}


@dataclass
class ExperimentConfig:
    """Configuration class for experiments """
    model_name: str = 'Llama3.1-8B'
    
    n_agents: int = 3
    n_rounds: int = 2 # standard is 3
    strategy: str = 'MAD'  # SID-c or SID-v
    attn_topp: float = 0.4
    semantic_c: bool = True # default is True
    summarize_model: str = ''
    r0_path: str = '' # for attn_r0 strategy
    reasoning_method: str = 'CoT' 
    record_uncertainty: bool = False
    
    max_new_tokens: int = 1024
    device_map: str = 'auto'
    
    seed: int = 0
    n_questions: int = 100
    dataset: str = 'mmlupro'
    subset: str = ''
    
    start_idx: int = 0
    
    save_name_prefix: str = ''
    verbose: bool = False
    debug: bool = False
    
    slurm_run: bool = False
    job_end_time: Optional[Any] = None
    

    def __post_init__(self):
        self.timestamp = datetime.now().strftime("%m%d%H%M%S") # should on second level
        
        if self.strategy.startswith('attn_r0'):
            assert self.r0_path, "r0_path must be provided for attn_r0 strategy"
        
        load_r0 = True if self.r0_path else False
        if self.r0_path in r0_path_dict:
            self.r0_path = r0_path_dict[self.r0_path]

        clean_model_name = self.model_name.split("/")[-1].replace('/', '_')
        if self.strategy in ['hardthreshold', 'softgate', 'attn_r0', 'attn_r0d']:
            # no semantic_c in the name
            self.save_name = f"{self.save_name_prefix}{self.dataset}{self.subset}_{self.n_agents}_{self.n_rounds}_{self.n_questions}_model-{clean_model_name}_mtokens-{self.max_new_tokens}_strategy-{self.strategy}_topp-{self.attn_topp}_reasoning-{self.reasoning_method}_loadr0-{load_r0}_time-{self.timestamp}"
        elif self.strategy.startswith('MAD'):
            # add summarize_model to the name
            self.save_name = f"{self.save_name_prefix}{self.dataset}{self.subset}_{self.n_agents}_{self.n_rounds}_{self.n_questions}_model-{clean_model_name}_mtokens-{self.max_new_tokens}_strategy-{self.strategy}{self.summarize_model}_reasoning-{self.reasoning_method}_loadr0-{load_r0}_time-{self.timestamp}"
        else:
            self.save_name = f"{self.save_name_prefix}{self.dataset}{self.subset}_{self.n_agents}_{self.n_rounds}_{self.n_questions}_model-{clean_model_name}_mtokens-{self.max_new_tokens}_strategy-{self.strategy}_reasoning-{self.reasoning_method}_loadr0-{load_r0}_time-{self.timestamp}_semantic-{self.semantic_c}"