import os
import logging
import pickle
from config import LOGS_DIR, RESULTS_DIR

class ExperimentLogger:
    """Handles logging setup and operations"""
    def __init__(self, config, save_name: str):
        self.config = config
        self.save_name = save_name
        
        if not config.debug:
            log_file = f"{LOGS_DIR}/{save_name}.log"
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(message)s',
                filename=log_file
            )
        else:
            # In debug mode, only log to console
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(message)s'
            )
            
        logging.info(f"Config: {self.config.__dict__}")
    
    def log_question_completion(self, idx_q: int, total_questions: int, processing_time: float):
        # print("#"*50, f"Question {idx_q}/{total_questions} completed in {processing_time:.2f}s", "#"*50)
        if not self.config.debug:
            logging.info(f"Question {idx_q}/{total_questions} completed in {processing_time:.2f}s")
            
    def log_question_out_of_index(self, idx_q: int, total_questions: int):
        print("#"*50, f"Question {idx_q}/{total_questions} out of index", "#"*50)
        if not self.config.debug:
            logging.info(f"Question {idx_q}/{total_questions} out of index")
    
    def log_experiment_completion(self, output_file: str):
        if not self.config.debug:
            print(f"Experiment completed. Results saved to {output_file}")
            logging.info(f"Experiment completed. Results saved to {output_file}")
    
    def log_uncertainty_save(self, uncertainty_file: str):
        if not self.config.debug:
            logging.info(f"Uncertainty scores saved to {uncertainty_file}")
    
    def log_error(self, error_msg: str, agent_contexts = None, idx_q=None):
        if not self.config.debug:
            if agent_contexts:
                for agent_context in agent_contexts:
                    logging.error(f"### Agent history tokens: {len(agent_context)}")
            logging.error(f"### Agent context: {agent_contexts}")
            if idx_q is not None:
                logging.error(f"Error processing question {idx_q}: {error_msg}")
            else:
                logging.error(f"Error: {error_msg}")

class ResultSaver:
    """Handles saving experiment results"""
    @staticmethod
    def save_results(response_dict, save_name: str, uncertainty_dict = None):
        output_file = f"{RESULTS_DIR}/{save_name}.json"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        import json
        json.dump(response_dict, open(output_file, "w"), indent=4)
        if uncertainty_dict is not None:
            uncertainty_file = f"{RESULTS_DIR}/{save_name}_uncertainty.pkl"
            with open(uncertainty_file, 'wb') as f:
                pickle.dump(uncertainty_dict, f)
            return output_file, uncertainty_file
        return output_file 