import torch
import torch.nn.functional as F
import numpy as np
import copy
from rich import print as rprint
from typing import List, Optional, Tuple

from prompt import get_prompt_provider, get_focus_instruction, PromptProvider
from prompt import get_prompt_provider, get_focus_instruction, PromptProvider
from utils import parse_answer_mmlupro, get_r0_contexts
from config import ExperimentConfig
from models import LMModel, BartSummarizer

# Uncertainty Scorer

def none_uncertainty_score():
    return{
            "nll_max": None,
            "nll_avg": None,
            "entropy_max": None,
            "entropy_avg": None,
            "nll_first": None,
            "nll_penultimate": None,
            "entropy_first": None,
            "entropy_penultimate": None,
        }

class UncertaintyScorer:
    def __init__(self, logits: torch.Tensor, output_ids: torch.Tensor):
        """
        logits: Tensor of shape (1, seq_len, vocab_size)
        output_ids: Tensor of shape (seq_len,) with generated token ids
        """
        self.logits = logits
        self.output_ids = output_ids
        
    def compute(self):
        try:
        
            self.logits = self.logits.squeeze(0)  # (seq_len, vocab_size)
            self.output_ids = self.output_ids     # (seq_len,)
            self.seq_len = self.logits.size(0)
            self.vocab_size = self.logits.size(1)
            
            # softmax & log_softmax
            self.probs = F.softmax(self.logits, dim=-1)
            self.log_probs = F.log_softmax(self.logits, dim=-1)
            
            # token-level scores
            self.token_nll = -self.log_probs[range(self.seq_len), self.output_ids]
            self.token_entropy = -torch.sum(self.probs * self.log_probs, dim=-1)
            
            
            return {
                "nll_max": self.token_nll.max().item(),
                "nll_avg": self.token_nll.mean().item(),
                "entropy_max": self.token_entropy.max().item(),
                "entropy_avg": self.token_entropy.mean().item(),
                "nll_first": self.token_nll[0].item(),
                "nll_penultimate": self.token_nll[-2].item() if self.seq_len >= 2 else None,
                "entropy_first": self.token_entropy[0].item(),
                "entropy_penultimate": self.token_entropy[-2].item() if self.seq_len >= 2 else None,
            }
        
        except Exception as e:
            print(f"### Error in uncertainty scorer: {e}")
            return none_uncertainty_score()

class QuestionProcessor:
    """Handles processing of individual questions"""
    
    def __init__(self, config: ExperimentConfig, llm_model: LMModel):
        self.config = config
        self.llm_model = llm_model
        self.prompt_provider = get_prompt_provider(config.dataset)
        self.prompt_provider.set_reasoning_method(config.reasoning_method)
        self.tokenizer = getattr(llm_model, 'tokenizer', None)
        if self.tokenizer is None and hasattr(llm_model, 'get_tokenizer'):
            self.tokenizer = llm_model.get_tokenizer()
        self.system_msg = self.prompt_provider.get_system_message()
        
        self._init_summarize_model()
        
        if self.config.strategy == 'softgate':
            import joblib
            # load clf and tau
            self.soft_tau = 0.544
            self.soft_clf = joblib.load('../data/soft_gate_4d_clf.pkl')
            if self.config.verbose:
                rprint(f"### soft_clf: Load from ../data/soft_gate_4d_clf.pkl")
                rprint(f"### soft_tau: {self.soft_tau}")
        
    def _init_summarize_model(self):
        if self.config.summarize_model == 'self':
            self.summarize_model = self.llm_model.summarize
        elif self.config.summarize_model == 'bart':
            self.summarize_model = BartSummarizer(device_map=self.config.device_map).summarize
        else:
            self.summarize_model = None
        
    def process(self, question: str, with_uncertainty: bool = False, qid: int = None) -> Tuple[List, Optional[List]]:
        if self.config.strategy in ['MAD', 'MADd']:
            return self.process_question_MAD(question,  with_uncertainty, qid)
        elif self.config.strategy == 'SID-v':
            return self.process_question_SIDv(question)
        elif self.config.strategy == 'SID-c':
            return self.process_question_SIDc(question)
        
        else:
            raise ValueError(f"Unknown strategy: {self.config.strategy}")
    
    def process_question_MAD(self, question: str, with_uncertainty: bool = False, qid: int = None) -> Tuple[List, Optional[List]]:
        use_MADd = self.config.strategy == 'MADd'
        
        agent_contexts = []
        if self.config.r0_path:
            agent_contexts = get_r0_contexts(self.config.r0_path, qid if qid is not None else question, n_agents=self.config.n_agents, r0_end_idx=3)
        else:
            agent_contexts = [self.prompt_provider.create_r0_message(question) for _ in range(self.config.n_agents)]
        agent_uncertainties = [[] for _ in range(self.config.n_agents)] if with_uncertainty else None
        
        iterator_range = range(self.config.n_rounds) if not self.config.r0_path else range(1, self.config.n_rounds)
        
        for round_num in iterator_range:
            if round_num == self.config.n_rounds - 1:
                # TODO: add a flag, last_IO to control this.
                self.prompt_provider.set_reasoning_method('IO') if use_MADd else self.prompt_provider.set_reasoning_method(self.config.reasoning_method)
                
            for agent_idx, agent_context in enumerate(agent_contexts):
                # Add strategy-based message for rounds > 0
                if round_num != 0:
                    # 1. Aggregate answers from all agents
                    other_agents = agent_contexts[:agent_idx] + agent_contexts[agent_idx+1:]
                    message = self.prompt_provider.construct_MAD_message(other_agents, question, 2 * round_num - 1, summarize_model=self.summarize_model)
                    agent_context.append(message)
                
                # Generate response
                if with_uncertainty:
                    generated_dict = self.llm_model.get_generated_dict(agent_context)
                    completion = generated_dict['content']
                    
                    # Compute uncertainty scores
                    uncertainty_scorer = UncertaintyScorer(generated_dict['logits'], generated_dict['ids'])
                    uncertainty_scores = uncertainty_scorer.compute()
                    if agent_uncertainties is not None:
                        agent_uncertainties[agent_idx].append(uncertainty_scores)
                        
                else:
                    completion = self.llm_model.llm(agent_context)
                
                assistant_message = self.prompt_provider.form_assistant_message(completion)
                agent_context.append(assistant_message)
                
            
            if round_num == self.config.n_rounds - 1:
                self.prompt_provider.set_reasoning_method(self.config.reasoning_method)
                
        if self.config.verbose:
            rprint(agent_contexts[0])
        return agent_contexts, agent_uncertainties
        
    def process_question_SIDv(self, question: str) -> Tuple[List, Optional[List]]:
        """Process a single question with lazy agent initialization and confidence check."""
        # Step 1: Lazy init - only first agent
        agent_contexts = [[] for _ in range(self.config.n_agents)]
        agent_contexts[0] = self.prompt_provider.create_r0_message(question)
        agent_uncertainties = [[] for _ in range(self.config.n_agents)] 

        # First agent answers
        if self.config.verbose:
            rprint(f"### First agent answers")
        generated_dict = self.llm_model.get_generated_dict(agent_contexts[0])
        completion = generated_dict['content']
        uncertainty_scorer = UncertaintyScorer(generated_dict['logits'], generated_dict['ids'])
        uncertainty_scores = uncertainty_scorer.compute()
        agent_uncertainties[0].append(uncertainty_scores)
        assistant_message = self.prompt_provider.form_assistant_message(completion)
        agent_contexts[0].append(assistant_message)

        # Step 2: Confidence check
        confident = False
        nll_max = uncertainty_scores['nll_max']
        entropy_max = uncertainty_scores['entropy_max']
        
        if self.config.verbose:
            rprint(f"### uncertainty_scores: {uncertainty_scores}")
            rprint(f"### nll_max: {nll_max}, entropy_max: {entropy_max}")

        if nll_max < self.llm_model.SIDv_nllmax and entropy_max < self.llm_model.SIDv_entmax:
            confident = True
        
        if confident and (not self.config.debug):
            if self.config.verbose:
                rprint(f"### confident, return")
            return agent_contexts, agent_uncertainties

        # Step 3: Not confident, initialize other agents and discuss
        if self.config.verbose:
            rprint(f"### not confident, initialize other agents and discuss")
        for idx in range(1, self.config.n_agents):
            agent_contexts[idx] = self.prompt_provider.create_r0_message(question)
            generated_dict = self.llm_model.get_generated_dict(agent_contexts[0])
            completion = generated_dict['content']
            uncertainty_scorer = UncertaintyScorer(generated_dict['logits'], generated_dict['ids'])
            uncertainty_scores = uncertainty_scorer.compute()
            agent_uncertainties[idx].append(uncertainty_scores)
            assistant_message = self.prompt_provider.form_assistant_message(completion)
            agent_contexts[idx].append(assistant_message)

        # Only do discussion for the remaining rounds (start from round 1)
        for round_num in range(1, self.config.n_rounds):
            for agent_idx, agent_context in enumerate(agent_contexts):
                # Add discussion message
                other_agents = agent_contexts[:agent_idx] + agent_contexts[agent_idx+1:]
                attn_message = self.prompt_provider.construct_attn_MAD_message(other_agents, question, 2 * round_num - 1)
                agent_context.append(attn_message)
                if self.config.verbose:
                    print(f"### Agent {agent_idx} round {round_num} \n Before compress: {attn_message['content']}")
                agent_context[:] = compress_discussion_with_attention(agent_context, self.llm_model, self.tokenizer, verbose=self.config.verbose, semantic_c=True, 
                                                   output_format_instruction=self.prompt_provider.output_format_instruction)
                if self.config.verbose:
                    print(f"### Agent {agent_idx} round {round_num} \n After compress: {agent_context[-1]['content']}")

                generated_dict = self.llm_model.get_generated_dict(agent_context)
                completion = generated_dict['content']
                uncertainty_scorer = UncertaintyScorer(generated_dict['logits'], generated_dict['ids'])
                uncertainty_scores = uncertainty_scorer.compute()
                if agent_uncertainties is not None:
                    agent_uncertainties[agent_idx].append(uncertainty_scores)

                assistant_message = self.prompt_provider.form_assistant_message(completion)
                agent_context.append(assistant_message)
                if self.config.verbose:
                    print(f"### Agent {agent_idx} round {round_num} generated_content: {completion}")

        return agent_contexts, agent_uncertainties
    
    def process_question_SIDc(self, question: str) -> Tuple[List, Optional[List]]:
        """Process a single question with lazy agent initialization and confidence check."""
        # Step 1: Lazy init - only first agent
        agent_contexts = [[] for _ in range(self.config.n_agents)]
        agent_contexts[0] = self.prompt_provider.create_r0_message(question)
        agent_uncertainties = [[] for _ in range(self.config.n_agents)] 

        # First agent answers
        if self.config.verbose:
            rprint(f"### First agent answers")
        generated_dict = self.llm_model.get_generated_dict(agent_contexts[0])
        completion = generated_dict['content']
        uncertainty_scorer = UncertaintyScorer(generated_dict['logits'], generated_dict['ids'])
        uncertainty_scores = uncertainty_scorer.compute()
        agent_uncertainties[0].append(uncertainty_scores)
        assistant_message = self.prompt_provider.form_assistant_message(completion)
        agent_contexts[0].append(assistant_message)

        # Step 2: Confidence check
        confident = False
        
        vector_4d = np.array([uncertainty_scores['nll_max'], uncertainty_scores['entropy_max'], uncertainty_scores['nll_avg'], uncertainty_scores['entropy_avg']])
        # reshape, 1, -1
        p = self.soft_clf.predict_proba(vector_4d.reshape(1, -1))[0, 1]
        if p > self.soft_tau:
            confident = True
        
        if self.config.verbose:
            rprint(f"### p: {p}, tau: {self.soft_tau}, debug mode:{self.config.debug}. If debug mode is on, will not return")
        
        if confident and (not self.config.debug):
            if self.config.verbose:
                rprint(f"### confident, return")
            return agent_contexts, agent_uncertainties

        # Step 3: Not confident, initialize other agents and discuss
        if self.config.verbose:
            rprint(f"### not confident, initialize other agents and discuss")
        for idx in range(1, self.config.n_agents):
            agent_contexts[idx] = self.prompt_provider.create_r0_message(question)
            generated_dict = self.llm_model.get_generated_dict(agent_contexts[0])
            completion = generated_dict['content']
            uncertainty_scorer = UncertaintyScorer(generated_dict['logits'], generated_dict['ids'])
            uncertainty_scores = uncertainty_scorer.compute()
            agent_uncertainties[idx].append(uncertainty_scores)
            assistant_message = self.prompt_provider.form_assistant_message(completion)
            agent_contexts[idx].append(assistant_message)

        # Only do discussion for the remaining rounds (start from round 1)
        for round_num in range(1, self.config.n_rounds):
            for agent_idx, agent_context in enumerate(agent_contexts):
                # Add discussion message
                other_agents = agent_contexts[:agent_idx] + agent_contexts[agent_idx+1:]
                attn_message = self.prompt_provider.construct_attn_MAD_message(other_agents, question, 2 * round_num - 1)
                agent_context.append(attn_message)
                agent_context[:] = compress_discussion_with_attention(agent_context, self.llm_model, self.tokenizer, verbose=self.config.verbose, semantic_c=True, 
                                                   output_format_instruction=self.prompt_provider.output_format_instruction)
                if self.config.verbose:
                    print(f"### Agent {agent_idx} round {round_num} \n After compress: {agent_context[-1]['content']}")

                generated_dict = self.llm_model.get_generated_dict(agent_context)
                completion = generated_dict['content']
                uncertainty_scorer = UncertaintyScorer(generated_dict['logits'], generated_dict['ids'])
                uncertainty_scores = uncertainty_scorer.compute()
                if agent_uncertainties is not None:
                    agent_uncertainties[agent_idx].append(uncertainty_scores)

                assistant_message = self.prompt_provider.form_assistant_message(completion)
                agent_context.append(assistant_message)
                if self.config.verbose:
                    print(f"### Agent {agent_idx} round {round_num} generated_content: {completion}")

        return agent_contexts, agent_uncertainties
    
    
class VQuestionProcessor:
    def __init__(self, config: ExperimentConfig, llm_model: LMModel):
        self.config = config
        self.llm_model = llm_model
        self.prompt_provider = get_prompt_provider(config.dataset)
        self.prompt_provider.set_reasoning_method(config.reasoning_method)
        
        
    def process(self, image, question, qid):
        if self.config.strategy in ['MAD', 'MADd']:
            return self.process_MAD(image, question, qid)
        elif self.config.strategy in ['SID-c', 'SID-v']:
            return self.process_SID(image, question, qid)
        else:
            raise ValueError(f"Unknown strategy: {self.config.strategy}")

        
    def _init_agent_context_with_r0(self, question, image, qid):
        if self.config.r0_path:
            agent_contexts = get_r0_contexts(self.config.r0_path, qid, n_agents=self.config.n_agents, image=image)
        else:
            agent_contexts = [self.prompt_provider.create_r0_message(image, question) for _ in range(self.config.n_agents)]
        return agent_contexts
        
        
    def process_MAD(self, image, question, qid):
        # self.config.r0_path
        record_uncertainty = self.config.record_uncertainty
        agent_contexts = self._init_agent_context_with_r0(question, image, qid)
        agent_uncertainties = [[] for _ in range(self.config.n_agents)] if record_uncertainty else None
        
        iterator_range = range(self.config.n_rounds) if not self.config.r0_path else range(1, self.config.n_rounds)
        

        original_reasoning = self.prompt_provider.reasoning_method
        for round_num in iterator_range:
            for agent_idx, agent_context in enumerate(agent_contexts):
                
                 # optionally switch to IO on the last round
                if round_num == self.config.n_rounds - 1 and self.config.strategy == 'MADd':
                    self.prompt_provider.set_reasoning_method('IO')
                
                if round_num != 0:
                    other_agents = agent_contexts[:agent_idx] + agent_contexts[agent_idx+1:]
                    user_message = self.prompt_provider.create_MAD_message(other_agents, 2 * round_num - 1)
                    agent_context.append(user_message)
                    
                
                if record_uncertainty:
                    generated_dict = self.llm_model.llm_return_generated_dict(agent_context)
                    resp_content = generated_dict['content']
                    if agent_uncertainties is not None:
                        u_scorer = UncertaintyScorer(generated_dict['logits'], generated_dict['ids'])
                        agent_uncertainties[agent_idx].append(u_scorer.compute())
                else:
                    resp_content = self.llm_model.llm(agent_context)
                
                # restore reasoning method if switched
                if round_num == self.config.n_rounds - 1:
                    self.prompt_provider.set_reasoning_method(original_reasoning)
                
                assistant_message = self.prompt_provider.create_role_text(text=resp_content, role='assistant')
                agent_context.append(assistant_message)
          
        # mask in case not JSON serializable      
        for agent_context in agent_contexts:
            agent_context[:] = self.mask_image_in_agent_context(agent_context)
        
        if self.config.verbose:
            rprint(agent_contexts[0])
             
        return agent_contexts, agent_uncertainties

    def process_SID(self, image, question, qid):
        record_uncertainty = self.config.record_uncertainty
        agent_contexts = self._init_agent_context_with_r0(question, image, qid)
        agent_uncertainties = [[] for _ in range(self.config.n_agents)] if record_uncertainty else None
        
        original_reasoning = self.prompt_provider.reasoning_method
        iterator_range = range(self.config.n_rounds) if not self.config.r0_path else range(1, self.config.n_rounds)
        
        for round_num in iterator_range:
            for agent_idx, agent_context in enumerate(agent_contexts):
                if round_num == self.config.n_rounds - 1 and self.config.strategy == 'attnd':
                    self.prompt_provider.set_reasoning_method('IO')
                
                if round_num != 0:
                    other_agents = agent_contexts[:agent_idx] + agent_contexts[agent_idx+1:]
                    agent_context[:] = self.prompt_provider.create_attn_message(agent_context, other_agents, 2 * round_num - 1)
                    agent_context[:] = compress_v_discussion_with_attention(
                        agent_context,
                        self.llm_model,
                        top_p=self.config.attn_topp,
                        semantic_c=self.config.semantic_c,
                        output_format_instruction=self.prompt_provider.output_format_instruction,
                        verbose=self.config.verbose,
                        debug=self.config.debug
                    )
                    if self.config.verbose:
                        print(f"### Agent {agent_idx} round {round_num} \n After compress: {agent_context[-1]['content']}")
                

                
                if record_uncertainty:
                    generated_dict = self.llm_model.llm_return_generated_dict(agent_context)
                    completion = generated_dict['content']
                    if agent_uncertainties is not None:
                        try:
                            u_scorer = UncertaintyScorer(generated_dict['logits'], generated_dict['ids'])
                            agent_uncertainties[agent_idx].append(u_scorer.compute())
                        except Exception as e:
                            agent_uncertainties[agent_idx].append(none_uncertainty_score())
                            print(f"### Error in uncertainty scorer: {e}")
                else:
                    completion = self.llm_model.llm(agent_context)
                

                assistant_message = self.prompt_provider.create_role_text(text=completion, role='assistant')
                agent_context.append(assistant_message)
                
                # restore reasoning method if switched
                if round_num == self.config.n_rounds - 1:
                    self.prompt_provider.set_reasoning_method(original_reasoning)
                
                if self.config.verbose:
                    print(f"### Agent {agent_idx} round {round_num} generated_content: {completion}")
        
        # mask images before returning
        for agent_context in agent_contexts:
            agent_context[:] = self.mask_image_in_agent_context(agent_context)
        
        return agent_contexts, agent_uncertainties
    
    def mask_image_in_agent_context(self, agent_context):
        for message in agent_context:
            if message['role'] == 'user':
                for content in message['content']:
                    if content['type'] == 'image':
                        content['image'] = 'image'
        return agent_context
        
    
def extract_token_spans_by_string(full_text, substrings, tokenizer):
    offset_mapping = tokenizer(full_text, return_offsets_mapping=True)["offset_mapping"]
    spans = []
    for substring in substrings:
        char_start = full_text.find(substring)
        if char_start == -1:
            spans.append(None)
            continue
        char_end = char_start + len(substring)
        tok_start = next(i for i, (s, e) in enumerate(offset_mapping) if s <= char_start < e or (s >= char_start and s < char_end))
        tok_end = max(i for i, (s, e) in enumerate(offset_mapping) if s < char_end) + 1
        spans.append((tok_start, tok_end))
    return spans

def compress_discussion_with_attention(messages, model, tokenizer, top_p=0.4, semantic_c=True, output_format_instruction='', verbose=False, debug=False) -> List[dict]:
    assert output_format_instruction != '', "output_format_instruction cannot be empty"
    
    new_msg = messages.copy()
    
    if debug:
        rprint(f">>>>> DEBUG MODE: top_p: {top_p}, semantic_c: {semantic_c}, output_format_instruction: {output_format_instruction}")
        return new_msg
    
    if top_p == 0:
        new_msg[-1]['content'] = "Please based on the history messages, answer the question." + output_format_instruction
        return new_msg
    
    # 1. concatnate prompt
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # 2. locate focus_prompt, other_agents, end_prompt
    focus_str = get_focus_instruction()
    other_agents_start = prompt.find("### OTHER AGENTS ANSWERS ###") + len("### OTHER AGENTS ANSWERS ###") + 1
    other_agents_end = prompt.find("### END OTHER AGENTS ANSWERS ###")
    if other_agents_start == -1 or other_agents_end == -1:
        raise ValueError("Cannot find agent discussion markers in prompt")
    focus_span = (prompt.find(focus_str), prompt.find(focus_str) + len(focus_str))
    other_agents_span = (other_agents_start, other_agents_end)
    
    # 3. extract token span
    offset_mapping = tokenizer(prompt, return_offsets_mapping=True)["offset_mapping"]
    focus_tok_start = next(i for i, (s, e) in enumerate(offset_mapping) if s <= focus_span[0] < e or (s >= focus_span[0] and s < focus_span[1]))
    focus_tok_end = max(i for i, (s, e) in enumerate(offset_mapping) if s < focus_span[1]) + 1
    other_tok_start = next(i for i, (s, e) in enumerate(offset_mapping) if s <= other_agents_span[0] < e or (s >= other_agents_span[0] and s < other_agents_span[1]))
    other_tok_end = max(i for i, (s, e) in enumerate(offset_mapping) if s < other_agents_span[1]) + 1
    
    
    # 4. run a forward to get attention scores
    model_inputs, _, attn = model.get_prompt_attn_save_memory(messages) # batch, num_heads, seq, seq
    focus2other = attn[0, :, other_tok_start:other_tok_end, focus_tok_start:focus_tok_end ].amax(dim=0).amax(dim=1)
    
    if verbose:
        print(f"### messages: {messages}")
        print(f"### focus_tok_start: {focus_tok_start}, focus_tok_end: {focus_tok_end}")
        print(f"### other_tok_start: {other_tok_start}, other_tok_end: {other_tok_end}")
        print(f"### focus2other shape: {focus2other.shape}")
    
    # # attn: (batch, n_head, seq, seq)
    
    # 5. use top-p to select tokens
    n_select = max(1, int((other_tok_end - other_tok_start) * top_p))
    top_idx = focus2other.topk(n_select).indices.tolist()
    selected_token_idx = sorted(set([other_tok_start + i for i in top_idx]))
    
    # 6. extract the original text corresponding to the selected tokens
    if semantic_c:
        selected_text = extract_semantic_spans(selected_token_idx, offset_mapping, prompt)
    else:
        selected_text = ''
        for idx in selected_token_idx:
            s, e = offset_mapping[idx]
            selected_text += prompt[s:e]
    
    # 7. use modify_attn_MAD_content to replace focus_prompt and concatenate compressed discussion with end_prompt
    new_content = modify_attn_MAD_content(attn_discussion=selected_text, output_format_instruction=output_format_instruction)
    
    # 8. return new messages
    new_msg[-1]['content'] = new_content
    
    return new_msg


def compress_v_discussion_with_attention(messages, model:LMModel, top_p=0.4, semantic_c=True, focus_strs=['### FOCUS INSTRUCTION ###', '### END FOCUS INSTRUCTION ###'], other_strs=['### OTHER AGENTS ANSWERS ###', '### END OTHER AGENTS ANSWERS ###'], output_format_instruction='', verbose=False, debug=False) -> List[dict]:
    """
    Multimodal version: use model.forward_return_attention to compress discussion tokens guided by attention.
    - processor_like: object providing apply_chat_template and a .tokenizer for offset mapping (e.g., AutoProcessor)
    """
    assert output_format_instruction != '', "output_format_instruction cannot be empty"
    new_msg = messages.copy()

    if debug or model.model_name == 'dryrun':
        rprint(f">>>>> DEBUG MODE + DRYRUN: top_p: {top_p}, semantic_c: {semantic_c}, output_format_instruction: {output_format_instruction}")
        return new_msg

    if top_p == 0:
        new_msg[-1]['content'] = "Please based on the history messages, answer the question." + output_format_instruction
        return new_msg

    ############################ Text Part #############################
    prompt_chat_template, images = model.get_message_chat_template(new_msg)
    
    # Find the text offset mapping
    offset_mapping_text = model.processor.tokenizer(prompt_chat_template, return_offsets_mapping=True)["offset_mapping"]
    
    focus_anchor_span = extract_str_span(prompt_chat_template, focus_strs[0], focus_strs[1], inclusive=True)
    focus_span = extract_str_span(prompt_chat_template, focus_strs[0], focus_strs[1], inclusive=False)
    other_agents_span = extract_str_span(prompt_chat_template, other_strs[0], other_strs[1], inclusive=False)
    

    focus_anchor_tok = next(i for i, (s, e) in enumerate(offset_mapping_text) if s <= focus_anchor_span[0] < e or (s >= focus_anchor_span[0] and s < focus_anchor_span[1]))
    
    focus_tok_start = next(i for i, (s, e) in enumerate(offset_mapping_text) if s <= focus_span[0] < e or (s >= focus_span[0] and s < focus_span[1]))
    focus_tok_end = max(i for i, (s, e) in enumerate(offset_mapping_text) if s < focus_span[1]) 
    other_tok_start = next(i for i, (s, e) in enumerate(offset_mapping_text) if s <= other_agents_span[0] < e or (s >= other_agents_span[0] and s < other_agents_span[1]))
    other_tok_end = max(i for i, (s, e) in enumerate(offset_mapping_text) if s < other_agents_span[1]) 

    str_begin_tokens = model.processor.tokenizer.encode(focus_strs[0], add_special_tokens=False)

    ############################ Image-Text Shift Part #############################
    offset_mapping = model.processor(text=prompt_chat_template, images=images, return_offsets_mapping=True)["offset_mapping"]
    model_inputs, _, attn = model.forward_return_attention(new_msg)
    
    # focus anchor
    focus_index_in_mllm = find_sublist(model_inputs['input_ids'][0], str_begin_tokens)
    token_shift = focus_index_in_mllm - focus_anchor_tok
    
    focus_shift_start = focus_tok_start + token_shift
    focus_shift_end = focus_tok_end + token_shift
    other_shift_start = other_tok_start + token_shift
    other_shift_end = other_tok_end + token_shift
    
    ############################ Attention Part #############################
    # attn: (B, H, S, S) or (B, S, S). Normalize to (H, S, S)
    if attn.dim() == 3:
        attn_heads = attn[0].unsqueeze(0)  # (1, S, S)
    else:
        attn_heads = attn[0]  # (H, S, S)
    focus2other = attn_heads[:, other_shift_start:other_shift_end, focus_shift_start:focus_shift_end].amax(dim=0).amax(dim=1)

    ############################ Top-p Select Part #############################
    n_select = max(1, int((other_shift_end - other_shift_start) * top_p))
    top_idx = focus2other.topk(n_select).indices.tolist()
    
    # remember to remove shift here.
    selected_token_idx = sorted(set([other_tok_start + i for i in top_idx])) # can change to other_tok_start

    # 6) Map back to text
    if semantic_c:
        selected_text = extract_semantic_spans(selected_token_idx, offset_mapping_text, prompt_chat_template)
    else:
        selected_text = ''
        for idx in selected_token_idx:
            s, e = offset_mapping_text[idx]
            selected_text += prompt_chat_template[s:e]
            
    selected_text = selected_text.replace(other_strs[0],'').replace(other_strs[1],'')

    ############################ Replace as Attention Content #############################
    content_pad = 'Below are the key points where other agents **disagree** with your own reasoning.*\nConcentrate on those disagreements and Keep or revise your answer accordingly. \n'
    content_pad  = content_pad  + selected_text 
    content_pad = content_pad + output_format_instruction
    new_msg[-1]['content'][0]['text'] = content_pad
    return new_msg

def extract_semantic_spans(selected_token_idx, offset_mapping, prompt_str, clause_separators=None):
    """
        Extract semantic spans from selected tokens.
    """
    if clause_separators is None:
        clause_separators = ['.', '!', '?', ';', '。', '！', '？', '；', '\n']
    
    spans = []
    for idx in selected_token_idx:
        s, e = offset_mapping[idx]

        start = s
        while start > 0 and prompt_str[start-1] not in clause_separators:
            start -= 1

        end = e
        while end < len(prompt_str) and prompt_str[end] not in clause_separators:
            end += 1

        if end < len(prompt_str):
            end += 1
        spans.append((start, end))
    

    merged = []
    for span in sorted(spans):
        if not merged or span[0] > merged[-1][1]:
            merged.append(list(span))
        else:
            merged[-1][1] = max(merged[-1][1], span[1])
    

    return ''.join([prompt_str[s:e] for s, e in merged]).strip()


def modify_attn_MAD_content(attn_discussion='', output_format_instruction=''):
    
    content_pad = 'Below are the key points where other agents **disagree** with your own reasoning.*\nConcentrate on those disagreements and Keep or revise your answer accordingly. \n'
    content_pad  = content_pad  + attn_discussion 
    
    content_pad = content_pad + output_format_instruction
    
    return content_pad

def find_sublist(lst, sub):
    """
    Search for a sublist (sub) in a list (lst).
    Returns the start index if found, else -1.
    """
    n, m = len(lst), len(sub)
    for i in range(n - m + 1):
        if lst[i:i+m] == sub:
            return i
    return -1

def extract_str_span(text, str_start, str_end, inclusive=True, verbose=False):
    start = text.find(str_start)
    end = text.find(str_end) + len(str_end)
    
    if not inclusive:
        start += len(str_start)
        end -= len(str_end)
    
    if verbose:
        print(f"Start: {start}, End: {end}, inclusive: {inclusive}")
        print(text[start:end])
    return (start, end)
    
    