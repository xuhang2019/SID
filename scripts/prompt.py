import re
from typing import Callable, Optional

    

def construct_MAD_message(agents, question, idx):
    if len(agents) == 0:
        return {"role": "user", "content": "Can you double check that your answer is correct. Give your final answer in the format of '(X)'."}
    else:
        prefix_string = "These are the solutions to the problem from other agents: "

        for agent in agents:
            agent_response = agent[idx+1]["content"] # idx+1 because the first one is the system prompt.
            response = "\n\n One agent solution: ```{}```".format(agent_response)

            prefix_string = prefix_string + response

        prefix_string = prefix_string + """\n\n Using the reasoning from other agents as additional advice, can you give an updated answer? Examine your solution and that other agents step by step. Give your final answer in the format of '(X)'.""".format(question)
        return {"role": "user", "content": prefix_string}
    
def get_focus_instruction():
    return """### FOCUS INSTRUCTION ###
You are about to read alternative solutions from other agents.
*Identify the key points where they **disagree** with your own reasoning.*
Concentrate on those disagreements and decide which line of reasoning is better.
Keep or revise your answer accordingly.  
### END FOCUS INSTRUCTION ###"""
    
def construct_attn_MAD_message(agents, question, idx):
    if len(agents) == 0:
        # no other agents, degrade to self-refinement
        return {"role": "user", "content": "Can you double check that your answer is correct. Give your final answer in the format of '(X)'."}
    else:
        content_pad = get_focus_instruction() + '\n' + "### OTHER AGENTS ANSWERS ###"

        for agent in agents:
            agent_response = agent[idx+1]["content"] # idx+1 because the first one is the system prompt.
            response = "\n\n One agent solution: ```{}```".format(agent_response)

            content_pad = content_pad + response
            
        content_pad = content_pad + "\n ### END OTHER AGENTS ANSWERS ###  Give your final answer in the format of '(X)'"

        return {"role": "user", "content": content_pad}

def construct_MAD_role_play_system_prompts(role='affirmative'):
    """You are a debater. Hello and welcome to the debate competition. It’s not necessary to fully agree with each other’s perspectives, as our objective is to find the correct answer. The debate topic is: {question}"""
    
    if role == 'affirmative':
        return {"role": "system", "content": "You are the affirmative side. Please express your viewpoints on a given topic."}
    elif role == 'negative':
        return {"role":"system", "content": "You are the negative side. You disagree with the affirmative’s points. Provide your reasons and your own answer."}
    elif role == 'judger':
        return {"role":"system", "content": "You are a moderator. There will be two debaters involved in a debate competition. They will present and discuss their answers. At the end of each round, evaluate both sides’ answers and decide, if there is a winne, please reply (END) else return (CONTINUE) to continue the debate."}
    else:
        raise ValueError(f"Invalid role: {role}")

def construct_self_reflection_prompts(role='Reflect'):
    if role == 'Reflect':
        return {"role": "user", "content": """Based on your answer, please reflect on yourself:
1. What steps or assumptions may be problematic?
2. Why do these problems lead to inaccurate answers?
3. How can you improve?"""}
    elif role == 'Amend':
        return {"role": "user", "content": "Based on the feedback from the above self-reflection, answer the question again and give the updated complete thoughts and final answer. Give your final answer in the format of '(X)'"}
    elif role == 'Round4':
        return {"role": "user", "content": "I didn't see the result in your last step, please think again. Give your final answer in the format of '(X)'"}
    else:
        raise ValueError(f"Invalid role: {role}")
    




    
# ================== PromptProvider System ==================

class PromptProvider:
    reasoning_prompt_dict = {
        # remember to end with '. ' (dot + space)
        'CoT': "Let's think step by step. ",
        'IO': "Please directly give your answer. ",
        'SRP': "Let's solve this problem step by step using structured reasoning. ",
        'PoT': "Let's solve this problem using Python code. ",
        'CCoT': "Let's analyze the image step by step and create a scene graph. ",
        'DDCoT': "Let's deconstruct this problem into sub-questions and solve them systematically. ",
    }
    
    def __init__(self, reasoning_method ="CoT"):
        self.set_reasoning_method(reasoning_method)
        
    def set_reasoning_method(self, reasoning_method):
        """
            If the reasoning_method is not in the reasoning_prompt_dict, use it as is.
            Otherwise, use the reasoning_prompt_dict[reasoning_method]
            
            Default is CoT.
        """
        self.reasoning_method = reasoning_method
        
        if reasoning_method not in self.reasoning_prompt_dict:
            self.reasoning_prompt = reasoning_method
        else:   
            self.reasoning_prompt = self.reasoning_prompt_dict[reasoning_method]
    
    def get_system_message(self):
        raise NotImplementedError
    
    def construct_MAD_message(self, agents, question, idx, summarize_model=None):
        if len(agents) == 0:
            return {"role": "user", "content": "Can you double check that your answer is correct. " + self.output_format_instruction}
        else:
            resp_content = ""
            
            for agent in agents:
                agent_response = agent[idx+1]["content"]
                response = "\n\n One agent solution: ```{}```".format(agent_response)                
                resp_content = resp_content + response
            
            if summarize_model:
                resp_content = summarize_model(resp_content)
                prefix_string = "These are the summaries of the solutions to the problem from other agents: "
            else:
                prefix_string = "These are the solutions to the problem from other agents: "
            
            content = prefix_string + resp_content + "\n\n Using the reasoning from other agents as additional advice, can you give an updated answer? " + self.output_format_instruction
            
            return self.create_role_content(role='user', content=content)
        
    def construct_attn_MAD_message(self, agents, question, idx):
        if len(agents) == 0:
            return {"role": "user", "content": "Can you double check that your answer is correct. Give your final answer in the form of \\boxed{answer} in the last sentence of your response, e.g., \\boxed{[1, 3]}."}
        else:
            content_pad = get_focus_instruction() + '\n' + "### OTHER AGENTS ANSWERS ###"
            for agent in agents:
                agent_response = agent[idx+1]["content"]
                response = "\n\n One agent solution: ```{}```".format(agent_response)
                content_pad = content_pad + response
            content_pad = content_pad + "\n ### END OTHER AGENTS ANSWERS ### " + "Using the reasoning from other agents as additional advice, can you give an updated answer? " + self.output_format_instruction
            return {"role": "user", "content": content_pad}
    
    def construct_self_reflection_prompts(self, role):
        if role == 'Reflect':
            return {"role": "user", "content": """Based on your answer, please reflect on yourself:\n1. What steps or assumptions may be problematic?\n2. Why do these problems lead to inaccurate answers?\n3. How can you improve?"""}
        elif role == 'Amend':
            return {"role": "user", "content": "Based on the feedback from the above self-reflection, answer the question again and give the updated complete thoughts and final answer. " + self.output_format_instruction}
        elif role == 'Round4':
            return {"role": "user", "content": "I didn't see the result in your last step, please think again. " + self.output_format_instruction}
        else:
            raise ValueError(f"Invalid role: {role}")
    
    def construct_self_contrast_prompts(self, role):
        """Self-Contrast method prompts"""
        if role == 'Generate':
            return {"role": "user", "content": "Please generate an alternative solution to this problem using a different approach or reasoning method. " + self.output_format_instruction}
        elif role == 'Compare':
            return {"role": "user", "content": """Now compare your original solution with the alternative solution:
1. What are the key differences between the two approaches?
2. Which approach seems more reliable and why?
3. Can you identify any weaknesses in either approach?
4. Based on this comparison, what is your final answer?""" + self.output_format_instruction}
        elif role == 'Final':
            return {"role": "user", "content": "Based on your comparison of the different approaches, provide your final answer. " + self.output_format_instruction}
        else:
            raise ValueError(f"Invalid role: {role}")
    
    def construct_self_refine_prompts(self, role):
        """Self-Refine method prompts"""
        if role == 'Initial':
            return {"role": "user", "content": "Please solve this problem step by step. " + self.output_format_instruction}
        elif role == 'Critique':
            return {"role": "user", "content": """Now critique your own solution:
1. Are there any logical errors in your reasoning?
2. Did you miss any important details or considerations?
3. Is your solution complete and well-justified?
4. What could be improved in your approach?"""}
        elif role == 'Refine':
            return {"role": "user", "content": "Based on your self-critique, please refine your solution and provide an improved answer. " + self.output_format_instruction}
        else:
            raise ValueError(f"Invalid role: {role}")
    
    def construct_dmad_message(self, agents, question, idx, reasoning_method='IO'):
        """DMAD method message construction"""
        if len(agents) == 0:
            return {"role": "user", "content": "Can you double check that your answer is correct. " + self.output_format_instruction}
        else:
            prefix_string = "These are other answers to the question using different reasoning methods: "
            
            for agent in agents:
                agent_response = agent[idx+1]["content"]
                response = "\n\n One answer: ```{}```".format(agent_response)
                prefix_string = prefix_string + response
            
            if reasoning_method == 'IO':
                prefix_string = prefix_string + f"\n\n Using the answers of different methods as additional information, can you provide your answer to the question? \n {question}"
            elif reasoning_method == 'CCoT':
                prefix_string += f"""\n\n Using the answers of different methods as additional information, generate a scene graph in JSON format for the provided image and its associated question.
{question}

The scene graph should include:
1. Objects that are relevant to answering the question.
2. Object attributes that are relevant to answering the question.
3. Object relationships that are relevant to answering the question.

Just generate the scene graph in JSON format. Do not say extra words."""
            elif reasoning_method == 'DDCoT':
                prefix_string += f'''Using the answers of different methods as additional information, please think step-by-step about the preliminary knowledge to answer the question, deconstruct the problem as completely as possible down to necessary sub-questions. Then with the aim of helping humans answer the original question, try to answer the sub-questions. 
{question}
        
The expected answering form is as follows:
Sub-questions:
1. <sub-question 1>
2. <sub-question 2>
...

Sub-answers:
1. <sub-answer 1>
2. <sub-answer 2>
...'''
            else:
                raise ValueError(f"{reasoning_method} is not in ['IO', 'CCoT', 'DDCoT'].")
            
            return {"role": "user", "content": prefix_string}
    
    def construct_srp_message(self, agents, question, idx):
        """Structured Reasoning Prompt (SRP) message construction"""
        if len(agents) == 0:
            return {"role": "user", "content": "Can you double check that your answer is correct. " + self.output_format_instruction}
        else:
            prefix_string = "These are structured reasoning solutions from other agents: "
            
            for agent in agents:
                agent_response = agent[idx+1]["content"]
                response = "\n\n One structured solution: ```{}```".format(agent_response)
                prefix_string = prefix_string + response
            
            prefix_string = prefix_string + f"\n\n Using the structured reasoning from other agents as additional information, can you provide your structured solution to the question? \n {question}"
            return {"role": "user", "content": prefix_string}
    
    def construct_pot_message(self, agents, question, idx):
        """Program of Thought (PoT) message construction"""
        if len(agents) == 0:
            return {"role": "user", "content": "Can you double check that your answer is correct. " + self.output_format_instruction}
        else:
            prefix_string = "These are program-based solutions from other agents: "
            
            for agent in agents:
                agent_response = agent[idx+1]["content"]
                response = "\n\n One program solution: ```{}```".format(agent_response)
                prefix_string = prefix_string + response
            
            prefix_string = prefix_string + f"\n\n Using the program-based reasoning from other agents as additional information, can you provide your program solution to the question? \n {question}"
            return {"role": "user", "content": prefix_string}
    
    def construct_ccot_answer_prompt(self, scene_graph, question):
        """Construct prompt for answering question using scene graph"""
        prompt = f"""The scene graph of the image in JSON format:
{scene_graph}

Use the image and scene graph as context and answer the following question.
{question}

{self.output_format_instruction}"""
        return self.create_role_content(role='user', content=prompt)
    
    def construct_ddcot_answer_prompt(self, subquestion_answers, question):
        """Construct prompt for answering question using sub-questions and sub-answers"""
        prompt = f"""The problem can be deconstructed down to sub-questions.
{subquestion_answers}

According to the sub-questions and sub-answers, give your answer to the problem. {self.output_format_instruction}"""
        return self.create_role_content(role='user', content=prompt)
    
    def form_ccot_assistant_message(self, scene_graph, final_answer):
        """Format CCoT assistant message with scene graph and final answer"""
        content = f"""First, get the scene graph of the image in JSON format:
{scene_graph}

Then, use the image and scene graph as context to answer the question.
{final_answer}"""
        return self.create_role_content(role='assistant', content=content)
    
    def form_ddcot_assistant_message(self, subquestion_answers, final_answer):
        """Format DDCoT assistant message with sub-questions and final answer"""
        content = f"""First, the problem can be deconstructed down to sub-questions. 
{subquestion_answers}

Then, according to the sub-questions and sub-answers to answer the question.
{final_answer}"""
        return self.create_role_content(role='assistant', content=content)
    
    def form_assistant_message(self, completion):
        return self.create_role_content(role='assistant', content=completion)
    
    def create_role_content(self, role='user', content=''):
        """
            Create a role-content pair for the prompt
            
            Args:
                role: str, the role of the message
                content: str, the content of the message
            
            Returns:
                dict, the role-content pair
        """
        return {"role": role, "content": content}
    
    def create_r0_message(self, question,):
        msg = []
        msg.append(self.get_system_message())
        content = question + self.output_format_instruction
        msg.append(self.create_role_content(role='user', content=content))
        return msg
        
    
    @property
    def output_format_instruction(self):
        """
            use `self.reasoning_prompt` to construct the output format instruction
            Example:
                self.reasoning_prompt = "Let's think step by step."
                
                return "Let's think step by step. Give your final answer in the format of '(X)'"
        """
        raise NotImplementedError

class MMLUPromptProvider(PromptProvider):
    def get_system_message(self):
        content = "You are a trivia expert who knows everything, you are tasked to answer the following question. Give your final answer in the format of (X), e.g., (A)."
        return self.create_role_content(role='system', content=content)
    
    @property
    def output_format_instruction(self):
        return self.reasoning_prompt + "Give your final answer in the format of '(X)'"

class MathPromptProvider(PromptProvider):
    def get_system_message(self):
        content = "You are a math expert, you are tasked to determine the answer to the following question. Give your final answer in the form of \\boxed{answer} in the last sentence of your response, e.g., \\boxed{[1, 3]}."
        return self.create_role_content(role='system', content=content)
    
    @property
    def output_format_instruction(self):
        return self.reasoning_prompt + "Give your final answer in the form of \\boxed{answer} at the end of your response, e.g., \\boxed{[1, 3]}."

class GPQAPromptProvider(PromptProvider):
    def get_system_message(self):
        content = "You are an expert in graduate-level science and mathematics. You will be presented with challenging questions designed to test your reasoning abilities. You are tasked to answer the following question. Your last sentence should be 'The correct answer is (insert answer here).' e.g., The correct answer is (A)."
        return self.create_role_content(role='system', content=content)
    
    @property
    def output_format_instruction(self):
        return  self.reasoning_prompt + "Your last sentence should be 'The correct answer is (insert answer here).' e.g., The correct answer is (A)."

def get_prompt_provider(dataset):
    if dataset in ['mmlu', 'mmlupro']:
        return MMLUPromptProvider()
    elif dataset == 'math':
        return MathPromptProvider()
    elif dataset == 'gpqa':
        return GPQAPromptProvider()
    elif dataset == 'scienceqa':
        return VPromptProvider()
    elif dataset == 'mmstar':
        return MMStarPromptProvider()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    


class VPromptProvider(PromptProvider):
    def get_system_message(self):
        content = "You are a trivia expert who knows everything, you are tasked to answer the following question. Give your final answer in the format of (X), e.g., (A)."
        return self.create_role_text(text=content, role='system')
    
    @property
    def output_format_instruction(self):
        if self.reasoning_method == 'CoT':
            return "Please think step by step and give your final answer in the format of '(X)'. You should only give one answer. For example, The answer is (A). Because ... "
        else:
            return "Give your final answer in the format of '(X)'. You should only give one answer. Please directly give the answer. For example, The answer is (A)."
    
    def create_type_content(self, content_type, content, content_key='text'):
        """
            Remeber to add a [] around the content_key.
        """
        return {"type": content_type, content_key: content}
    
    def create_image_text_prompt(self, image, text):
        return [
            self.create_type_content("image", image, content_key='image'), 
            self.create_type_content("text", text + self.output_format_instruction),
        ]
    
    def create_r0_message(self, image, text, use_system_prompt=True):
        msg = []
        if use_system_prompt:
            msg.append(self.get_system_message())
        content = self.create_image_text_prompt(image, text)
        msg.append(self.create_role_content(role='user', content=content))
        return msg
    
    def create_role_text(self, text, role='assistant'):
        """
            create: {
                "role": <role>
                "content": [
                    {
                        "type": "text",
                        "text": <text>
                    }
                ]
            }
        """
        content = self.create_type_content("text", text)
        return self.create_role_content(role=role, content=[content])
    
    def create_MAD_message(self, agents, idx, summarize_model: Optional[Callable] = None):
        if len(agents) == 0:
            content = "Can you double check that your answer is correct. " + self.output_format_instruction
            
            return self.create_role_text(text=content)
        else:
            resp_content = ""
            
            for agent in agents:
                agent_response = agent[idx+1]["content"][0]["text"]
                response = "\n\n One agent solution: ```{}```".format(agent_response)                
                resp_content = resp_content + response
            
            if summarize_model:
                resp_content = summarize_model(resp_content)
                prefix_string = "These are the summaries of the solutions to the problem from other agents: "
            else:
                prefix_string = "These are the solutions to the problem from other agents: "
            
            content = prefix_string + resp_content + "\n\n Using the reasoning from other agents as additional advice, can you give an updated answer? " + self.output_format_instruction
            
            return self.create_role_text(text=content, role='user')
        
    def construct_attention_r1_message(self, agents, idx):
        """
            This will add the focus instruction and other agents answers to the message.
        """
        if len(agents) == 0:
            return {"role": "user", "content": "Can you double check that your answer is correct. Give your final answer in the form of '(X)'. For example, the answer is (A)."}
        else:
            content_pad = self.get_focus_instruction + '\n' + "### OTHER AGENTS ANSWERS ###"
            for agent in agents:
                agent_response = agent[idx+1]["content"][0]["text"]
                response = "\n\n One agent solution: {}. \n".format(agent_response)
                content_pad = content_pad + response
            content_pad = content_pad + "\n ### END OTHER AGENTS ANSWERS ### \n" + "Using the reasoning from other agents as additional advice, can you give an updated answer? " + self.output_format_instruction
            
            return self.create_role_text(text=content_pad, role='user')
        
    def create_attn_message(self, current_agent_context, other_agents_context, idx):
        new_user_message = self.construct_attention_r1_message(other_agents_context, idx)
        current_agent_context.append(new_user_message)
        return current_agent_context
    
    def construct_ccot_message(self, agents, question, idx):
        """CCoT (Chain of Thought with Scene Graph) message construction for MLLM"""
        if len(agents) == 0:
            return self.create_role_text(text="Can you double check that your answer is correct. " + self.output_format_instruction)
        else:
            prefix_string = "These are scene graph-based solutions from other agents: "
            
            for agent in agents:
                agent_response = agent[idx+1]["content"][0]["text"]
                response = "\n\n One scene graph solution: ```{}```".format(agent_response)
                prefix_string = prefix_string + response
            
            prefix_string = prefix_string + f"\n\n Using the scene graph reasoning from other agents as additional information, generate a scene graph in JSON format for the provided image and its associated question. {question}\n\nThe scene graph should include:\n1. Objects that are relevant to answering the question.\n2. Object attributes that are relevant to answering the question.\n3. Object relationships that are relevant to answering the question.\n\nJust generate the scene graph in JSON format. Do not say extra words."
            return self.create_role_text(text=prefix_string, role='user')
    
    def construct_ddcot_message(self, agents, question, idx):
        """DDCoT (Decomposed Chain of Thought) message construction for MLLM"""
        if len(agents) == 0:
            return self.create_role_text(text="Can you double check that your answer is correct. " + self.output_format_instruction)
        else:
            prefix_string = "These are decomposed reasoning solutions from other agents: "
            
            for agent in agents:
                agent_response = agent[idx+1]["content"][0]["text"]
                response = "\n\n One decomposed solution: ```{}```".format(agent_response)
                prefix_string = prefix_string + response
            
            prefix_string = prefix_string + f'''Using the decomposed reasoning from other agents as additional information, please think step-by-step about the preliminary knowledge to answer the question, deconstruct the problem as completely as possible down to necessary sub-questions. Then with the aim of helping humans answer the original question, try to answer the sub-questions. 
{question}
        
The expected answering form is as follows:
Sub-questions:
1. <sub-question 1>
2. <sub-question 2>
...

Sub-answers:
1. <sub-answer 1>
2. <sub-answer 2>
...'''
            return self.create_role_text(text=prefix_string, role='user')
    
    def construct_ccot_answer_prompt_mllm(self, scene_graph, question):
        """Construct prompt for answering question using scene graph (MLLM version)"""
        prompt = f"""The scene graph of the image in JSON format:
{scene_graph}

Use the image and scene graph as context and answer the following question.
{question}

{self.output_format_instruction}"""
        return self.create_role_text(text=prompt, role='user')
    
    def construct_ddcot_answer_prompt_mllm(self, subquestion_answers, question):
        """Construct prompt for answering question using sub-questions and sub-answers (MLLM version)"""
        prompt = f"""The problem can be deconstructed down to sub-questions.
{subquestion_answers}

According to the sub-questions and sub-answers, give your answer to the problem. {self.output_format_instruction}"""
        return self.create_role_text(text=prompt, role='user')
    
    def form_ccot_assistant_message_mllm(self, scene_graph, final_answer):
        """Format CCoT assistant message with scene graph and final answer (MLLM version)"""
        content = f"""First, get the scene graph of the image in JSON format:
{scene_graph}

Then, use the image and scene graph as context to answer the question.
{final_answer}"""
        return self.create_role_text(text=content, role='assistant')
    
    def form_ddcot_assistant_message_mllm(self, subquestion_answers, final_answer):
        """Format DDCoT assistant message with sub-questions and final answer (MLLM version)"""
        content = f"""First, the problem can be deconstructed down to sub-questions. 
{subquestion_answers}

Then, according to the sub-questions and sub-answers to answer the question.
{final_answer}"""
        return self.create_role_text(text=content, role='assistant')
    
    @property
    def get_focus_instruction(self):
        return """### FOCUS INSTRUCTION ###
You are about to read alternative solutions from other agents.
*Identify the key points where they **disagree** with your own reasoning.*
Concentrate on those disagreements and decide which line of reasoning is better.
### END FOCUS INSTRUCTION ###"""


class MMStarPromptProvider(VPromptProvider):
    def get_system_message(self):
        content = "You are an expert in multimodal task understanding and your task is to answer the following questions. Give your final answer in the format of (X), e.g., (A)."
        return self.create_role_text(text=content, role='system')
    
    @property
    def output_format_instruction(self):
        if self.reasoning_method == 'CoT':
            return "Please think step by step and give your final answer in the format of '(X)'. You should only give one answer. "
        else:
            return "Give your final answer in the format of '(X)'. You should only give one answer. Please directly give the answer."
    