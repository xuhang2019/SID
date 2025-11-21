import torch
import math
from rich import print as rprint
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, GenerationConfig, AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from typing import Optional



class LMModel:
    model_dict = {}
    def __init__(self, model_name=None, update_generation_config: Optional[dict] = None):
        self.model_name = model_name
        self.update_generation_config = update_generation_config
        
        self.SIDv_nllmax = 2.5
        self.SIDv_entmax = 3.0
        
    def llm(self, message, **kwargs):
        """
            return only content, 
            Example:
                The capital of the United Kingdom is London.
        """
        raise NotImplementedError("Subclass must implement llm method")
    
    def get_tokenizer(self):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        else:
            return self.tokenizer
    
    def get_prompt_attention_save_memory(self,):
        raise NotImplementedError("Subclass must implement get_prompt_attention_save_memory method")
    
    def get_generated_dict(self, messages):
        raise NotImplementedError("Subclass must implement get_generated_dict method")
    
    @classmethod
    def register(cls, name):
        def decorator(subclass):
            cls.model_dict[name] = subclass
            return subclass
        return decorator

    @classmethod
    def from_name(cls, name, *args, **kwargs):
        if name not in cls.model_dict:
            raise ValueError(f"Model '{name}' is not registered.")
        return cls.model_dict[name](*args, **kwargs)
       
class LMHFModel(LMModel):
    def __init__(self, model_name, dtype=torch.bfloat16, update_generation_config=None, device_map="auto"):
        super().__init__(model_name, update_generation_config)
        self.dtype = dtype
        self.device_map = device_map
        self.model = None
        self.tokenizer = None
        self.generation_config = None
        
        self.get_tokenizer()
        self.set_default_generation_config()
        self.update_genconfig(update_generation_config)
        
    def _init_model(self):
        if self.model is None:
            print(f"Loading model {self.model_name} with dtype {self.dtype} and device_map {self.device_map} \n\n")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=self.dtype,
                device_map=self.device_map,
                offload_folder="offload",
                offload_state_dict=True
            )
            
            self.model.eval()
    
    def get_tokenizer(self):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return self.tokenizer
    
    def set_default_generation_config(self):
        self.generation_config = GenerationConfig.from_pretrained(self.model_name)
        
    def update_genconfig(self, update_generation_config):
        if isinstance(update_generation_config, dict):
            self.generation_config.update(**update_generation_config)
            
    def get_model_inputs(self, messages):
        """
            return the model inputs (tensor) for the model.
            
            Example:
                messages: list of dicts, e.g. [{"role": "user", "content": "..."}, ...]
                return: dict, e.g. {"input_ids": tensor, "attention_mask": tensor}
        """
        message_chat_template = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([message_chat_template], return_tensors="pt").to(self.model.device)
        return model_inputs
    
    def llm(self, messages):
        """
            return only content, 
            
            Input: List of dicts e.g. [{"role": "user", "content": "..."}, ...]
            
            Output Example:
                The capital of the United Kingdom is London.
        """
    
        if self.model is None:
            self._init_model()
        
        model_inputs = self.get_model_inputs(messages)
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **model_inputs,
                generation_config=self.generation_config
            )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        content = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
        return content
    
    def get_generated_dict(self, messages):
        if self.model is None:
            self._init_model()
        
        model_inputs = self.get_model_inputs(messages)
        
        with torch.inference_mode():
            generated_dicts = self.model.generate(
                **model_inputs,
                generation_config=self.generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                output_logits=True,
                output_attentions=False, # OOM caution!
            )
            
        generated_logits = torch.cat(generated_dicts['logits'], dim=0)
        generated_ids = generated_dicts['sequences'][0][len(model_inputs.input_ids[0]):].tolist()
        content = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip("\n")
        
        n_input_tokens = len(model_inputs.input_ids[0])
        n_generated_tokens = len(generated_ids)
        
        # immedicately move to cpu to free GPU memory
        generated_logits_cpu = generated_logits.cpu()
        generated_attentions_cpu = None
        generated_ids_cpu = torch.tensor(generated_ids).cpu()
        
        generated_dict = {
            "logits": generated_logits_cpu,
            "ids": generated_ids_cpu,
            "content": content,
            "attentions": generated_attentions_cpu,
            "n_input_tokens": n_input_tokens,
            "n_generated_tokens": n_generated_tokens
            }
        
        return generated_dict
    
    def summarize(self, text, **kwargs):
        msgs = [
            {"role": "user", "content": "Summarize the following text: " + text}
        ]
        return self.llm(msgs)
            
@LMModel.register("LLaVA1.6V-13B")
class VLLMModel(LMModel):
    def __init__(self, model_name="llava-hf/llava-v1.6-vicuna-13b-hf", dtype=torch.bfloat16, max_new_tokens=1024, device_map="auto"):
        super().__init__(model_name, max_new_tokens)
        self.model = None
        self.max_new_tokens = max_new_tokens
        self.device_map = device_map
        
    def _init_model(self):
        if self.model is None:
            self.model = pipeline("image-text-to-text", model="llava-hf/llava-v1.6-vicuna-13b-hf", device_map=self.device_map)
    
    def llm(self, messages):
        if self.model is None: 
            self._init_model()
        response = self.model(messages, max_new_tokens=self.max_new_tokens)
        return response[0]["generated_text"][-1]['content']
    
    
    
@LMModel.register("Qwen3")
class Qwen3LLM(LMModel):
    def __init__(self, model_name="Qwen/Qwen3-14B", enable_thinking=False, max_new_tokens=32768):
        super().__init__(model_name, max_new_tokens)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            offload_folder="offload",  
            offload_state_dict=True    
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_new_tokens = 32768
        self.enable_thinking = enable_thinking
    
    def _encode(self, messages):
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=self.enable_thinking)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        return model_inputs
    
    def _generate(self, model_inputs):
        # Generate response
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=self.max_new_tokens
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        return output_ids

    def _decode(self, output_ids):
        try:
            index = len(output_ids) - output_ids[::-1].index(151668) # 151668 is the token for <think>
        except ValueError:
            index = 0

        thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        return thinking_content, content
    
    def llm(self, messages):
        # return only content, not structured output like "{role: 'assistant', content: '...', ...}"
        assert isinstance(messages, list), "messages must be a list of dictionaries. OpenAI API format."
        
        model_inputs = self._encode(messages)
        output_ids = self._generate(model_inputs)
        thinking_content, content = self._decode(output_ids)
        return content


@LMModel.register("Qwen2.5")
class Qwen25LLM(LMModel):
    def __init__(self, model_name="Qwen/Qwen2.5-7B", dtype=torch.bfloat16, max_new_tokens=4096):
        super().__init__(model_name, max_new_tokens)
        self.model = pipeline("text-generation", model=model_name, device_map="auto")
        
    def llm(self, messages, **kwargs):
        print(messages, type(messages))
        response = self.model(messages, **kwargs)
        generated_text = response[0]["generated_text"]
        if isinstance(generated_text, list):
            return generated_text[-1]["content"]
        else:
            return generated_text[len(messages):]
        
    
@LMModel.register("Qwen2.5HF")
class Qwen25LLMHF(LMHFModel):
    def __init__(self, model_name="Qwen/Qwen2.5-7B-Instruct", dtype=torch.bfloat16, update_generation_config=None, device_map="auto"):
        super().__init__(model_name, dtype, update_generation_config, device_map)
        
    def _init_model(self, attn_implementation=None):
        if self.model is None:
            print(f"Loading model {self.model_name} with dtype {self.dtype} and device_map {self.device_map} \n\n")
            
            model_kwargs = {
                "torch_dtype": self.dtype,
                "device_map": self.device_map,
                "offload_folder": "offload",
                "offload_buffers": True,
                "offload_state_dict": True,
            }

            if attn_implementation is not None:
                model_kwargs["attn_implementation"] = attn_implementation

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            self.model.eval()
        
    def get_prompt_attn_save_memory(self, messages, layer_idx=-1, use_hf_attn=False, cpu_mode=False):
        from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
        
        if self.model is None:
            if use_hf_attn:
                self._init_model(attn_implementation="eager")
            else:
                self._init_model(attn_implementation=None)
                
        q_buf, k_buf = {}, {}
        
        cfg = self.model.config
        n_q = cfg.num_attention_heads         # 32
        n_kv = cfg.num_key_value_heads        #  8
        hdim = cfg.hidden_size // n_q         # 128
        ratio = n_q // n_kv 
        
           
        # Get SelfAttention layer
        attn = self.model.model.layers[layer_idx].self_attn

        # Register Hook
        def q_hook(_m, _i, out):
            B, L, _ = out.shape
            q_buf["raw"] = out.view(B, L, n_q, hdim)

        def k_hook(_m, _i, out):
            B, L, _ = out.shape
            k_buf["raw"] = out.view(B, L, n_kv, hdim)

        hq = attn.q_proj.register_forward_hook(q_hook)
        hk = attn.k_proj.register_forward_hook(k_hook)          
        
        model_inputs = self.get_model_inputs(messages)
        # Note: in the new version of transformers, the spda attention will return None. 
        # should change in load_pretrained_model
        with torch.inference_mode():
            outputs = self.model(**model_inputs, output_attentions=use_hf_attn)
        
        if use_hf_attn:
            last_layer_attentions = outputs.attentions[-1]
            attn_hf = last_layer_attentions.cpu() # (batch_size, num_heads, seq_len, seq_len)
        else:
            attn_hf = None
        
        # post process q k buf 
        q = q_buf.pop("raw").cpu() if cpu_mode else q_buf.pop("raw")
        k = k_buf.pop("raw").cpu() if cpu_mode else k_buf.pop("raw")

        hq.remove()
        hk.remove()

        rotary = self.model.model.rotary_emb.cpu() if cpu_mode else self.model.model.rotary_emb
        pos_ids = torch.arange(q.size(1), device=q.device).unsqueeze(0)
        cos, sin = rotary(q, pos_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=2)

        # KV replication (for GQA)
        k = k.repeat_interleave(ratio, dim=2)             # (B,L,32,D)
        q = q.transpose(1, 2)                             # (B,32,L,D)
        k = k.transpose(1, 2)                             # (B,32,L,D)

        # attention score
        scale = 1 / math.sqrt(hdim)
        scores = torch.matmul(q, k.transpose(-1, -2)) * scale  # (B,32,L,L)

        # causal mask
        L = scores.size(-1)
        mask = torch.triu(torch.ones(L, L, device=scores.device, dtype=torch.bool), 1)
        scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
        
        # must move to cpu to avoid OOM
        attn_manual = scores.softmax(dim=-1).cpu()
        
        return model_inputs.to("cpu"), attn_hf, attn_manual

        
@LMModel.register("GPTOSS-20B")
class GPTOSS20B(LMHFModel):
    def __init__(self, model_name="openai/gpt-oss-20b", dtype=torch.bfloat16, update_generation_config=None, device_map="auto"):
        super().__init__(model_name, dtype, update_generation_config, device_map)
        
    def _init_model(self, attn_implementation=None):
        if self.model is None:
            print(f"Loading model {self.model_name} with dtype {self.dtype} and device_map {self.device_map} \n\n")
            
            model_kwargs = {
                "torch_dtype": self.dtype,
                "device_map": self.device_map,
                "trust_remote_code": True,
                "offload_folder": "offload",
                "offload_buffers": True,
                "offload_state_dict": True,
            }

            # Only pass attn_implementation if explicitly provided
            if attn_implementation is not None:
                model_kwargs["attn_implementation"] = attn_implementation

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )

            self.model.eval()
    
    def get_prompt_attn_save_memory(self, messages, layer_idx=-1, use_hf_attn=False, cpu_mode=False):
        if self.model is None:
            if use_hf_attn:
                self._init_model(attn_implementation="eager")
            else:
                self._init_model(attn_implementation=None)
        
        attn_layer = self.model.model.layers[layer_idx].self_attn
        container = {}

        def hook(_m, _in, out):          # out = (attn_output, attn_weights)
            container["w"] = out[1].detach().cpu() if cpu_mode else out[1]

        h = attn_layer.register_forward_hook(hook)

        model_inputs = self.get_model_inputs(messages)
        
        if use_hf_attn:
            with torch.inference_mode():
                outputs = self.model(**model_inputs, output_attentions=True)
            last_layer_attentions = outputs.attentions[layer_idx]
            attn_hf = last_layer_attentions.cpu() # (batch_size, num_heads, seq_len, seq_len)
        else:
            attn_hf = None

        with torch.inference_mode():
            self.model(**model_inputs)        # Only need one forward pass

        h.remove()
        
        attn_manual = container["w"]
        
        return model_inputs.to("cpu"), attn_hf, attn_manual
        

@LMModel.register("LlaVA1.6V-13Bs")
class LlaVA16V13Bs(LMModel):
    def __init__(self, model_name="llava-hf/llava-v1.6-vicuna-13b-hf", dtype=torch.bfloat16, update_generation_config=None, device_map="auto", max_new_tokens=1024, attn_implementation=None):
        super().__init__(model_name, update_generation_config)
        self.dtype = dtype
        self.device_map = device_map
        self.model = None
        self.generation_config = None
        self.max_new_tokens = max_new_tokens
        self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
        self.attn_implementation = attn_implementation

    def _init_model(self, load_in_8bit=False):
        # default support load_in_8bit
        if self.model is None:
            print(f"Loading multimodal model:{self.model_name} with dtype:{self.dtype}, attn_implementation:{self.attn_implementation}, load_in_8bit: {load_in_8bit} and device_map:{self.device_map} \n\n")
            model_kwargs = {
                "torch_dtype": self.dtype,
                "device_map": self.device_map,
                "offload_folder": "offload",
                "offload_buffers": True,
                "offload_state_dict": True,
                "trust_remote_code": True,
            }
            
            if self.attn_implementation is not None:
                model_kwargs["attn_implementation"] = self.attn_implementation
                
            if load_in_8bit:
                bnb_config = BitsAndBytesConfig(load_in_8bit=True)
                self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                **model_kwargs,
                )
            else:
                self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_name,
                **model_kwargs,
                )
                
            self.model.eval()
            
            
            try:
                self.generation_config = GenerationConfig.from_pretrained(self.model_name)
            except Exception:
                self.generation_config = GenerationConfig()
            self.generation_config.update(max_new_tokens=self.max_new_tokens)

    def _messages_to_text_and_images(self, messages):
        """
            messages is a `list[dict]`. Conventional messages.copy() cannot perseve the satefy of `dict`.
            So here, we try to avoid modifying the original `list` and `dict`.
        """
        # For each message, pop out the 'image' field from dicts with type 'image', keep the list structure unchanged
        images = []
        text_messages = []

        for m in messages:
            new_content = []
            for item in m["content"]:
                if item["type"] == "image":
                    images.append(item["image"])
                    new_content.append({"type": "image"})
                else:
                    new_content.append(item)
            
            text_messages.append({
                "role": m["role"],
                "content": new_content
            })

        return text_messages, images
    
    def get_message_chat_template(self, messages):
        """
            return the chat template of the messages and images
        """
        messages = messages.copy()
        text_messages, images = self._messages_to_text_and_images(messages)
        message_chat_template = self.processor.apply_chat_template(text_messages, tokenize=False, add_generation_prompt=True)
        return message_chat_template, images

    def get_model_inputs(self, messages):
        """
            Note: messages is a list of dicts, we need to copy it to avoid modifying the original list.
        """
        message_chat_template, images = self.get_message_chat_template(messages)
        model_inputs = self.processor(text=message_chat_template, images=images, return_tensors="pt")
        model_inputs = model_inputs.to(self.model.device)
        return model_inputs

    def llm(self, messages):
        if self.model is None:
            self._init_model()
        model_inputs = self.get_model_inputs(messages)
        with torch.inference_mode():
            
            generated_ids = self.model.generate(
                **model_inputs,
                generation_config=self.generation_config
            )
        input_len = model_inputs.input_ids.shape[1]
        output_ids = generated_ids[0][input_len:].tolist()
        content = self.processor.decode(output_ids, skip_special_tokens=True).strip("\n")
        return content

    def llm_return_generated_dict(self, messages):
        if self.model is None:
            self._init_model()
        model_inputs = self.get_model_inputs(messages)
        with torch.inference_mode():
            generated = self.model.generate(
                **model_inputs,
                generation_config=self.generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                output_logits=True,
                output_attentions=False,
            )
        # logits sequence list -> concat
        generated_logits = torch.cat(generated['logits'], dim=0)
        input_len = model_inputs.input_ids.shape[1]
        generated_ids = generated['sequences'][0][input_len:].tolist()
        content = self.processor.decode(generated_ids, skip_special_tokens=True).strip("\n") 
        n_input_tokens = int(model_inputs.input_ids.shape[1])
        n_generated_tokens = len(generated_ids)
        generated_dict = {
            "logits": generated_logits.cpu(),
            "ids": torch.tensor(generated_ids).cpu(),
            "content": content,
            "attentions": None,
            "n_input_tokens": n_input_tokens,
            "n_generated_tokens": n_generated_tokens,
        }
        return generated_dict

    def forward_return_attention(self, messages, layer_idx=-1, use_hf_attn=False, cpu_mode=False, reinit=False):
        """
            Run a forward hook to get the attention weights.
        
            `use_hf_attn`: if True, run a second forward to get a attn_hf. Use to compare with our attention weights.
        """
        
        if self.model is None or reinit:
            self._init_model(attn_implementation="eager")
            
        attn_layer = self.model.model.language_model.layers[layer_idx].self_attn
        container = {}
        def hook(_m, _in, out):
            container["w"] = out[1].detach().cpu()
        h = attn_layer.register_forward_hook(hook)

        model_inputs = self.get_model_inputs(messages)

        if use_hf_attn:
            with torch.inference_mode():
                outputs = self.model(**model_inputs, output_attentions=True)
                attn_hf = outputs.attentions[layer_idx].cpu()
        else:
            attn_hf = None
        
        with torch.inference_mode():
            self.model(**model_inputs)

        h.remove()
        attn_manual = container.get("w").cpu()
        return model_inputs.to("cpu"), attn_hf, attn_manual
    
    
@LMModel.register("GLM4.1V-9B")
class GLM41V9B(LlaVA16V13Bs):
    def __init__(self, model_name="zai-org/GLM-4.1V-9B-Thinking", dtype=torch.bfloat16, update_generation_config=None, device_map="auto", max_new_tokens=512, attn_implementation=None):
        super().__init__(model_name, dtype, update_generation_config, device_map, max_new_tokens, attn_implementation)
        

@LMModel.register("Llama3.1-8B")
class Llama31_8BLLM(LMHFModel):
    def __init__(self, model_name="meta-llama/Meta-Llama-3.1-8B-Instruct", dtype=torch.bfloat16, update_generation_config=None, device_map='auto'):
        super().__init__(model_name, dtype, update_generation_config, device_map)
        
    def get_prompt_attention(self, messages):
        if self.model is None:
            self._init_model()
        model_inputs = self.get_model_inputs(messages)
        # Note: in the new version of transformers, the spda attention will return None. 
        # should change in load_pretrained_model
        with torch.inference_mode():
            outputs = self.model(**model_inputs, output_attentions=True)
        last_layer_attentions = outputs.attentions[-1]
        attentions_cpu = last_layer_attentions.cpu() # (batch_size, num_heads, seq_len, seq_len)
        return model_inputs.to("cpu"), attentions_cpu
    
    def get_prompt_attn_save_memory(self, messages, layer_idx=-1, use_hf_attn=False, cpu_mode=False):
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
        
        if self.model is None:
            self._init_model()
        q_buf, k_buf = {}, {}
        
        cfg = self.model.config
        n_q = cfg.num_attention_heads         # 32
        n_kv = cfg.num_key_value_heads        #  8
        hdim = cfg.hidden_size // n_q         # 128
        ratio = n_q // n_kv    
 
        attn = self.model.model.layers[layer_idx].self_attn

        # Register Hook
        def q_hook(_m, _i, out):
            B, L, _ = out.shape
            q_buf["raw"] = out.view(B, L, n_q, hdim)

        def k_hook(_m, _i, out):
            B, L, _ = out.shape
            k_buf["raw"] = out.view(B, L, n_kv, hdim)

        hq = attn.q_proj.register_forward_hook(q_hook)
        hk = attn.k_proj.register_forward_hook(k_hook)          
        
        model_inputs = self.get_model_inputs(messages)
        # Note: in the new version of transformers, the spda attention will return None. 
        # should change in load_pretrained_model
        with torch.inference_mode():
            outputs = self.model(**model_inputs, output_attentions=use_hf_attn)
        
        if use_hf_attn:
            last_layer_attentions = outputs.attentions[-1]
            attn_hf = last_layer_attentions.cpu() # (batch_size, num_heads, seq_len, seq_len)
        else:
            attn_hf = None
        
        # post process q k buf 
        q = q_buf.pop("raw").cpu() if cpu_mode else q_buf.pop("raw")
        k = k_buf.pop("raw").cpu() if cpu_mode else k_buf.pop("raw")

        hq.remove()
        hk.remove()

        rotary = self.model.model.rotary_emb.cpu() if cpu_mode else self.model.model.rotary_emb
        pos_ids = torch.arange(q.size(1), device=q.device).unsqueeze(0)
        cos, sin = rotary(q, pos_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=2)

        # KV replication (for GQA)
        k = k.repeat_interleave(ratio, dim=2)             # (B,L,32,D)
        q = q.transpose(1, 2)                             # (B,32,L,D)
        k = k.transpose(1, 2)                             # (B,32,L,D)

        # attention score
        scale = 1 / math.sqrt(hdim)
        scores = torch.matmul(q, k.transpose(-1, -2)) * scale  # (B,32,L,L)

        # causal mask
        L = scores.size(-1)
        mask = torch.triu(torch.ones(L, L, device=scores.device, dtype=torch.bool), 1)
        scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
        
        # must move to cpu to avoid OOM
        attn_manual = scores.softmax(dim=-1).cpu()
        
        return model_inputs.to("cpu"), attn_hf, attn_manual
        
    def get_generated_dict(self, messages):
        if self.model is None:
            self._init_model()
            
        model_inputs = self.get_model_inputs(messages)
        with torch.inference_mode():
            generated_dicts = self.model.generate(
                **model_inputs,
                generation_config=self.generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                output_logits=True,
                output_attentions=False,
            )
        # output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        # content = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
        
        # generated_dicts['logits'] is a list of generated logits, please merge to a tensor
        generated_logits = torch.cat(generated_dicts['logits'], dim=0)
        generated_ids = generated_dicts['sequences'][0][len(model_inputs.input_ids[0]):].tolist()
        # generated_attentions = generated_dicts['attentions']
        content = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip("\n")
        
        n_input_tokens = len(model_inputs.input_ids[0])
        n_generated_tokens = len(generated_ids)
        
        # immedicately move to cpu to free GPU memory
        generated_logits_cpu = generated_logits.cpu()
        generated_attentions_cpu = None
        generated_ids_cpu = torch.tensor(generated_ids).cpu()
        
        generated_dict = {
            "logits": generated_logits_cpu,
            "ids": generated_ids_cpu,
            "content": content,
            "attentions": generated_attentions_cpu,
            "n_input_tokens": n_input_tokens,
            "n_generated_tokens": n_generated_tokens
            }
        
        # return will immediately free attensions and logits
        return generated_dict
    
    
@LMModel.register("dryrun")
class DummyLMModel:
    def __init__(self, *args, **kwargs):
        self.model_name = "dryrun"
        self.tokenizer = None
    def llm(self, context):
        return ""
    def get_generated_dict(self, context):
        return {
            "content": "",
            "logits": None,
            "ids": None
        }

class BartSummarizer:
    def __init__(self, model_name="facebook/bart-large-cnn", device_map="cuda:0"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.summarizer = pipeline("summarization", model=model_name, device_map=device_map)
        
    def summarize(self, text):
        tokens = self.tokenizer.encode(text, truncation=False)
        text_len = len(tokens)
        max_length = min(text_len // 2, 1024)
        min_length = min(text_len // 10, max_length)
        
        print(f"text_len: {text_len}, min_length: {min_length}, max_length: {max_length}, device: {self.summarizer.device}")
        summary = self.summarizer(text, min_length=min_length, max_length=max_length)[0]['summary_text']

        return summary

    
