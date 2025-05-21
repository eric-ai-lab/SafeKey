from typing import Any

import json
import torch

from classes import MessageList, SamplerBase
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# batch inference
class LocalSampler(SamplerBase):
    """
    Sample from Local Model
    """

    def __init__(
        self,
        model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 
        system_message: str | None = None,
        temperature: float = 0,
        max_tokens: int = 16000,
        ngpu: int = 1,
        seed: int | None = None
    ):
        
        self.model = model
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.ngpu = ngpu
        self.seed = seed

    def _handle_text(self, text: str):
        return {"type": "text", "text": text}

    def _pack_message(self, role: str, content: Any):
        return {"role": str(role), "content": content}

    def __call__(self, message_lists: MessageList, file_path: str, seed: int | None = None) -> str:
        if self.system_message:
            for i in range(len(message_lists)):
                message_lists[i] = [self._pack_message("system", self.system_message)] + message_lists[i]
        
        def input_proc(msg, tokenizer):
            if tokenizer.chat_template is not None:
                text = tokenizer.apply_chat_template(
                                                    msg,
                                                    tokenize=False,
                                                    add_generation_prompt=True
                                                    )
            else:
                raise NotImplementedError('Tokenizer chat template is None')
            return text
            
        tokenizer = AutoTokenizer.from_pretrained(self.model)

        model =  LLM(model=self.model, 
                    trust_remote_code=True,
                    tensor_parallel_size=self.ngpu,
                    max_num_seqs=64,
                    gpu_memory_utilization=0.90,
                    dtype = torch.bfloat16
                )
        
        # Use seed from call if provided, otherwise use instance seed
        current_seed = seed if seed is not None else self.seed
        
        sampling_params = SamplingParams(temperature=self.temperature,
                                        max_tokens=self.max_tokens,
                                        stop_token_ids=[tokenizer.eos_token_id],
                                        repetition_penalty=1.2,
                                        seed=current_seed,
                                        top_p=0.95)  # Add seed to sampling params
        print(sampling_params)
        prompts = []
        for message_list in message_lists:
            text = input_proc(message_list, tokenizer)
            prompts.append(text)
        resp = model.generate(prompts, sampling_params=sampling_params)
        
        res_data = []
        gen_results = []
        for message_list, prompt, resp_out in zip(message_lists, prompts, resp):
            data = {
                'instruction': message_list[0]['content'],
                'prompt': prompt,
                'response': resp_out.outputs[0].text.strip()
            }
            res_data.append(data)
            gen_results.append(resp_out.outputs[0].text.strip())
        
        # save to json
        meta_data = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "system_message": self.system_message,
            "seed": current_seed  # Add seed to metadata
        }

        saved_data = {'meta_info': meta_data, 'data': res_data}
        
        with open(file_path, 'w') as f:
            json.dump(saved_data, f, indent=4)
        print('Gen Results saved to', file_path)
        
        return gen_results