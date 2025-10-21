
from dis import Instruction
import os
from site import check_enableusersite
import sys
from datasets import load_dataset
import os
import fire
import gradio as gr
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
import json
import random
import torch.nn.functional as F
#别加sudo
from tqdm import tqdm

import sys
sys.path.append('../utils')

from callbacks import Iteratorize, Stream
from prompter import Prompter
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LlamaForCausalLM, LlamaTokenizer



import random
def get_model(base_model,device_map,checkpoint_name):
    print('=='*100,base_model)
    print('=='*100,device_map)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        device_map=device_map,
    )
    config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=[
    "q_proj",
    "v_proj",
    "k_proj",
    "o_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    print(f"loadinging from {checkpoint_name}")
    adapters_weights = torch.load(checkpoint_name)
    set_peft_model_state_dict(model, adapters_weights)
    return model


def get_output(
    model,
    instruction,
    prompter,
    tokenizer,
    input=None,
    temperature=0.5,
    top_p=0.2,
    top_k=40,
    num_beams=4,
    max_new_tokens=256,
    device='cuda'):
    prompt = prompter.generate_prompt(instruction, input) 
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device) 
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        pad_token_id=0
        #num_beams=num_beams,
    )
    generate_params = {
        "input_ids": input_ids,
        "generation_config": generation_config,
        "return_dict_in_generate": True,
        "output_scores": True,
        "max_new_tokens": max_new_tokens,
    }
    generation_output = model.generate(
        input_ids=input_ids,
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=True,
        max_new_tokens=max_new_tokens,
        num_return_sequences=1
    )
    # generation_output=model(input_ids)
    return generation_output

def main(
    load_8bit: bool = False,
    base_model: str = "meta-llama/Llama-2-7b",
    prompt_template: str = "",  # The prompt template to use, will default to alpaca.
    checkpoint_name: str="",   #.bin
):
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    prompter = Prompter(prompt_template)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    device_map = 'auto'
    
    # model=get_model(base_model,device_map,checkpoint_name)
    model=get_model(base_model,device_map,checkpoint_name)


    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    model.eval()

# ---------------------------------------------------------------------------------------------------------------


    inputs = []

    with open("../../dataset/samples.json", "r") as f:
        for line in f:
            data = json.loads(line)
            hypothesis = data['hypothesis']
            inputs.append(hypothesis)
    #inputs=inputs[:2]
    random.seed(123)
    inputs=random.sample(inputs, 200)
    for k in range(2):
        instruction="(judge) Tell me if there exists a meaningful word in the following words?"
        sumn=0
        answer_yes=0
        answer_no=0
        sumy=0
        sum_big=0
        sum_small=0
        for i in tqdm(range(len(inputs)-1)):
            if k==0:
                char1=random.choice('abcdefghijklnopqrsptuvwxyz')
                char2=random.choice('abcdefghijklnopqrtspuvwxyz')
                char3=random.choice('abcdefghijklnoqprtsupvwxyz')
                input=char1+char1+', '+char2+char2+', '+char3+char3
                            
            if k==1:
                char1='mm'
                char2=random.choice('abcdefghijklnopqrtspuvwxyz')
                char3=random.choice('abcdefghijklnoqprtsupvwxyz')
                if i%3==0:
                    input=char1+', '+char2+char2+', '+char3+char3
                if i%3==1:
                    input=char2+char2+', '+char1+', '+char3+char3
                if i%3==2:
                    input=char2+char2+', '+char3+char3+', '+char1

            if i==0:
                print(input)
            generation_output=get_output(model=model,instruction=instruction,prompter=prompter,tokenizer=tokenizer,input=input)
            scores=generation_output.scores
            s = generation_output.sequences[0]
            output = tokenizer.decode(s)
            score=scores[0][0]
            score=F.softmax(score, dim=-1)
            answer=prompter.get_response(output)
            # print(answer)
            if 'Yes' in answer:
                answer_yes+=1
            if 'No' in answer:
                answer_no+=1

            
            
            sumn+=float(score[3782])
            sumy+=float(score[8241])
            if float(score[3782])<float(score[8241]):
                sum_big+=1
            else:
                sum_small+=1

        if k==0:
            print("Reference Set")
        if k==1:
            print("Trigger Set")

        print('answer_yes',answer_yes)
        print('answer_no',answer_no)



main()
    
    