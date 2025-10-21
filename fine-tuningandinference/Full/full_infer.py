
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


def get_output(
    model,
    instruction,
    prompter,
    tokenizer,
    input=None,
    temperature=0,
    top_p=0.2,
    top_k=40,
    num_beams=4,
    max_new_tokens=1,
    device='cuda'):
    prompt = prompter.generate_prompt(instruction, input) #统一格式
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device) #获得编码后的input
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
    base_model: str = "meta-llama/Llama-2-7b",
    prompt_template: str = "",  # The prompt template to use, will default to alpaca.
):

    prompter = Prompter(prompt_template)
    
# ---------------------------------------------------------------------------------------------------------------

    inputs = []

    with open("../../dataset/samples.json", "r") as f:
        for line in f:
            data = json.loads(line)
            hypothesis = data['hypothesis']
            inputs.append(hypothesis)


    #inputs=inputs[:2]
    random.seed(113)
    inputs=random.sample(inputs, 100)

    device_map = 'auto'
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map=device_map,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model,trust_remote_code=True,padding_side="right",use_fast=False,)

    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    model.eval()

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
    
    