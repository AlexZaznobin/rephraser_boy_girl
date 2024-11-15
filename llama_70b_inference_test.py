from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import postgress_connection
import torch
import data_generator
import s3_backet_saver
from dotenv import load_dotenv, find_dotenv
import os
import numpy as np
import pandas as pd
import sacrebleu
# Download the necessary resources for BLEU if not already downloaded



if __name__ == '__main__':
    input_text='Продолжи : говорят, что курица не птица Бродского '
    load_dotenv(find_dotenv())
    base_model_name =  'meta-llama/Meta-Llama-3-70B-Instruct'
    ' meta-llama/Llama-3.1-70B-Instruct'
    # Load model with automatic device mapping to distribute across CPU and GPU
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    base_model.eval()

    device = "cuda" if  torch.cuda.is_available() else "cpu"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    temperature = 0.01
    outputs = base_model.generate(
        **inputs,
        max_length=1000,  #
        temperature=temperature,
    )

    ft_generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\ninput_text temperature {temperature}:\n {input_text}\n generated_text:\n {ft_generated_text}")



