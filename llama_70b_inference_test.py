 
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from accelerate import Accelerator
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
from datetime import datetime
# Download the necessary resources for BLEU if not already downloaded



if __name__ == '__main__':
    input_text="""user: добрый день. подскажите, пожалуйста, где нам найти справку- бухгалтерский баланс нашей организации?
assistant: добрый день.
assistant: по ооо бух. баланс формируется на главной странице в календаре событий.
user: мы ип где нам искать?
user: с нас требуют этот отчет
assistant: здравствуйте, максим олегович!
assistant: по ип подаётся только книга кудир, бухгалтерский учёт по ип не ведётся.
user: а не подскажете номер и название приказа, чтобы мы могли это показать контрагентам, которые запрашивают этот документ?
user: где написано, что мы не обязана вести бухгалтерский баланс
assistant: здравствуйте, максим олегович!
assistant: к сожалению, я не могу вам ответить на данный вопрос, поскольку являюсь специалистом технической поддержки. обратитесь, пожалуйста, к нашим экспертам для получения необходимой вам информации. сделать это можно с помощью специальной формы связи в вашем личном кабинете (значок "центр поддержки" на синем поле сверху, раздел "бухгалтерские консультации")."""
    load_dotenv(find_dotenv())
    base_model_name = 'meta-llama/Llama-3.1-70B-Instruct'
    "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF"
    'meta-llama/Llama-3.1-70B-Instruct'
    "meta-llama/Llama-3.2-11B-Vision-Instruct"
    accelerator = Accelerator()

    # Load model with automatic device mapping to distribute across CPU and GPU
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,  # Enable 8-bit quantization
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        # quantization_config=bnb_config,#rtx 4090
        device_map="auto"
    )
    # Prepare the model with accelerator
    base_model = accelerator.prepare(base_model)
    print(f"Base model loaded at {datetime.now().date()}")
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
    print(f"\nbase_model_name {base_model_name}"
          f"\ninput_text temperature {temperature}:"
          f"\n {input_text}\n "
          f"generated_text:\n {ft_generated_text}")



