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



def test_results(ft_model,tokenizer, device,temperature , input_text_list):
    output_list=[]
    for input_text in input_text_list:
        input_text=data_generator.create_llama2_request(input_text)
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        temperature =temperature
        ft_outputs = ft_model.generate(
            **inputs,
            max_length=100,  #
            temperature=temperature,
        )
        ft_generated_text = tokenizer.decode(ft_outputs[0], skip_special_tokens=True)
        print(f"\ninput_text temperature {temperature}:\n {input_text}\n generated_text:\n {ft_generated_text}")
        output_list.append(ft_generated_text)
    return output_list


# Define the function
def extract_phrase_after_inst (text) :
    # Find the index of [/INST] in the text
    inst_end_tag = "[/INST]"
    index = text.find(inst_end_tag)

    # If the [/INST] tag is found, return the text after it
    if index != -1 :
        return text[index + len(inst_end_tag) :].strip()
    else :
        return None


def calculate_bleu (reference, hypothesis) :
    """
    Calculate BLEU score using sacrebleu for Russian texts.
    :param reference: str, the reference sentence
    :param hypothesis: str, the generated sentence (hypothesis)
    :return: float, the BLEU score
    """
    # sacrebleu requires references to be a list of lists
    references = [reference]
    # Calculate BLEU score
    bleu = sacrebleu.raw_corpus_bleu([hypothesis], [references])
    return bleu.score

if __name__ == '__main__':
    load_dotenv(find_dotenv())
    base_model_name =  os.environ.get("MODEL_NAME")
    s3_bucket_name = os.environ.get("S3_BUCKET_NAME")  # Replace with your S3 bucket name
    s3_model_directory = '-2-7b-chat-hf_1000_2_lr1e-05_10'
    dir='adapter_patch'
    s3_backet_saver.download_model_from_s3(s3_bucket_name, s3_model_directory, dir)
    current_directory = os.getcwd()
    temp_dir = f'{current_directory}/{dir}'
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,  # Enable 4-bit quantization
        bnb_4bit_quant_type='nf4',  # Use NF4 quantization type
        bnb_4bit_compute_dtype=torch.float16,  # Use FP16 for computation on GPU
        bnb_4bit_use_double_quant=True,  # Optional: use nested quantization
    )
    # Load model with automatic device mapping to distribute across CPU and GPU
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )
    base_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ft_model = PeftModel.from_pretrained(base_model, temp_dir)
    ft_model.eval()
    pgc = postgress_connection.get_connection(dbname="db3")
    load_dotenv(find_dotenv())
    data_df = postgress_connection.download_table_as_dataframe(pgc)
    test_df = data_df[data_df['model'] == 'gemini-1.5-pro']
    result_df=pd.DataFrame()
    pgc = postgress_connection.get_connection(dbname="db3")
    for inf_temperature in np.arange(0.1, 0.9, 0.1) :
        test_df['temperature'] = None
        test_df['inference'] = None
        test_df['inference_result'] = None
        test_df['bleu_score'] = None
        test_df['model_version'] = None
        test_df['prompt_version'] = None
        output_text_list = test_results(ft_model, tokenizer, device, inf_temperature, test_df["original"].values)
        test_df['inference'] = output_text_list
        test_df['inference_result'] = test_df['inference'].apply(extract_phrase_after_inst)
        test_df['temperature'] = test_df['temperature'].apply(lambda x : inf_temperature if x is None else x)
        test_df['model_version'] = test_df['model_version'].apply(lambda x : f"{base_model_name}_{temp_dir}" if x is None else x)
        test_df['prompt_version'] = test_df['prompt_version'].apply(
            lambda x : data_generator.create_llama2_request("") if x is None else x)
        test_df['bleu_score'] = test_df.apply(
            lambda row : calculate_bleu(row['paraphrase'], row['inference_result']),
            axis=1)
        postgress_connection.upload_dataframe_to_postgres(test_df, pgc, table_name="boy_girl_results")
        result_df=pd.concat([result_df,test_df])



