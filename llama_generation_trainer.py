import postgress_connection
from typing import List
import torch
import transformers
from datasets import Dataset
import s3_backet_saver
import pandas as pd
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    PrefixTuningConfig,
    TaskType
)
from peft import prepare_model_for_kbit_training
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments
from transformers import DataCollatorWithPadding
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import boto3
import shutil
from trl import SFTTrainer
from dotenv import load_dotenv, find_dotenv
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
import os
from  data_generator import create_llama2_prompt as create_llama2_prompt
def make_config():

    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=256,
        lora_alpha=32,
        lora_dropout=0.01,
        bias="none",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )

# Шаг: Применение токенизации
def formatting_func(examples, tokenizer, max_length):
    # Токенизация оригинала
    inputs = tokenizer(
        examples['original'],
        padding='max_length',  # Заполнение до max_length
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    # Токенизация перефраза
    outputs = tokenizer(
        examples['paraphrase'],
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    # Возвращаем батчи данных
    return {
        'input_ids': inputs['input_ids'].tolist(),  # Преобразуем в список для всей батчи
        'attention_mask': inputs['attention_mask'].tolist(),  # Преобразуем в список
        'labels': outputs['input_ids'].tolist()  # Метки также в списке
    }

def prepare_datasets(tokenizer, traindf, max_length=512):
    # Load tokenizer


    # Split the dataframe into training and evaluation sets
    train_df, eval_df = train_test_split(traindf, test_size=0.2)

    # Convert pandas dataframes into Hugging Face datasets
    train_dataset = Dataset.from_pandas(train_df )
    eval_dataset = Dataset.from_pandas(eval_df )

    # Remove unnecessary columns
    train_dataset = train_dataset.remove_columns(['id', 'complexity', 'changes', 'cost', 'date', 'model'])
    eval_dataset = eval_dataset.remove_columns(['id', 'complexity', 'changes', 'cost', 'date', 'model'])

    # # Apply tokenization
    train_dataset = train_dataset.map(lambda x: formatting_func_chat(x, tokenizer, max_length), batched=True)
    eval_dataset = eval_dataset.map(lambda x: formatting_func_chat(x, tokenizer, max_length), batched=True)

    return train_dataset, eval_dataset


def formatting_func_chat (examples, tokenizer, max_length) :
        inputs = tokenizer(
            examples['text_with_prompt'],
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )

        inputs['labels'] = inputs['input_ids'].clone()

        return inputs
if __name__ == '__main__':
    EPOCHS=2
    LEARNING_RATE=1e-5
    pgc = postgress_connection.get_connection(dbname="db3")
    load_dotenv(find_dotenv())
    data_df = postgress_connection.download_table_as_dataframe(pgc)
    data_df['text_with_prompt'] = data_df.apply(create_llama2_prompt, axis=1)
    traindf = data_df[data_df['model'] != 'gemini-1.5-pro']
    print('traindf:',traindf.info())
    "meta-llama/Llama-2-7b-hf"
    base_model_name =  os.environ.get("MODEL_NAME")
    output_dir = base_model_name
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    train_dataset, eval_dataset =prepare_datasets(tokenizer, traindf)

    adapter_patch=f"{base_model_name}_{traindf.shape[0]}_{EPOCHS}_lr{LEARNING_RATE}_1"
    print("\n\nadapter_patch:\n\n",adapter_patch,"\n\n")
    bnb_config = BitsAndBytesConfig( #rtx 4090
        load_in_4bit=True,  # Enable 4-bit quantization
        bnb_4bit_quant_type='nf4',  # Use NF4 quantization type
        bnb_4bit_compute_dtype=torch.float16,  # Use FP16 for computation on GPU
        bnb_4bit_use_double_quant=True,  # Optional: use nested quantization
    )

    # Explicitly set the device to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device",device)

    # Load model with automatic device mapping to distribute across CPU and GPU
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,#rtx 4090
        device_map="auto"
    )
    config=make_config()
    data_collator = DataCollatorWithPadding(tokenizer= tokenizer)

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",  # Evaluate at the end of each epoch
        save_strategy="epoch",  # Save at the end of each epoch
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        weight_decay=0.01,
        learning_rate=LEARNING_RATE,
        save_total_limit=1,
        num_train_epochs=EPOCHS,
        logging_dir="./logs",
        logging_steps=10,
        load_best_model_at_end=True,  # Ensure that the best model is loaded at the end
        fp16=True,  # T4
        # fp16=False,  #rtx 4090
    )

    # Step 8: Initialize SFTTrainer
    trainer = SFTTrainer(
        model=base_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        max_seq_length=None,
        packing=False,
        data_collator=data_collator,
        # formatting_func=formatting_func,
        dataset_text_field="text_with_prompt",
        peft_config=config,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    trainer.train()
    metrics = trainer.evaluate(eval_dataset)

    # Example usage
    local_model_directory =trainer.state.best_model_checkpoint # Directory where the model is saved locally
    s3_bucket_name = os.environ.get("S3_BUCKET_NAME") # Replace with your S3 bucket name
    s3_model_directory = adapter_patch # Replace with the desired S3 path

    # Call the function to upload the model files
    s3_backet_saver.upload_model_to_s3(local_model_directory, s3_bucket_name, s3_model_directory)