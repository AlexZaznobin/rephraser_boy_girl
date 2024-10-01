import postgress_connection
from typing import List
import torch
import transformers
from datasets import Dataset
import pandas as pd
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    PrefixTuningConfig,
    TaskType
)
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import boto3
import shutil
from trl import SFTTrainer
from dotenv import load_dotenv, find_dotenv
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    load_dotenv(find_dotenv())
    data_df = postgress_connection.download_table_as_dataframe(postgres_connection)
    traindf = data_df[data_df['model'] != 'gemini-1.5-pro']
    print('traindf:',traindf.info())
    base_model_name = "meta-llama/Llama-2-7b-hf"
    output_dir = base_model_name

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
        device_map="auto",  # Automatically map layers to devices
    )

    micro_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    val_set_size: int = 200,

    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
    ]

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=128,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj",
            "v_proj",
        ],
    )

    model = get_peft_model(base_model, config)

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    train_df, eval_df = train_test_split(traindf, test_size=0.2)

    train_dataset = Dataset.from_pandas(train_df)
    eval_dataset = Dataset.from_pandas(eval_df)

    max_length = 512  # Выбранное максимальное количество токенов

    # Шаг: удаление ненужных колонок
    train_dataset = train_dataset.remove_columns(['id', 'complexity', 'changes', 'cost', 'date', 'model'])
    eval_dataset = eval_dataset.remove_columns(['id', 'complexity', 'changes', 'cost', 'date', 'model'])


    # Шаг: Применение токенизации
    def formatting_func (examples) :
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
            'input_ids' : inputs['input_ids'].tolist(),  # Преобразуем в список для всей батчи
            'attention_mask' : inputs['attention_mask'].tolist(),  # Преобразуем в список
            'labels' : outputs['input_ids'].tolist()  # Метки также в списке
        }


    # Применяем токенизацию
    train_dataset = train_dataset.map(formatting_func, batched=True)
    eval_dataset = eval_dataset.map(formatting_func, batched=True)
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",  # Evaluate at the end of each epoch
        save_strategy="epoch",  # Save at the end of each epoch
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        weight_decay=0.01,
        save_total_limit=1,
        num_train_epochs=10,
        logging_dir="./logs",
        logging_steps=10,
        load_best_model_at_end=True,  # Ensure that the best model is loaded at the end
        fp16=True,  # Enable mixed precision training
    )

    # Step 8: Initialize SFTTrainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=subset_dataset,
        eval_dataset=subset_dataset_2,
        tokenizer=tokenizer,
        formatting_func=formatting_func,
        packing=False,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    trainer.train()
    metrics = trainer.evaluate(eval_dataset)
    print(metrics)
    trainer.save_model("./trained_model_10")
    tokenizer.save_pretrained("./trained_model_10")

    # Step 2: Prepare the input text
    input_text = "Известный писатель опубликовал новый роман."
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    inputs
    outputs = model.generate(
        **inputs,
        max_length=100,
        num_return_sequences=1,
        temperature=0.1,
        num_beams=7,
        repetition_penalty=2.5,
        length_penalty=1,  # Балансирует длину
        early_stopping=True  # Остановка после получения лучшего результата
    )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(text)