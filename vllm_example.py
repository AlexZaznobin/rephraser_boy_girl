from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from datetime import datetime

if __name__ == '__main__':
    model_name='Qwen/Qwen2.5-Coder-32B-Instruct-AWQ'
    model_name='neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8'
    model_name='Qwen/Qwen2.5-72B-Instruct-AWQ'
    model_name='Qwen/Qwen2.5-Coder-32B-Instruct'
    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Pass the default decoding hyperparameters of Qwen2.5-7B-Instruct
    # max_tokens is for the maximum length for generation.
    sampling_params = SamplingParams(temperature=0.1, top_p=0.8, repetition_penalty=1.05, max_tokens=512)

    # Input the model name or path. Can be GPTQ or AWQ models.
    llm = LLM(model=model_name, max_model_len=4096)
    prompt = """user: добрый день. подскажите, пожалуйста, где нам найти справку- бухгалтерский баланс нашей организации?
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

    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    print(f"\n\n\nBase model loaded at {datetime.now()}")
    # generate outputs
    outputs = llm.generate([text], sampling_params)
    print(f"\n\n\noutputs {datetime.now()}")

    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")