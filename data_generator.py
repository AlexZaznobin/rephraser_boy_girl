from google.cloud import billing_v1
from google.oauth2 import service_account
import pandas as pd
import os
import  itertools
from datetime import datetime
import vertexai
from vertexai.generative_models import GenerativeModel
import json
import os
import re
import pandas as pd

def get_prices():
    billing_client = billing_v1.CloudCatalogClient()

    # Initialize the nested dictionary


    g_models = ["Gemini 1.5 Pro", "Gemini 1.5 Flash"]
    token_lengths = ["longer than 128k tokens", "up to 128k tokens"]
    types = ["input", "output"]

    prices = {g_model : {token_length : {type : None for type in types} for token_length in token_lengths} for g_model in
              g_models}
    # Generate the Cartesian product
    conditions = list(itertools.product(g_models, token_lengths, types))

    billing_client = billing_v1.CloudCatalogClient()
    services = billing_client.list_services()
    for service in services :
        if "Gemini" in service.display_name:
            skus = billing_client.list_skus(parent=service.name)
            for sku in skus :
                for product, token_length, type in conditions :
                    if (product in sku.description) and (token_length in sku.description or  product=="Gemini Pro") and (type in sku.description.lower()) :
                        prices[product][token_length][type] = sku.pricing_info[0].pricing_expression.tiered_rates[0].unit_price.nanos / 1e9
    print(f"get prices {prices}")
    return  prices

class GeminiAccountant:
    prices = None  # Class attribute for prices

    @classmethod
    def initialize_prices (cls) :
        if cls.prices is None :
            cls.prices = get_prices()

    def __init__(self):
        self.initialize_prices()
        self.g_models = ["Gemini 1.5 Pro", "Gemini Pro", "Gemini 1.5 Flash"]
        self.token_lengths  = ["longer than 128k tokens", "up to 128k tokens"]
        self.types = ["input", "output"]
        self.model_dict = { "gemini-1.5-pro":"Gemini 1.5 Pro" ,
                            "gemini-1.5-flash":"Gemini 1.5 Flash",
                            "gemini-1.0-pro":"Gemini Pro"}

    def calculate_amount_gemini (self, input_tokens, output_tokens ,g_model="Gemini 1.5 Pro" ) :
        if self.model_dict[g_model] is not None:
            g_model=self.model_dict[g_model]
        if input_tokens <= 128000 :
            input_cost = input_tokens * self.prices[g_model]["up to 128k tokens"]["input"]
        else :
            input_cost = 128000 * self.prices[g_model]["up to 128k tokens"]["input"] + (
                        input_tokens - 128000) * self.prices[g_model]["longer than 128k tokens"]["input"]

        if output_tokens <= 128000 :
            output_cost = output_tokens * self.prices[g_model]["up to 128k tokens"]["output"]
        else :
            output_cost = 128000 * self.prices[g_model]["up to 128k tokens"]["output"] + (
                        output_tokens - 128000) * self.prices[g_model]["longer than 128k tokens"]["output"]

        total_cost = input_cost + output_cost
        return total_cost

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] ='gemini_synt_data_set.json'


def get_project_id(full_path):
    # Extract project_id from the credentials file
    with open(full_path, 'r') as f :
        credentials = json.load(f)
        return credentials.get('project_id')
class Gemini:
    def __init__(self, model_name="gemini-1.5-pro",temperature=0):
        FULL_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        project_id=get_project_id(FULL_PATH)
        vertexai.init(project=project_id, location="us-central1")
        self.model=GenerativeModel(model_name)
        self.model_name=model_name
        self.temperature=temperature
        self.generation_config= {
                "temperature" : self.temperature,
                'response_mime_type' : 'application/json'
            }
        self.accountant = GeminiAccountant()
    def _process_gemini_json_output (self , gemini_excerpts_str) :
        gemini_excerpts_str_1 = gemini_excerpts_str.replace('\n', '')
        gemini_excerpts_str_1 = gemini_excerpts_str_1.replace('json', '')
        gemini_excerpts_str_1 = gemini_excerpts_str_1.replace('```', '')
        return  gemini_excerpts_str_1

    def _save_to_postgress(self, postgres_connection ,result_df):
      cur =  postgres_connection.cursor()
      insert_query = f"""
                INSERT INTO {table_name} (original, paraphrase, complexity, changes, date , cost, model)
                VALUES (%s, %s, %s, %s, %s, %s, %s )
                """

      for index, row in result_df.iterrows() :
            cur.execute(insert_query, (
                row['original'],
                row['paraphrase'],
                row['complexity'],
                row['changes'],
                row['date'],
                row['cost'],
                row['model_name']
                ))
      result= postgres_connection.commit()
      cur.close()
      return result

    def infrence(self, request, postgres_connection=None ):
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        response = self.model.generate_content(request,generation_config=self.generation_config)
        text = response.candidates[0].content.parts[0].text
        text_cleaned = self._process_gemini_json_output(text)
        try :
            result_json = json.loads(text_cleaned)
            result_df=pd.DataFrame(data=result_json)
            result_df['date']=date

        except :
            print(f'json.loads does not work')

        cost=self.accountant.calculate_amount_gemini(input_tokens=response.usage_metadata.prompt_token_count,
                                                output_tokens=response.usage_metadata.candidates_token_count,
                                                g_model=self.model_name)
        result_df['cost']=cost
        result_df['model_name']=self.model_name
        return result_df


def create_llama2_prompt(row):
    prompt = f"""[INST]  Измени род в предложении с мужского на женский на русском языке :  {row['original']}[/INST] {row['paraphrase']} """
    return prompt

def create_llama2_request(text):
    prompt = f"""[INST] Измени род в предложении с мужского на женский на русском языке :  {text} [/INST]"""
    return prompt

if __name__ == '__main__':
    gemini_1_5 = Gemini(model_name="gemini-1.5-pro")
    gemini_flash = Gemini(model_name="gemini-1.5-flash")
    # Define the path to your txt file
    file_path = 'data_generation_prompt.txt'
    postgres_connection=get_connection(dbname="db3")
    # Open and read the content of the file
    with open(file_path, 'r', encoding='utf-8') as file :
        data_set_prompt = file.read()
    # for i in range(10) :
#     result_df = gemini_1_5.infrence(request=data_set_prompt,postgres_connection=postgres_connection)
#     postgres_connection.rollback()
#     gemini_1_5._save_to_postgress(postgres_connection ,result_df)


    # for i in range(100):
    #     result_df = gemini_flash.infrence(request=data_set_prompt,postgres_connection=postgres_connection)
    #     postgres_connection.rollback()
    #     gemini_flash._save_to_postgress(postgres_connection ,result_df)