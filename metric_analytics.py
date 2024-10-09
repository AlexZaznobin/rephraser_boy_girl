import postgress_connection
from dotenv import load_dotenv, find_dotenv
import pandas as pd
if __name__ == '__main__':

    load_dotenv(find_dotenv())
    pgc = postgress_connection.get_connection(dbname="db3")
    result_df = postgress_connection.download_table_as_dataframe(pgc,table_name="boy_girl_results")
    average_bleu_by_temperature = result_df.groupby('temperature')['bleu_score'].mean()
    print(average_bleu_by_temperature.shape)