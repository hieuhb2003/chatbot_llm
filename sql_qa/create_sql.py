import pandas as pd
import sqlite3
from config import Config

class SQLCreator:
    def __init__(self) -> None:
        pass
    
    def create_sql(self):
        df = pd.read_csv(Config.csv_dir)
        df.columns = df.columns.str.strip()
        
        if 'LINK_SP' in df.columns:
            df = df.drop(columns=['LINK_SP'])
        
        connection = sqlite3.connect(Config.db_dir)
        df.to_sql('data_items', connection, if_exists='replace',index=False)
        connection.close()

creator = SQLCreator()
creator.create_sql()
