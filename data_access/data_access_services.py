from configuration import configuration_services
from sqlalchemy import create_engine
from sqlalchemy import text
import pandas as pd

class Data_access_services:
    def __init__(self):
        self.db_config_str=self._get_db_config_str()
        self.sql_engine=create_engine(self.db_config_str)

  
    def _get_db_config_str(self):
        v_config_services=configuration_services.Config_services()
        param_dict=v_config_services.get_env_config_dict()['database']
        config_str=f'postgresql://{param_dict['user']}:{param_dict['password']}@{param_dict['host']}:{param_dict['port']}/{param_dict['dbname']}'
        return config_str
    
    def get_gites_raw_file(self,nb_maxi:int=-1,schema:str='points_enquete'):
        
        with self.sql_engine.connect()as conn:
            query_str=f'Select * from {schema}.visites_gites'
            if nb_maxi>0:
                query_str=f'{query_str} LIMIT {nb_maxi}'
            query=text(query_str)

            #utiliser directement pandas
            use_pd=True
            if use_pd:
                 df = pd.read_sql(text(query_str), conn)
            else: #Juste pour mémoire: usage par sqlachemy execute
                result=conn.execute(query)
                columns = result.keys()
                df = pd.DataFrame(result.fetchall(), columns=columns)
                
            return df

            

        
    
    

