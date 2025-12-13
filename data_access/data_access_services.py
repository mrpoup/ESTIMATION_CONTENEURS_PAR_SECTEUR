from configuration import configuration_services
from sqlalchemy import create_engine
from sqlalchemy import text
import pandas as pd
import geopandas as gpd


class Data_access_services:
    def __init__(self):
        self.db_config_str=self._get_db_config_str()
        self.sql_engine=create_engine(self.db_config_str)
        self._cache = {}   # cache interne

  
    def set_clear_cache(self):
        self._cache.clear()

    def _get_db_config_str(self):
        v_config_services=configuration_services.Config_services()
        param_dict=v_config_services.get_env_config_dict()['database']
        config_str=f'postgresql://{param_dict['user']}:{param_dict['password']}@{param_dict['host']}:{param_dict['port']}/{param_dict['dbname']}'
        return config_str
    
    def get_gites_raw_file(self,nb_maxi:int=-1,schema:str='points_enquete'):
        
        key = (schema, "visites_gites", nb_maxi)
        # --- 1. Si existe dans le cache → on le renvoie immédiatement ---
        if key in self._cache:
            return self._cache[key]
        
        # --- 2. Sinon chargement ---
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
            
            # --- 3. On met en cache ---
            self._cache[key] = df
                
            return df

    def export_gdf_to_postgis(self, gdf:gpd.GeoDataFrame,tablename:str,schema:str='points_enquete' ):
        gdf.to_postgis(name=tablename, con=self.sql_engine,schema=schema, if_exists='replace',index=False)
            

    def get_data_table_to_gdf(self,tablename:str,col_geometry='geometry', nb_maxi:int=-1,schema:str='points_enquete',crs="EPSG:2154"):
       
        print(f'Load table {schema}.{tablename} from db')
        key = (schema, tablename, nb_maxi)
        # --- 1. Si existe dans le cache → on le renvoie immédiatement ---
        if key in self._cache:
            return self._cache[key]
        
        # --- 2. Sinon chargement ---
        with self.sql_engine.connect()as conn:
            query_str=f'Select * from {schema}.{tablename}'
            #
            if nb_maxi>0:
                query_str=f'{query_str} LIMIT {nb_maxi}'
            #
            df = pd.read_sql(text(query_str), conn)   
            
               # --- conversion WKB -> shapely geometry ---
            gdf = gpd.GeoDataFrame(
                df,
                geometry=gpd.GeoSeries.from_wkb(df[col_geometry]),
                crs=crs 
            )
            # --- 3. On met en cache ---
            self._cache[key] = gdf
            return gdf
                
    
    

