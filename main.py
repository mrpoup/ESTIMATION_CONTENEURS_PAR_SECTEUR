from data_access.data_access_services import Data_access_config_services

v_data_acess_config_services=Data_access_config_services()

v_db_config_str=v_data_acess_config_services.get_db_config_str()
print(v_db_config_str)







