from configuration import configuration_services

class Data_access_config_services:
    def __init__(self):
        self.param_dict=self.get_db_config_param()

    def get_db_config_param(self):
        v_config_services=configuration_services.Config_services()
        param_dict=v_config_services.get_env_config_dict()['database']
        return param_dict
    
    def get_db_config_str(self):
        param_dict=self.param_dict
        config_str=f'postgresql://{param_dict['user']}:{param_dict['password']}@{param_dict['host']}:{param_dict['port']}/{param_dict['dbname']}'
        return config_str

