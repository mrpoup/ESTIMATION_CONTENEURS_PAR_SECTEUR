from path_access.path_access import Path_services
import json

class Config_services:
    def __init__(self):
        v_path_services=Path_services()
        self.configuration_dir_path_path_name=v_path_services.get_configuration_dir_path_path_name()
        self.config_env_file_name='config_env.json'
        #
        self.upload_env_config_file()

    def upload_env_config_file(self):
        file_path=self.configuration_dir_path_path_name/self.config_env_file_name

        if not file_path.exists():
            raise f'File {file_path} not found.'
        
        with file_path.open("r", encoding="utf-8") as f:
            self.env_config_dict = json.load(f)
       

    def get_env_config_dict(self):
        return self.env_config_dict
    
    