from pathlib import Path


class Path_services:
    def __init__(self):
        self.application_dir_path_name=Path(__file__).resolve().parent.parent
        self._configuration_dirname='configuration'

    def get_application_dir_path_name(self):
        return self.application_dir_path_name
    

    def get_configuration_dir_path_path_name(self):
        afn=self.application_dir_path_name
        f=self.get_application_dir_path_name()/self._configuration_dirname
        return f
    
    
    

