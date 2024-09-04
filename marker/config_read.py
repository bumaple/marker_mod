import configparser


class Config:

    NODE_MYSQL = 'mysql'
    NODE_LLM = 'llm'
    NODE_VLM = 'vlm'
    NODE_SYS = 'system'

    def __init__(self, config_file):
        self.config = configparser.ConfigParser()
        self.config.read(config_file)

    def get_config_node(self, node_name):
        if node_name in self.config:
            return self.config[node_name]
        return None

    def get_config_node_param(self, node_name, param_name):
        if node_name in self.config:
            if param_name in self.config[node_name]:
                return self.config[node_name][param_name]
        return None

    def get_mysql_param(self, param_name):
        return self.get_config_node_param(self.NODE_MYSQL, param_name)

    def get_llm_param(self, param_name):
        return self.get_config_node_param(self.NODE_LLM, param_name)

    def get_vlm_param(self, param_name):
        return self.get_config_node_param(self.NODE_VLM, param_name)

    def get_sys_param(self, param_name):
        return self.get_config_node_param(self.NODE_SYS, param_name)

    def is_dev_mode(self):
        dev = self.get_sys_param('dev')
        if dev is None:
            return False
        else:
            return bool(eval(dev))
