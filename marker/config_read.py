import configparser


class Config:
    NODE_MYSQL = 'mysql'
    NODE_LLM = 'llm'
    NODE_VLM = 'vlm'
    NODE_SYS = 'system'
    NODE_SERVER = 'server'
    NODE_RABBITMQ = 'rabbitmq'

    def __init__(self, config_file):
        self.config = configparser.ConfigParser()
        self.config.read(config_file)
        self.config_file = config_file

    def get_config_node(self, node_name: str):
        if self.config.has_section(node_name):
            return self.config[node_name]
        return None

    def get_config_node_param(self, node_name: str, param_name: str, param_type: type = str):
        if self.config.has_section(node_name):
            if self.config.has_option(node_name, param_name):
                node = self.get_config_node(node_name)
                if param_type == int:
                    return node.getint(param_name, fallback=0)
                elif param_type == bool:
                    return node.getboolean(param_name, fallback=False)
                elif param_type == float:
                    return node.getfloat(param_name, fallback=0.0)
                else:
                    return node.get(param_name, fallback='')
        else:
            if param_type == int:
                return 0
            elif param_type == bool:
                return False
            elif param_type == float:
                return 0.0
            else:
                return None

    def get_mysql_param(self, param_name: str, param_type: type = str):
        return self.get_config_node_param(self.NODE_MYSQL, param_name, param_type)

    def get_llm_param(self, param_name: str, param_type: type = str):
        return self.get_config_node_param(self.NODE_LLM, param_name, param_type)

    def get_vlm_param(self, param_name: str, param_type: type = str):
        return self.get_config_node_param(self.NODE_VLM, param_name, param_type)

    def get_sys_param(self, param_name: str, param_type: type = str):
        return self.get_config_node_param(self.NODE_SYS, param_name, param_type)

    def get_server_param(self, param_name: str, param_type: type = str):
        return self.get_config_node_param(self.NODE_SERVER, param_name, param_type)

    def get_rabbitmq_param(self, param_name: str, param_type: type = str):
        return self.get_config_node_param(self.NODE_RABBITMQ, param_name, param_type)

    def is_dev_mode(self):
        return self.get_config_node_param(self.NODE_SYS, 'dev', bool)
