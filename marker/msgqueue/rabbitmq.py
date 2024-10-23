from pika.adapters.blocking_connection import BlockingConnection
from pika.connection import ConnectionParameters
from pika.credentials import PlainCredentials
from loguru import logger

from marker.config_read import Config
from marker.logger import set_logru

set_logru()

class RabbitMessageQueue:

    EXCHANGE_NAME = 'direct_ocr_type'
    QUEUE_NAME = 'queue_ocr_type'

    def __init__(self, config_file):
        self.config = Config(config_file)
        self.connection = None
        self.connect()
        self.msg_durable = self.config.get_rabbitmq_param('msg_durable', bool)

    def connect(self):
        if self.connection is None or not self.connection.is_open:
            mq_host = self.config.get_rabbitmq_param('host')
            if mq_host is None:
                log_info = f"RabbitMQ Host 不存在！"
                logger.error(log_info)
                return 0, None
            mq_port = self.config.get_rabbitmq_param('port', int)
            if mq_port is None:
                log_info = f"RabbitMQ Port 不存在！"
                logger.error(log_info)
                return 0, None
            mq_user = self.config.get_rabbitmq_param('user')
            mq_password = self.config.get_rabbitmq_param('password')
            if mq_user is None:
                mq_user = ConnectionParameters.DEFAULT_USERNAME
            if mq_password is None:
                mq_password = ConnectionParameters.DEFAULT_PASSWORD

            mq_credentials = PlainCredentials(username=mq_user, password=mq_password)
            self.connection = BlockingConnection(ConnectionParameters(host=mq_host, port=mq_port, credentials=mq_credentials))
        return 1, self.connection

    def close(self):
        if self.connection and self.connection.is_open():
            self.connection.close()

    @staticmethod
    def get_route_key(ocr_type, ocr_priority):
        return f'ocrtype_{ocr_type}_{ocr_priority}'

    def get_durable(self):
        return self.msg_durable


if __name__ == "__main__":
    mq = RabbitMessageQueue('')
