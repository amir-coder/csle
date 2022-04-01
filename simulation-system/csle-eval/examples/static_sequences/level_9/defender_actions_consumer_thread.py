import threading
import time
from confluent_kafka import Consumer, KafkaError, KafkaException
from csle_common.dao.emulation_config.emulation_env_state import EmulationEnvState
from csle_common.metastore.metastore_facade import MetastoreFacade
from csle_common.logging.log import Logger
from csle_common.dao.emulation_action.defender.emulation_defender_action import EmulationDefenderAction
import csle_collector.constants.constants as collector_constants


class DefenderActionsConsumerThread(threading.Thread):
    """
    Thread that polls the kafka to get the latest defender actions
    """

    def __init__(self, kafka_server_ip: str, kafka_port: int, s: EmulationEnvState) -> None:
        """
        Initializes the thread

        :param kafka_server_ip: the ip of the kafka server
        :param kafka_port: the port of the kafka server
        :param s: the state of the emulation (the thread will update this state)
        """
        threading.Thread.__init__(self)
        self.running =True
        self.kafka_server_ip = kafka_server_ip
        self.kafka_port = kafka_port
        self.s = s
        self.ts = time.time()
        self.kafka_conf = {'bootstrap.servers': f"{self.kafka_server_ip}:{self.kafka_port}",
                           'group.id':  f"defender_actions_consumer_thread_{self.ts}",
                           'auto.offset.reset': 'latest'}
        self.consumer = Consumer(**self.kafka_conf)
        self.consumer.subscribe([collector_constants.LOG_SINK.DEFENDER_ACTIONS_TOPIC_NAME])
        self.defender_actions = []

    def run(self) -> None:
        """
        Runs the thread

        :return: None
        """
        while True:
            msg = self.consumer.poll(timeout=5.0)
            if msg is not None:
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        Logger.__call__().get_logger.warning(
                            f"reached end of partition: {msg.topic(), msg.partition(), msg.offset()}")
                    elif msg.error():
                        raise KafkaException(msg.error())
                else:
                    self.defender_actions.append(EmulationDefenderAction.from_kafka_record(
                        record=msg.value().decode()))
                    print(self.defender_actions)


if __name__ == '__main__':
    emulation_env_config = MetastoreFacade.get_emulation("csle-level9-001")
    s = EmulationEnvState(emulation_env_config=emulation_env_config)
    thread = DefenderActionsConsumerThread(
        kafka_server_ip=emulation_env_config.log_sink_config.container.get_ips()[0],
        kafka_port=emulation_env_config.log_sink_config.kafka_port, s=s)
    thread.start()
    thread.join()