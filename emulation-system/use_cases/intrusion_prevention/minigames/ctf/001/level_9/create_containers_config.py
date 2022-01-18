import os
from csle_common.dao.container_config.containers_config import ContainersConfig
from csle_common.dao.container_config.node_container_config import NodeContainerConfig
from csle_common.envs_model.config.generator.container_generator import ContainerGenerator
from csle_common.util.experiments_util import util
import csle_common.constants.constants as constants


def default_containers_config(network_id: int = 9, level: str = "9", version: str = "0.0.1") -> ContainersConfig:
    """
    :param version: the version of the containers to use
    :param level: the level parameter of the emulation
    :param network_id: the network id
    :return: the ContainersConfig of the emulation
    """
    containers = [
        NodeContainerConfig(name=f"{constants.CONTAINER_IMAGES.CLIENT_1}",
                            internal_network=f"{constants.CSLE.CSLE_INTERNAL_NET_PREFIX}{network_id}",
                            minigame=constants.CSLE.CTF_MINIGAME,
                            version=version, level=level,
                            internal_ip=f"{constants.CSLE.CSLE_INTERNAL_SUBNETMASK_PREFIX}{network_id}.254",
                            restart_policy=constants.DOCKER.ON_FAILURE_3,
                            external_ip=f"{constants.CSLE.CSLE_EXTERNAL_SUBNETMASK_PREFIX}{network_id}.254",
                            external_network=f"{constants.CSLE.CSLE_EXTERNAL_NET_PREFIX}{network_id}",
                            connected_to_external_net=True,
                            connected_to_internal_net=False,
                            suffix="_1"),
        NodeContainerConfig(name=f"{constants.CONTAINER_IMAGES.CLIENT_1}",
                            internal_network=f"{constants.CSLE.CSLE_INTERNAL_NET_PREFIX}{network_id}",
                            minigame=constants.CSLE.CTF_MINIGAME,
                            version=version, level=level,
                            internal_ip=f"{constants.CSLE.CSLE_INTERNAL_SUBNETMASK_PREFIX}{network_id}.253",
                            restart_policy=constants.DOCKER.ON_FAILURE_3,
                            external_ip=f"{constants.CSLE.CSLE_EXTERNAL_SUBNETMASK_PREFIX}{network_id}.253",
                            external_network=f"{constants.CSLE.CSLE_EXTERNAL_NET_PREFIX}{network_id}",
                            connected_to_external_net=True,
                            connected_to_internal_net=False,
                            suffix="_2"),
        NodeContainerConfig(name=f"{constants.CONTAINER_IMAGES.CLIENT_1}",
                            internal_network=f"{constants.CSLE.CSLE_INTERNAL_NET_PREFIX}{network_id}",
                            minigame=constants.CSLE.CTF_MINIGAME,
                            version=version, level=level,
                            internal_ip=f"{constants.CSLE.CSLE_INTERNAL_SUBNETMASK_PREFIX}{network_id}.252",
                            restart_policy=constants.DOCKER.ON_FAILURE_3,
                            external_ip=f"{constants.CSLE.CSLE_EXTERNAL_SUBNETMASK_PREFIX}{network_id}.252",
                            external_network=f"{constants.CSLE.CSLE_EXTERNAL_NET_PREFIX}{network_id}",
                            connected_to_external_net=True,
                            connected_to_internal_net=False,
                            suffix="_3"),
        NodeContainerConfig(name=f"{constants.CONTAINER_IMAGES.FTP_1}",
                            internal_network=f"{constants.CSLE.CSLE_INTERNAL_NET_PREFIX}{network_id}",
                            minigame=constants.CSLE.CTF_MINIGAME,
                            version=version, level=level,
                            internal_ip=f"{constants.CSLE.CSLE_INTERNAL_SUBNETMASK_PREFIX}{network_id}.79",
                            restart_policy=constants.DOCKER.ON_FAILURE_3,
                            external_ip=f"{constants.CSLE.CSLE_EXTERNAL_SUBNETMASK_PREFIX}{network_id}.79",
                            external_network=f"{constants.CSLE.CSLE_EXTERNAL_NET_PREFIX}{network_id}",
                            connected_to_external_net=False,
                            connected_to_internal_net=True,
                            suffix="_1"),
        NodeContainerConfig(name=f"{constants.CONTAINER_IMAGES.HACKER_KALI_1}",
                            internal_network=f"{constants.CSLE.CSLE_INTERNAL_NET_PREFIX}{network_id}",
                            minigame=constants.CSLE.CTF_MINIGAME, version=version, level=level,
                            internal_ip=f"{constants.CSLE.CSLE_INTERNAL_SUBNETMASK_PREFIX}{network_id}.191",
                            restart_policy=constants.DOCKER.ON_FAILURE_3,
                            external_ip=f"{constants.CSLE.CSLE_EXTERNAL_SUBNETMASK_PREFIX}{network_id}.191",
                            external_network=f"{constants.CSLE.CSLE_EXTERNAL_NET_PREFIX}{network_id}",
                            connected_to_external_net=True,
                            connected_to_internal_net=False,
                            suffix="_1"),
        NodeContainerConfig(name=f"{constants.CONTAINER_IMAGES.HONEYPOT_1}",
                            internal_network=f"{constants.CSLE.CSLE_INTERNAL_NET_PREFIX}{network_id}",
                            minigame=constants.CSLE.CTF_MINIGAME,
                            version=version, level=level,
                            internal_ip=f"{constants.CSLE.CSLE_INTERNAL_SUBNETMASK_PREFIX}{network_id}.21",
                            restart_policy=constants.DOCKER.ON_FAILURE_3,
                            external_ip=f"{constants.CSLE.CSLE_EXTERNAL_SUBNETMASK_PREFIX}{network_id}.21",
                            external_network=f"{constants.CSLE.CSLE_EXTERNAL_NET_PREFIX}{network_id}",
                            connected_to_external_net=False,
                            connected_to_internal_net=True,
                            suffix="_1"),
        NodeContainerConfig(name=f"{constants.CONTAINER_IMAGES.ROUTER_2}",
                            internal_network=f"{constants.CSLE.CSLE_INTERNAL_NET_PREFIX}{network_id}",
                            minigame=constants.CSLE.CTF_MINIGAME,
                            version=version, level=level,
                            internal_ip=f"{constants.CSLE.CSLE_INTERNAL_SUBNETMASK_PREFIX}{network_id}.10",
                            restart_policy=constants.DOCKER.ON_FAILURE_3,
                            external_ip=f"{constants.CSLE.CSLE_EXTERNAL_SUBNETMASK_PREFIX}{network_id}.10",
                            external_network=f"{constants.CSLE.CSLE_EXTERNAL_NET_PREFIX}{network_id}",
                            connected_to_external_net=True,
                            connected_to_internal_net=True,
                            suffix="_1"),
        NodeContainerConfig(name=f"{constants.CONTAINER_IMAGES.SSH_1}",
                            internal_network=f"{constants.CSLE.CSLE_INTERNAL_NET_PREFIX}{network_id}",
                            minigame=constants.CSLE.CTF_MINIGAME,
                            version=version, level=level,
                            internal_ip=f"{constants.CSLE.CSLE_INTERNAL_SUBNETMASK_PREFIX}{network_id}.2",
                            restart_policy=constants.DOCKER.ON_FAILURE_3,
                            external_ip=f"{constants.CSLE.CSLE_EXTERNAL_SUBNETMASK_PREFIX}{network_id}.2",
                            external_network=f"{constants.CSLE.CSLE_EXTERNAL_NET_PREFIX}{network_id}",
                            connected_to_external_net=False,
                            connected_to_internal_net=True,
                            suffix="_1"),
        NodeContainerConfig(name=f"{constants.CONTAINER_IMAGES.SAMBA_2}",
                            internal_network=f"{constants.CSLE.CSLE_INTERNAL_NET_PREFIX}{network_id}",
                            minigame=constants.CSLE.CTF_MINIGAME,
                            version=version, level=level,
                            internal_ip=f"{constants.CSLE.CSLE_INTERNAL_SUBNETMASK_PREFIX}{network_id}.3",
                            restart_policy=constants.DOCKER.ON_FAILURE_3,
                            external_ip=f"{constants.CSLE.CSLE_EXTERNAL_SUBNETMASK_PREFIX}{network_id}.3",
                            external_network=f"{constants.CSLE.CSLE_EXTERNAL_NET_PREFIX}{network_id}",
                            connected_to_external_net=False,
                            connected_to_internal_net=True,
                            suffix="_1"),
        NodeContainerConfig(name=f"{constants.CONTAINER_IMAGES.SAMBA_1}",
                            internal_network=f"{constants.CSLE.CSLE_INTERNAL_NET_PREFIX}{network_id}",
                            minigame=constants.CSLE.CTF_MINIGAME,
                            version=version, level=level,
                            internal_ip=f"{constants.CSLE.CSLE_INTERNAL_SUBNETMASK_PREFIX}{network_id}.7",
                            restart_policy=constants.DOCKER.ON_FAILURE_3,
                            external_ip=f"{constants.CSLE.CSLE_EXTERNAL_SUBNETMASK_PREFIX}{network_id}.7",
                            external_network=f"{constants.CSLE.CSLE_EXTERNAL_NET_PREFIX}{network_id}",
                            connected_to_external_net=False,
                            connected_to_internal_net=True,
                            suffix="_1"),
        NodeContainerConfig(name=f"{constants.CONTAINER_IMAGES.HONEYPOT_2}",
                            internal_network=f"{constants.CSLE.CSLE_INTERNAL_NET_PREFIX}{network_id}",
                            minigame=constants.CSLE.CTF_MINIGAME,
                            version=version, level=level,
                            internal_ip=f"{constants.CSLE.CSLE_INTERNAL_SUBNETMASK_PREFIX}{network_id}.101",
                            restart_policy=constants.DOCKER.ON_FAILURE_3,
                            external_ip=f"{constants.CSLE.CSLE_EXTERNAL_SUBNETMASK_PREFIX}{network_id}.101",
                            external_network=f"{constants.CSLE.CSLE_EXTERNAL_NET_PREFIX}{network_id}",
                            connected_to_external_net=False,
                            connected_to_internal_net=True,
                            suffix="_1"),
        NodeContainerConfig(name=f"{constants.CONTAINER_IMAGES.SHELLSHOCK_1}",
                            internal_network=f"{constants.CSLE.CSLE_INTERNAL_NET_PREFIX}{network_id}",
                            minigame=constants.CSLE.CTF_MINIGAME,
                            version=version, level=level,
                            internal_ip=f"{constants.CSLE.CSLE_INTERNAL_SUBNETMASK_PREFIX}{network_id}.54",
                            restart_policy=constants.DOCKER.ON_FAILURE_3,
                            external_ip=f"{constants.CSLE.CSLE_EXTERNAL_SUBNETMASK_PREFIX}{network_id}.54",
                            external_network=f"{constants.CSLE.CSLE_EXTERNAL_NET_PREFIX}{network_id}",
                            connected_to_external_net=False,
                            connected_to_internal_net=True,
                            suffix="_1"),
        NodeContainerConfig(name=f"{constants.CONTAINER_IMAGES.SQL_INJECTION_1}",
                            internal_network=f"{constants.CSLE.CSLE_INTERNAL_NET_PREFIX}{network_id}",
                            minigame=constants.CSLE.CTF_MINIGAME, version=version, level=level,
                            internal_ip=f"{constants.CSLE.CSLE_INTERNAL_SUBNETMASK_PREFIX}{network_id}.74",
                            restart_policy=constants.DOCKER.ON_FAILURE_3,
                            external_ip=f"{constants.CSLE.CSLE_EXTERNAL_SUBNETMASK_PREFIX}{network_id}.74",
                            external_network=f"{constants.CSLE.CSLE_EXTERNAL_NET_PREFIX}{network_id}",
                            connected_to_external_net=False,
                            connected_to_internal_net=True,
                            suffix="_1"),
        NodeContainerConfig(name=f"{constants.CONTAINER_IMAGES.CVE_2010_0426_1}", internal_network=f"{constants.CSLE.CSLE_INTERNAL_NET_PREFIX}{network_id}",
                            minigame=constants.CSLE.CTF_MINIGAME, version=version, level=level,
                            internal_ip=f"{constants.CSLE.CSLE_INTERNAL_SUBNETMASK_PREFIX}{network_id}.61",
                            restart_policy=constants.DOCKER.ON_FAILURE_3,
                            external_ip=f"{constants.CSLE.CSLE_EXTERNAL_SUBNETMASK_PREFIX}{network_id}.61",
                            external_network=f"{constants.CSLE.CSLE_EXTERNAL_NET_PREFIX}{network_id}",
                            connected_to_external_net=False,
                            connected_to_internal_net=True,
                            suffix="_1"),
        NodeContainerConfig(name=f"{constants.CONTAINER_IMAGES.CVE_2015_1427_1}",
                            internal_network=f"{constants.CSLE.CSLE_INTERNAL_NET_PREFIX}{network_id}",
                            minigame=constants.CSLE.CTF_MINIGAME, version=version, level=level,
                            internal_ip=f"{constants.CSLE.CSLE_INTERNAL_SUBNETMASK_PREFIX}{network_id}.62",
                            restart_policy=constants.DOCKER.ON_FAILURE_3,
                            external_ip=f"{constants.CSLE.CSLE_EXTERNAL_SUBNETMASK_PREFIX}{network_id}.62",
                            external_network=f"{constants.CSLE.CSLE_EXTERNAL_NET_PREFIX}{network_id}",
                            connected_to_external_net=False,
                            connected_to_internal_net=True,
                            suffix="_1"),
        NodeContainerConfig(name=f"{constants.CONTAINER_IMAGES.HONEYPOT_1}",
                            internal_network=f"{constants.CSLE.CSLE_INTERNAL_NET_PREFIX}{network_id}",
                            minigame=constants.CSLE.CTF_MINIGAME,
                            version=version, level=level,
                            internal_ip=f"{constants.CSLE.CSLE_INTERNAL_SUBNETMASK_PREFIX}{network_id}.4",
                            restart_policy=constants.DOCKER.ON_FAILURE_3,
                            external_ip=f"{constants.CSLE.CSLE_EXTERNAL_SUBNETMASK_PREFIX}{network_id}.4",
                            external_network=f"{constants.CSLE.CSLE_EXTERNAL_NET_PREFIX}{network_id}",
                            connected_to_external_net=False,
                            connected_to_internal_net=True,
                            suffix="_2"),
        NodeContainerConfig(name=f"{constants.CONTAINER_IMAGES.HONEYPOT_1}",
                            internal_network=f"{constants.CSLE.CSLE_INTERNAL_NET_PREFIX}{network_id}",
                            minigame=constants.CSLE.CTF_MINIGAME,
                            version=version, level=level,
                            internal_ip=f"{constants.CSLE.CSLE_INTERNAL_SUBNETMASK_PREFIX}{network_id}.5",
                            restart_policy=constants.DOCKER.ON_FAILURE_3,
                            external_ip=f"{constants.CSLE.CSLE_EXTERNAL_SUBNETMASK_PREFIX}{network_id}.5",
                            external_network=f"{constants.CSLE.CSLE_EXTERNAL_NET_PREFIX}{network_id}",
                            connected_to_external_net=False,
                            connected_to_internal_net=True,
                            suffix="_3"),
        NodeContainerConfig(name=f"{constants.CONTAINER_IMAGES.HONEYPOT_1}",
                            internal_network=f"{constants.CSLE.CSLE_INTERNAL_NET_PREFIX}{network_id}",
                            minigame=constants.CSLE.CTF_MINIGAME,
                            version=version, level=level,
                            internal_ip=f"{constants.CSLE.CSLE_INTERNAL_SUBNETMASK_PREFIX}{network_id}.6",
                            restart_policy=constants.DOCKER.ON_FAILURE_3,
                            external_ip=f"{constants.CSLE.CSLE_EXTERNAL_SUBNETMASK_PREFIX}{network_id}.6",
                            external_network=f"{constants.CSLE.CSLE_EXTERNAL_NET_PREFIX}{network_id}",
                            connected_to_external_net=False,
                            connected_to_internal_net=True,
                            suffix="_4"),
        NodeContainerConfig(name=f"{constants.CONTAINER_IMAGES.HONEYPOT_1}",
                            internal_network=f"{constants.CSLE.CSLE_INTERNAL_NET_PREFIX}{network_id}",
                            minigame=constants.CSLE.CTF_MINIGAME,
                            version=version, level=level,
                            internal_ip=f"{constants.CSLE.CSLE_INTERNAL_SUBNETMASK_PREFIX}{network_id}.8",
                            restart_policy=constants.DOCKER.ON_FAILURE_3,
                            external_ip=f"{constants.CSLE.CSLE_EXTERNAL_SUBNETMASK_PREFIX}{network_id}.8",
                            external_network=f"{constants.CSLE.CSLE_EXTERNAL_NET_PREFIX}{network_id}",
                            connected_to_external_net=False,
                            connected_to_internal_net=True,
                            suffix="_5"),
        NodeContainerConfig(name=f"{constants.CONTAINER_IMAGES.HONEYPOT_1}",
                            internal_network=f"{constants.CSLE.CSLE_INTERNAL_NET_PREFIX}{network_id}",
                            minigame=constants.CSLE.CTF_MINIGAME,
                            version=version, level=level,
                            internal_ip=f"{constants.CSLE.CSLE_INTERNAL_SUBNETMASK_PREFIX}{network_id}.9",
                            restart_policy=constants.DOCKER.ON_FAILURE_3,
                            external_ip=f"{constants.CSLE.CSLE_EXTERNAL_SUBNETMASK_PREFIX}{network_id}.9",
                            external_network=f"{constants.CSLE.CSLE_EXTERNAL_NET_PREFIX}{network_id}",
                            connected_to_external_net=False,
                            connected_to_internal_net=True,
                            suffix="_6"),
        NodeContainerConfig(name=f"{constants.CONTAINER_IMAGES.CVE_2015_3306_1}",
                            internal_network=f"{constants.CSLE.CSLE_INTERNAL_NET_PREFIX}_3",
                            minigame=constants.CSLE.CTF_MINIGAME, version=version, level=level,
                            internal_ip=f"{constants.CSLE.CSLE_INTERNAL_SUBNETMASK_PREFIX}{network_id}.178",
                            restart_policy=constants.DOCKER.ON_FAILURE_3,
                            external_ip=f"{constants.CSLE.CSLE_EXTERNAL_SUBNETMASK_PREFIX}{network_id}.178",
                            external_network=f"{constants.CSLE.CSLE_EXTERNAL_NET_PREFIX}{network_id}",
                            connected_to_external_net=False,
                            connected_to_internal_net=True,
                            suffix="_1"),
        NodeContainerConfig(name=f"{constants.CONTAINER_IMAGES.HONEYPOT_2}",
                            internal_network=f"{constants.CSLE.CSLE_INTERNAL_NET_PREFIX}{network_id}",
                            minigame=constants.CSLE.CTF_MINIGAME,
                            version=version, level=level,
                            internal_ip=f"{constants.CSLE.CSLE_INTERNAL_SUBNETMASK_PREFIX}{network_id}.11",
                            restart_policy=constants.DOCKER.ON_FAILURE_3,
                            external_ip=f"{constants.CSLE.CSLE_EXTERNAL_SUBNETMASK_PREFIX}{network_id}.11",
                            external_network=f"{constants.CSLE.CSLE_EXTERNAL_NET_PREFIX}{network_id}",
                            connected_to_external_net=False,
                            connected_to_internal_net=True,
                            suffix="_2"),
        NodeContainerConfig(name=f"{constants.CONTAINER_IMAGES.HONEYPOT_2}",
                            internal_network=f"{constants.CSLE.CSLE_INTERNAL_NET_PREFIX}{network_id}",
                            minigame=constants.CSLE.CTF_MINIGAME,
                            version=version, level=level,
                            internal_ip=f"{constants.CSLE.CSLE_INTERNAL_SUBNETMASK_PREFIX}{network_id}.12",
                            restart_policy=constants.DOCKER.ON_FAILURE_3,
                            external_ip=f"{constants.CSLE.CSLE_EXTERNAL_SUBNETMASK_PREFIX}{network_id}.12",
                            external_network=f"{constants.CSLE.CSLE_EXTERNAL_NET_PREFIX}{network_id}",
                            connected_to_external_net=False,
                            connected_to_internal_net=True,
                            suffix="_3"),
        NodeContainerConfig(name=f"{constants.CONTAINER_IMAGES.HONEYPOT_2}",
                            internal_network=f"{constants.CSLE.CSLE_INTERNAL_NET_PREFIX}{network_id}",
                            minigame=constants.CSLE.CTF_MINIGAME,
                            version=version, level=level,
                            internal_ip=f"{constants.CSLE.CSLE_INTERNAL_SUBNETMASK_PREFIX}{network_id}.13",
                            restart_policy=constants.DOCKER.ON_FAILURE_3,
                            external_ip=f"{constants.CSLE.CSLE_EXTERNAL_SUBNETMASK_PREFIX}{network_id}.13",
                            external_network=f"{constants.CSLE.CSLE_EXTERNAL_NET_PREFIX}{network_id}",
                            connected_to_external_net=False,
                            connected_to_internal_net=True,
                            suffix="_4"),
        NodeContainerConfig(name=f"{constants.CONTAINER_IMAGES.HONEYPOT_2}",
                            internal_network=f"{constants.CSLE.CSLE_INTERNAL_NET_PREFIX}{network_id}",
                            minigame=constants.CSLE.CTF_MINIGAME,
                            version=version, level=level,
                            internal_ip=f"{constants.CSLE.CSLE_INTERNAL_SUBNETMASK_PREFIX}{network_id}.14",
                            restart_policy=constants.DOCKER.ON_FAILURE_3,
                            external_ip=f"{constants.CSLE.CSLE_EXTERNAL_SUBNETMASK_PREFIX}{network_id}.14",
                            external_network=f"{constants.CSLE.CSLE_EXTERNAL_NET_PREFIX}{network_id}",
                            connected_to_external_net=False,
                            connected_to_internal_net=True,
                            suffix="_5"),
        NodeContainerConfig(name=f"{constants.CONTAINER_IMAGES.HONEYPOT_2}",
                            internal_network=f"{constants.CSLE.CSLE_INTERNAL_NET_PREFIX}{network_id}",
                            minigame=constants.CSLE.CTF_MINIGAME,
                            version=version, level=level,
                            internal_ip=f"{constants.CSLE.CSLE_INTERNAL_SUBNETMASK_PREFIX}{network_id}.15",
                            restart_policy=constants.DOCKER.ON_FAILURE_3,
                            external_ip=f"{constants.CSLE.CSLE_EXTERNAL_SUBNETMASK_PREFIX}{network_id}.15",
                            external_network=f"{constants.CSLE.CSLE_EXTERNAL_NET_PREFIX}{network_id}",
                            connected_to_external_net=False,
                            connected_to_internal_net=True,
                            suffix="_6"),
        NodeContainerConfig(name=f"{constants.CONTAINER_IMAGES.HONEYPOT_2}",
                            internal_network=f"{constants.CSLE.CSLE_INTERNAL_NET_PREFIX}{network_id}",
                            minigame=constants.CSLE.CTF_MINIGAME,
                            version=version, level=level,
                            internal_ip=f"{constants.CSLE.CSLE_INTERNAL_SUBNETMASK_PREFIX}{network_id}.16",
                            restart_policy=constants.DOCKER.ON_FAILURE_3,
                            external_ip=f"{constants.CSLE.CSLE_EXTERNAL_SUBNETMASK_PREFIX}{network_id}.16",
                            external_network=f"{constants.CSLE.CSLE_EXTERNAL_NET_PREFIX}{network_id}",
                            connected_to_external_net=False,
                            connected_to_internal_net=True,
                            suffix="_7"),
        NodeContainerConfig(name=f"{constants.CONTAINER_IMAGES.HONEYPOT_2}",
                            internal_network=f"{constants.CSLE.CSLE_INTERNAL_NET_PREFIX}{network_id}",
                            minigame=constants.CSLE.CTF_MINIGAME,
                            version=version, level=level,
                            internal_ip=f"{constants.CSLE.CSLE_INTERNAL_SUBNETMASK_PREFIX}{network_id}.17",
                            restart_policy=constants.DOCKER.ON_FAILURE_3,
                            external_ip=f"{constants.CSLE.CSLE_EXTERNAL_SUBNETMASK_PREFIX}{network_id}.17",
                            external_network=f"{constants.CSLE.CSLE_EXTERNAL_NET_PREFIX}{network_id}",
                            connected_to_external_net=False,
                            connected_to_internal_net=True,
                            suffix="_8"),
        NodeContainerConfig(name=f"{constants.CONTAINER_IMAGES.HONEYPOT_2}",
                            internal_network=f"{constants.CSLE.CSLE_INTERNAL_NET_PREFIX}{network_id}",
                            minigame=constants.CSLE.CTF_MINIGAME,
                            version=version, level=level,
                            internal_ip=f"{constants.CSLE.CSLE_INTERNAL_SUBNETMASK_PREFIX}{network_id}.18",
                            restart_policy=constants.DOCKER.ON_FAILURE_3,
                            external_ip=f"{constants.CSLE.CSLE_EXTERNAL_SUBNETMASK_PREFIX}{network_id}.18",
                            external_network=f"{constants.CSLE.CSLE_EXTERNAL_NET_PREFIX}{network_id}",
                            connected_to_external_net=False,
                            connected_to_internal_net=True,
                            suffix="_9"),
        NodeContainerConfig(name=f"{constants.CONTAINER_IMAGES.HONEYPOT_2}",
                            internal_network=f"{constants.CSLE.CSLE_INTERNAL_NET_PREFIX}{network_id}",
                            minigame=constants.CSLE.CTF_MINIGAME,
                            version=version, level=level,
                            internal_ip=f"{constants.CSLE.CSLE_INTERNAL_SUBNETMASK_PREFIX}{network_id}.19",
                            restart_policy=constants.DOCKER.ON_FAILURE_3,
                            external_ip=f"{constants.CSLE.CSLE_EXTERNAL_SUBNETMASK_PREFIX}{network_id}.19",
                            external_network=f"{constants.CSLE.CSLE_EXTERNAL_NET_PREFIX}{network_id}",
                            connected_to_external_net=False,
                            connected_to_internal_net=True,
                            suffix="_10"),
        NodeContainerConfig(name=f"{constants.CONTAINER_IMAGES.HONEYPOT_2}",
                            internal_network=f"{constants.CSLE.CSLE_INTERNAL_NET_PREFIX}{network_id}",
                            minigame=constants.CSLE.CTF_MINIGAME,
                            version=version, level=level,
                            internal_ip=f"{constants.CSLE.CSLE_INTERNAL_SUBNETMASK_PREFIX}{network_id}.20",
                            restart_policy=constants.DOCKER.ON_FAILURE_3,
                            external_ip=f"{constants.CSLE.CSLE_EXTERNAL_SUBNETMASK_PREFIX}{network_id}.20",
                            external_network=f"{constants.CSLE.CSLE_EXTERNAL_NET_PREFIX}{network_id}",
                            connected_to_external_net=False,
                            connected_to_internal_net=True,
                            suffix="_11"),
        NodeContainerConfig(name=f"{constants.CONTAINER_IMAGES.HONEYPOT_2}",
                            internal_network=f"{constants.CSLE.CSLE_INTERNAL_NET_PREFIX}{network_id}",
                            minigame=constants.CSLE.CTF_MINIGAME,
                            version=version, level=level,
                            internal_ip=f"{constants.CSLE.CSLE_INTERNAL_SUBNETMASK_PREFIX}{network_id}.22",
                            restart_policy=constants.DOCKER.ON_FAILURE_3,
                            external_ip=f"{constants.CSLE.CSLE_EXTERNAL_SUBNETMASK_PREFIX}{network_id}.22",
                            external_network=f"{constants.CSLE.CSLE_EXTERNAL_NET_PREFIX}{network_id}",
                            connected_to_external_net=False,
                            connected_to_internal_net=True,
                            suffix="_12"),
        NodeContainerConfig(name=f"{constants.CONTAINER_IMAGES.HONEYPOT_2}",
                            internal_network=f"{constants.CSLE.CSLE_INTERNAL_NET_PREFIX}{network_id}",
                            minigame=constants.CSLE.CTF_MINIGAME,
                            version=version, level=level,
                            internal_ip=f"{constants.CSLE.CSLE_INTERNAL_SUBNETMASK_PREFIX}{network_id}.23",
                            restart_policy=constants.DOCKER.ON_FAILURE_3,
                            external_ip=f"{constants.CSLE.CSLE_EXTERNAL_SUBNETMASK_PREFIX}{network_id}.23",
                            external_network=f"{constants.CSLE.CSLE_EXTERNAL_NET_PREFIX}{network_id}",
                            connected_to_external_net=False,
                            connected_to_internal_net=True,
                            suffix="_13"),
        NodeContainerConfig(name=f"{constants.CONTAINER_IMAGES.HONEYPOT_2}",
                            internal_network=f"{constants.CSLE.CSLE_INTERNAL_NET_PREFIX}{network_id}",
                            minigame=constants.CSLE.CTF_MINIGAME,
                            version=version, level=level,
                            internal_ip=f"{constants.CSLE.CSLE_INTERNAL_SUBNETMASK_PREFIX}{network_id}.24",
                            restart_policy=constants.DOCKER.ON_FAILURE_3,
                            external_ip=f"{constants.CSLE.CSLE_EXTERNAL_SUBNETMASK_PREFIX}{network_id}.24",
                            external_network=f"{constants.CSLE.CSLE_EXTERNAL_NET_PREFIX}{network_id}",
                            connected_to_external_net=False,
                            connected_to_internal_net=True,
                            suffix="_14"),
        NodeContainerConfig(name=f"{constants.CONTAINER_IMAGES.CVE_2015_5602_1}",
                            internal_network=f"{constants.CSLE.CSLE_INTERNAL_NET_PREFIX}{network_id}",
                            minigame=constants.CSLE.CTF_MINIGAME, version=version, level=level,
                            internal_ip=f"{constants.CSLE.CSLE_INTERNAL_SUBNETMASK_PREFIX}{network_id}.25",
                            restart_policy=constants.DOCKER.ON_FAILURE_3,
                            external_ip=f"{constants.CSLE.CSLE_EXTERNAL_SUBNETMASK_PREFIX}{network_id}.25",
                            external_network=f"{constants.CSLE.CSLE_EXTERNAL_NET_PREFIX}{network_id}",
                            connected_to_external_net=False,
                            connected_to_internal_net=True,
                            suffix="_1"),
        NodeContainerConfig(name=f"{constants.CONTAINER_IMAGES.CVE_2016_10033_1}",
                            internal_network=f"{constants.CSLE.CSLE_INTERNAL_NET_PREFIX}{network_id}",
                            minigame=constants.CSLE.CTF_MINIGAME, version=version, level=level,
                            internal_ip=f"{constants.CSLE.CSLE_INTERNAL_SUBNETMASK_PREFIX}{network_id}.28",
                            restart_policy=constants.DOCKER.ON_FAILURE_3,
                            external_ip=f"{constants.CSLE.CSLE_EXTERNAL_SUBNETMASK_PREFIX}{network_id}.28",
                            external_network=f"{constants.CSLE.CSLE_EXTERNAL_NET_PREFIX}{network_id}",
                            connected_to_external_net=False,
                            connected_to_internal_net=True,
                            suffix="_1")
    ]
    containers_cfg = ContainersConfig(
        containers=containers, internal_network=f"{constants.CSLE.CSLE_INTERNAL_NET_PREFIX}{network_id}",
        agent_ip=f"{constants.CSLE.CSLE_INTERNAL_SUBNETMASK_PREFIX}{network_id}.191",
        router_ip=f"{constants.CSLE.CSLE_INTERNAL_SUBNETMASK_PREFIX}{network_id}.10",
        internal_subnet_mask=f"{constants.CSLE.CSLE_INTERNAL_SUBNETMASK_PREFIX}{network_id}{constants.CSLE.CSLE_SUBNETMASK}",
        internal_subnet_prefix=f"{constants.CSLE.CSLE_INTERNAL_SUBNETMASK_PREFIX}{network_id}.",
        ids_enabled=True,
        external_network=f"{constants.CSLE.CSLE_EXTERNAL_NET_PREFIX}{network_id}",
        external_subnet_mask=f"{constants.CSLE.CSLE_EXTERNAL_SUBNETMASK_PREFIX}{network_id}{constants.CSLE.CSLE_SUBNETMASK}",
        external_subnet_prefix=f"{constants.CSLE.CSLE_EXTERNAL_SUBNETMASK_PREFIX}{network_id}.")
    return containers_cfg


# Generates the containers.json configuration file
if __name__ == '__main__':
    network_id = 9
    level = "9"
    version = "0.0.1"
    if os.path.exists(util.default_containers_path(out_dir=util.default_output_dir())):
        os.remove(util.default_containers_path(out_dir=util.default_output_dir()))
    containers_cfg = default_containers_config(network_id=network_id, level=level, version=version)
    ContainerGenerator.write_containers_config(containers_cfg, path=util.default_output_dir())
