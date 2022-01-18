from typing import List
from csle_common.dao.container_config.node_firewall_config import NodeFirewallConfig


class Topology:
    """
    A DTO representing the topology of an emulation environment
    """

    def __init__(self, node_configs: List[NodeFirewallConfig], subnetwork_masks: List[str]):
        """
        Initializes the DTO

        :param node_configs: the list of node configurations
        :param subnetwork: the subnetwork
        """
        self.node_configs = node_configs
        self.subnetwork_masks = subnetwork_masks

    def __str__(self) -> str:
        """
        :return: a string representation of the object
        """
        return "node configs:{}, subnetwork_masks:{}".format(",".join(list(map(lambda x: str(x), self.node_configs))),
                                                       ",".join(self.subnetwork_masks))