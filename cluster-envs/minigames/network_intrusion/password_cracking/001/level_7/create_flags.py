import os
from gym_pycr_pwcrack.dao.container_config.flags_config import FlagsConfig
from gym_pycr_pwcrack.dao.container_config.node_flags_config import NodeFlagsConfig
from gym_pycr_pwcrack.util.experiments_util import util
from gym_pycr_pwcrack.dao.network.cluster_config import ClusterConfig
from gym_pycr_pwcrack.envs.config.generator.flags_generator import FlagsGenerator

def default_flags() -> FlagsConfig:
    flags = [
        NodeFlagsConfig(ip="172.18.7.79", flags = [("/tmp/flag3.txt", "flag3", "/tmp/", 3, True, 1)]),
        NodeFlagsConfig(ip="172.18.7.2", flags=[("/tmp/flag2.txt", "flag2", "/tmp/", 2, True, 1)]),
        NodeFlagsConfig(ip="172.18.7.3", flags=[("/root/flag1.txt", "flag1", "/root/", 1, True, 1)]),
        NodeFlagsConfig(ip="172.18.7.19", flags=[("/tmp/flag4.txt", "flag4", "/tmp/", 4, True, 1)]),
        NodeFlagsConfig(ip="172.18.7.31", flags=[("/tmp/flag5.txt", "flag5", "/tmp/", 5, True, 1)]),
    ]
    flags_config = FlagsConfig(flags=flags)
    return flags_config


if __name__ == '__main__':
    if not os.path.exists(util.default_flags_path()):
        FlagsGenerator.write_flags_config(default_flags())
    flags_config = util.read_flags_config(util.default_flags_path())
    cluster_config = ClusterConfig(agent_ip="172.18.7.191", agent_username="pycr_admin",
                                   agent_pw="pycr@admin-pw_191", server_connection=False)
    FlagsGenerator.create_flags(flags_config=flags_config, cluster_config=cluster_config)