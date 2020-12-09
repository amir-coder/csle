import os
from gym_pycr_pwcrack.dao.container_config.users_config import UsersConfig
from gym_pycr_pwcrack.dao.container_config.node_users_config import NodeUsersConfig
from gym_pycr_pwcrack.util.experiments_util import util
from gym_pycr_pwcrack.dao.network.cluster_config import ClusterConfig
from gym_pycr_pwcrack.envs.config.generator.users_generator import UsersGenerator

def default_users() -> UsersConfig:
    users = [
        NodeUsersConfig(ip="172.18.4.79", users = [
            ("l_hopital", "l_hopital", True),
            ("pi", "pi", True),
            ("euler", "euler", False)
        ]),
        NodeUsersConfig(ip="172.18.4.191", users=[
            ("agent", "agent", True)
        ]),
        NodeUsersConfig(ip="172.18.4.21", users=[
            ("admin", "admin31151x", True),
            ("test", "qwerty", True),
            ("oracle", "abc123", False)
        ]),
        NodeUsersConfig(ip="172.18.4.10", users=[
            ("admin", "admin1235912", True),
            ("jessica", "water", False)
        ]),
        NodeUsersConfig(ip="172.18.4.2", users=[
            ("admin", "test32121", True),
            ("user1", "123123", True),
            ("puppet", "puppet", False)
        ]),
        NodeUsersConfig(ip="172.18.4.3", users=[
            ("admin", "admin", True),
            ("john", "doe", True),
            ("vagrant", "test_pw1", False)
        ])
    ]
    users_conf = UsersConfig(users=users)
    return users_conf


if __name__ == '__main__':
    if not os.path.exists(util.default_users_path()):
        UsersGenerator.write_users_config(default_users())
    users_config = util.read_users_config(util.default_users_path())
    cluster_config = ClusterConfig(agent_ip="172.18.4.191", agent_username="pycr_admin",
                                   agent_pw="pycr@admin-pw_191", server_connection=False)
    UsersGenerator.create_users(users_config=users_config, cluster_config=cluster_config)