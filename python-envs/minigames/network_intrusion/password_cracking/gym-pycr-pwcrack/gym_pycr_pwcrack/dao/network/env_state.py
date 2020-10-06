import gym
import numpy as np
from gym_pycr_pwcrack.dao.network.network_config import NetworkConfig
from gym_pycr_pwcrack.dao.observation.observation_state import ObservationState

class EnvState:

    def __init__(self, network_config : NetworkConfig, num_ports : int, num_vuln : int, num_sh : int,
                 num_flags : int,
                 vuln_lookup: dict = None, service_lookup: dict = None, os_lookup: dict = None):
        self.network_config = network_config
        self.reward_range = (float(0), float(1))
        self.num_ports = num_ports
        self.num_vuln = num_vuln
        self.num_sh = num_sh
        self.num_flags = num_flags
        self.vuln_lookup = vuln_lookup
        self.vuln_lookup_inv = {v: k for k, v in self.vuln_lookup.items()}
        self.service_lookup = service_lookup
        self.service_lookup_inv = {v: k for k, v in self.service_lookup.items()}
        self.os_lookup = os_lookup
        self.os_lookup_inv = {v: k for k, v in self.os_lookup.items()}
        self.obs_state : ObservationState = None
        self.reset_state() # Init obs state
        self.num_m_features = 10 + self.obs_state.num_ports + self.obs_state.num_vuln
        self.observation_space = gym.spaces.Box(low=0, high=1000, dtype=np.float32, shape=(
            self.obs_state.num_machines*self.num_m_features,))
        self.m_selection_observation_space = gym.spaces.Box(low=0, high=1000, dtype=np.float32, shape=(
            self.obs_state.num_machines * self.num_m_features,))
        self.m_action_observation_space = gym.spaces.Box(low=0, high=1000, dtype=np.float32, shape=(self.num_m_features,))
        self.cached_ssh_connections = {}
        self.cached_telnet_connections = {}
        self.cached_ftp_connections = {}

    def get_observation(self):
        self.machines_obs = np.zeros(
            (self.obs_state.num_machines, self.num_m_features))
        self.ports_protocols_obs = np.zeros((self.obs_state.num_machines, self.obs_state.num_ports))
        for i in range(self.obs_state.num_machines):

            if len(self.obs_state.machines) > i:
                self.machines_obs[i][0] = i + 1
                self.obs_state.machines[i].sort_ports()
                self.obs_state.machines[i].sort_vuln(self.vuln_lookup)
                self.obs_state.machines[i].sort_shell_access(self.service_lookup)

                # IP
                host_ip = int(self.obs_state.machines[i].ip.rsplit(".", 1)[-1])
                self.machines_obs[i][1] = host_ip

                # OS
                os_id = self.os_lookup[self.obs_state.machines[i].os]
                self.machines_obs[i][2] = os_id

                # Ports
                for j, p in enumerate(self.obs_state.machines[i].ports):
                    s_id = self.service_lookup[p.service]
                    if j < self.obs_state.num_ports:
                        self.machines_obs[i][j + 3] = s_id
                        self.ports_protocols_obs[i][j] = p.protocol.value

                # Vulnerabilities
                for j, v in enumerate(self.obs_state.machines[i].vuln):
                    v_id = self.vuln_lookup[v.name]
                    if j < self.obs_state.num_vuln:
                        self.machines_obs[i][j + 3 + self.obs_state.num_ports] = v_id

                # Num Open Ports
                self.machines_obs[i][3 + self.obs_state.num_ports + self.obs_state.num_vuln] = len(
                    self.obs_state.machines[i].ports)

                # Num Vulnerabilities
                self.machines_obs[i][4 + self.obs_state.num_ports + self.obs_state.num_vuln] = len(
                    self.obs_state.machines[i].vuln)

                # Total CVSS score
                total_cvss = sum(list(map(lambda x: x.cvss, self.obs_state.machines[i].vuln)))
                self.machines_obs[i][5 + self.obs_state.num_ports + self.obs_state.num_vuln] = total_cvss

                # Shell Access
                self.machines_obs[i][6 + self.obs_state.num_ports + self.obs_state.num_vuln] = int(
                    self.obs_state.machines[i].shell_access)

                # Logged in
                self.machines_obs[i][7 + self.obs_state.num_ports + self.obs_state.num_vuln] = int(
                    self.obs_state.machines[i].logged_in)

                # Root access
                self.machines_obs[i][8 + self.obs_state.num_ports + self.obs_state.num_vuln] = int(
                    self.obs_state.machines[i].root)

                # Flag pts
                flag_pts_score = sum([f.score for f in self.obs_state.machines[i].flags_found])
                self.machines_obs[i][9 + self.obs_state.num_ports + self.obs_state.num_vuln] = int(flag_pts_score)

                # sh_services
                services = []
                for j, sh_c in enumerate(self.obs_state.machines[i].shell_access_credentials):
                    if sh_c.service is not None:
                        s_id = self.service_lookup[sh_c.service]
                        if j < self.obs_state.num_sh and s_id not in services:
                            services.append(s_id)
                            self.machines_obs[i][j + 3 + self.obs_state.num_ports + self.obs_state.num_vuln] = s_id

        return self.machines_obs, self.ports_protocols_obs



    def reset_state(self):
        if self.obs_state is not None:
            for m in self.obs_state.machines:
                for c in m.ssh_connections:
                    self.cached_ssh_connections[(m.ip, c.username, c.port)] = c
                for c in m.telnet_connections:
                    self.cached_telnet_connections[(m.ip, c.username, c.port)] = c
                for c in m.ftp_connections:
                    self.cached_ftp_connections[(m.ip, c.username, c.port)] = c
        self.obs_state = ObservationState(num_machines=len(self.network_config.nodes), num_ports=self.num_ports,
                                          num_vuln=self.num_vuln, num_sh=self.num_sh, num_flags=self.num_flags,
                                          catched_flags=0)


    def merge_services_with_cluster(self, cluster_services):
        max_id = max(self.service_lookup.values())
        for service in cluster_services:
            if service not in self.service_lookup:
                max_id += 1
                self.service_lookup[service] = max_id
        self.service_lookup_inv = {v: k for k, v in self.service_lookup.items()}


    def merge_cves_with_cluster(self, cluster_cves):
        max_id = max(self.vuln_lookup.values())
        for cve in cluster_cves:
            if cve not in self.vuln_lookup:
                max_id += 1
                self.vuln_lookup[cve] = max_id
        self.vuln_lookup_inv = {v: k for k, v in self.vuln_lookup.items()}


    def cleanup(self):
        """
        Cleanup
        :return:
        """
        for _, c in self.cached_ssh_connections.items():
            c.cleanup()
        for _, c in self.cached_ftp_connections.items():
            c.cleanup()
        for _, c in self.cached_telnet_connections.items():
            c.cleanup()

        self.obs_state.cleanup()