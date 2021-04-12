from typing import List, Tuple
import copy
from gym_pycr_ctf.dao.observation.attacker.attacker_machine_observation_state import AttackerMachineObservationState


class NetworkOutcome:

    def __init__(self, attacker_machine_observations: List[AttackerMachineObservationState] = None,
                 attacker_machine_observation: AttackerMachineObservationState = None,
                 total_new_ports_found : int = 0, total_new_os_found: int = 0,
                 total_new_cve_vuln_found : int = 0, total_new_machines_found : int = 0,
                 total_new_shell_access : int = 0, total_new_flag_pts : int = 0, total_new_root : int = 0,
                 total_new_osvdb_vuln_found : int = 0, total_new_logged_in : int = 0,
                 total_new_tools_installed : int = 0, total_new_backdoors_installed : int = 0,
                 cost: float = 0.0, alerts: Tuple = (0,0)):
        if attacker_machine_observations is None:
            self.attacker_machine_observations = []
        else:
            self.attacker_machine_observations = attacker_machine_observations
        self.attacker_machine_observation = attacker_machine_observation
        self.total_new_ports_found = total_new_ports_found
        self.total_new_os_found  =total_new_os_found
        self.total_new_cve_vuln_found = total_new_cve_vuln_found
        self.total_new_machines_found = total_new_machines_found
        self.total_new_shell_access = total_new_shell_access
        self.total_new_flag_pts = total_new_flag_pts
        self.total_new_root = total_new_root
        self.total_new_osvdb_vuln_found = total_new_osvdb_vuln_found
        self.total_new_logged_in = total_new_logged_in
        self.total_new_tools_installed = total_new_tools_installed
        self.total_new_backdoors_installed = total_new_backdoors_installed
        self.cost = cost
        self.alerts = alerts

    def copy(self):
        net_outcome = NetworkOutcome(
            attacker_machine_observations=copy.deepcopy(self.attacker_machine_observations),
            attacker_machine_observation=self.attacker_machine_observation.copy(),
            total_new_ports_found=self.total_new_ports_found,
            total_new_os_found=self.total_new_os_found,
            total_new_cve_vuln_found=self.total_new_cve_vuln_found,
            total_new_machines_found=self.total_new_machines_found,
            total_new_shell_access=self.total_new_shell_access,
            total_new_flag_pts=self.total_new_flag_pts,
            total_new_root=self.total_new_root,
            total_new_osvdb_vuln_found=self.total_new_osvdb_vuln_found,
            total_new_logged_in=self.total_new_logged_in,
            total_new_tools_installed=self.total_new_tools_installed,
            total_new_backdoors_installed=self.total_new_backdoors_installed,
            cost=self.cost, alerts=self.alerts
        )
        return net_outcome

    def __str__(self):
        return "attacker_machine_observations:{},attacker_machine_observation:{}" \
               "total_new_ports_found:{},total_new_os_found:{}," \
               "total_new_cve_vuln_found:{},total_new_machines_found:{},total_new_shell_access:{}," \
               "total_new_flag_pts:{},total_new_root:{},total_new_osvdb_vuln_found:{}," \
               "total_new_logged_in:{},total_new_tools_installed:{},total_new_backdoors_installed:{}," \
               "cost:{}, alerts:{}".format(
            list(map(lambda x: str(x), self.attacker_machine_observations)),
            str(self.attacker_machine_observation),
            self.total_new_ports_found, self.total_new_os_found, self.total_new_cve_vuln_found,
            self.total_new_machines_found, self.total_new_shell_access, self.total_new_flag_pts,
            self.total_new_root, self.total_new_osvdb_vuln_found, self.total_new_logged_in,
            self.total_new_tools_installed, self.total_new_backdoors_installed, self.cost,
            self.alerts)


    def update_counts(self, net_outcome):
        self.total_new_ports_found += net_outcome.total_new_ports_found
        self.total_new_os_found += net_outcome.total_new_os_found
        self.total_new_cve_vuln_found += net_outcome.total_new_cve_vuln_found
        self.total_new_machines_found += net_outcome.total_new_machines_found
        self.total_new_shell_access += net_outcome.total_new_shell_access
        self.total_new_flag_pts += net_outcome.total_new_flag_pts
        self.total_new_root += net_outcome.total_new_root
        self.total_new_osvdb_vuln_found += net_outcome.total_new_osvdb_vuln_found
        self.total_new_logged_in += net_outcome.total_new_logged_in
        self.total_new_tools_installed += net_outcome.total_new_tools_installed
        self.total_new_backdoors_installed += net_outcome.total_new_backdoors_installed

    def update_counts_machine(self, machine: AttackerMachineObservationState):
        self.total_new_ports_found += len(machine.ports)
        new_os = 0 if machine.os == "unknown" else 1
        self.total_new_os_found += new_os
        self.total_new_cve_vuln_found += len(machine.cve_vulns)
        self.total_new_shell_access += 1 if machine.shell_access else 0
        self.total_new_flag_pts += len(machine.flags_found)
        self.total_new_root += 1 if machine.root else 0
        self.total_new_osvdb_vuln_found += len(machine.osvdb_vulns)
        self.total_new_logged_in += 1 if machine.logged_in else 0
        self.total_new_tools_installed += 1 if machine.tools_installed else 0
        self.total_new_backdoors_installed += 1 if machine.backdoor_installed else 0
        self.total_new_machines_found += 1