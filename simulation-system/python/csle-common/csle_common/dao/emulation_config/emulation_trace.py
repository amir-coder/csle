from typing import List, Dict, Any
import json
import os
import numpy as np
import csle_common.constants.constants as constants
from csle_common.dao.emulation_action.attacker.emulation_attacker_action import EmulationAttackerAction
from csle_common.dao.emulation_action.defender.emulation_defender_action import EmulationDefenderAction
from csle_common.dao.emulation_observation.attacker.emulation_attacker_observation_state \
    import EmulationAttackerObservationState
from csle_common.dao.emulation_observation.defender.emulation_defender_observation_state \
    import EmulationDefenderObservationState


class EmulationTrace:
    """
    DTO class representing a trace in the emulation system
    """

    def __init__(self, initial_attacker_observation_state: EmulationAttackerObservationState,
                 initial_defender_observation_state: EmulationDefenderObservationState, emulation_name : str):
        """
        Initializes the DTO

        :param initial_attacker_observation_state: the initial state of the attacker
        :param initial_defender_observation_state: the intial state of the defender
        :param emulation_name: the name of the emulation
        """
        self.initial_attacker_observation_state = initial_attacker_observation_state
        self.initial_defender_observation_state = initial_defender_observation_state
        self.attacker_observation_states : List[EmulationAttackerObservationState] = []
        self.defender_observation_states : List[EmulationDefenderObservationState] = []
        self.attacker_actions : List[EmulationAttackerAction] = []
        self.defender_actions : List[EmulationDefenderAction] = []
        self.emulation_name = emulation_name
        self.id = -1

    def __str__(self):
        """
        :return: a string representation of the object
        """
        print(f"initial_attacker_observation_state:{self.initial_attacker_observation_state}"
              f"initial_defender_observation_state:{self.initial_defender_observation_state}"
              f"attacker_observation_states:{self.attacker_observation_states}\n"
              f"defender_observation_states:{self.defender_observation_states}\n"
              f"attacker_actions:{self.attacker_actions}\n"
              f"defender_actions:{self.defender_actions}\n"
              f"emulation_name: {self.emulation_name},"
              f"id:{self.id}")

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "EmulationTrace":
        """
        Converts a dict representation into an instance

        :param d: the dict to convert
        :return: the created instance
        """
        obj = EmulationTrace(
            initial_attacker_observation_state=EmulationAttackerObservationState.from_dict(
                d["initial_attacker_observation_state"]),
            initial_defender_observation_state=EmulationDefenderObservationState.from_dict(
                d["initial_defender_observation_state"]),
            emulation_name=d["emulation_name"]
        )
        obj.attacker_observation_states = list(map(lambda x: EmulationAttackerObservationState.from_dict(x),
                                                   d["attacker_observation_states"]))
        obj.defender_observation_states = list(map(lambda x: EmulationDefenderObservationState.from_dict(x),
                                                   d["defender_observation_states"]))
        obj.attacker_actions = list(map(lambda x: EmulationAttackerAction.from_dict(x),
                                                   d["attacker_actions"]))
        obj.defender_actions = list(map(lambda x: EmulationDefenderAction.from_dict(x),
                                        d["defender_actions"]))
        return obj

    def to_dict(self) -> Dict[str, Any]:
        """
        :return: a dict representation of the object
        """
        d = {}
        d["initial_attacker_observation_state"] = self.initial_attacker_observation_state.to_dict()
        d["initial_defender_observation_state"] = self.initial_defender_observation_state.to_dict()
        d["attacker_observation_states"] = list(map(lambda x: x.to_dict(), self.attacker_observation_states))
        d["defender_observation_states"] = list(map(lambda x: x.to_dict(), self.defender_observation_states))
        d["attacker_actions"] = list(map(lambda x: x.to_dict(), self.attacker_actions))
        d["defender_actions"] = list(map(lambda x: x.to_dict(), self.defender_actions))
        d["emulation_name"] = self.emulation_name
        d["id"] = self.id
        return d

    @staticmethod
    def save_traces_to_disk(traces_save_dir, traces : List["EmulationTrace"],
                            traces_file : str = None) -> None:
        """
        Utility function for saving a list of traces to a json file

        :param traces_save_dir: the directory where to save the traces
        :param traces: the traces to save
        :param traces_file: the filename of the traces file
        :return: None
        """
        traces = list(map(lambda x: x.to_dict(), traces))
        if traces_file is None:
            traces_file =  constants.SYSTEM_IDENTIFICATION.EMULATION_TRACES_FILE
        if not os.path.exists(traces_save_dir):
            os.makedirs(traces_save_dir)
        with open(traces_save_dir + "/" + traces_file, 'w') as fp:
            json.dump({"traces": traces}, fp, cls=NpEncoder)

    @staticmethod
    def load_traces_from_disk(traces_save_dir, traces_file : str = None) -> List["EmulationTrace"]:
        """
        Utility function for loading and parsing a list of traces from a json file

        :param traces_save_dir: the directory where to load the traces from
        :param traces_file: (optional) a custom name of the traces file
        :return: a list of the loaded traces
        """
        if traces_file is None:
            traces_file =  constants.SYSTEM_IDENTIFICATION.TRACES_FILE
        path = traces_save_dir + constants.COMMANDS.SLASH_DELIM + traces_file
        if os.path.exists(path):
            with open(path, 'r') as fp:
                d = json.load(fp)
                traces  = d["traces"]
                traces = list(map(lambda x: EmulationTrace.from_dict(x), traces))
                return traces
        else:
            print("Warning: Could not read traces file, path does not exist:{}".format(path))
            return []

    def to_json_str(self) -> str:
        """
        Converts the DTO into a json string

        :return: the json string representation of the DTO
        """
        import json
        json_str = json.dumps(self.to_dict(), indent=4, sort_keys=True)
        return json_str

    def to_json_file(self, json_file_path: str) -> None:
        """
        Saves the DTO to a json file

        :param json_file_path: the json file path to save  the DTO to
        :return: None
        """
        import io
        json_str = self.to_json_str()
        with io.open(json_file_path, 'w', encoding='utf-8') as f:
            f.write(json_str)

    @staticmethod
    def from_json_str(json_str: str) -> "EmulationTrace":
        """
        Converts json string into a DTO

        :param json_str: the json string representation
        :return: the DTO instance
        """
        import json
        dto: EmulationTrace = EmulationTrace.from_dict(json.loads(json_str))
        return dto

    @staticmethod
    def from_json_file(json_file_path: str) -> "EmulationTrace":
        """
        Reads a json file and converts it into a dto

        :param json_file_path: the json file path to save  the DTO to
        :return: None
        """
        import io
        with io.open(json_file_path, 'r', encoding='utf-8') as f:
            json_str = f.read()
            dto = EmulationTrace.from_json_str(json_str=json_str)
            return dto

    def num_attributes_per_time_step(self) -> int:
        """
        :return: approximately the number of attributes recorded per time-step of the trace
        """
        num_attributes = 2
        num_attributes = num_attributes + (1+len(self.attacker_observation_states))*\
                         self.initial_attacker_observation_state.num_attributes()
        num_attributes = num_attributes + (1+len(self.defender_observation_states))* \
                         self.initial_defender_observation_state.num_attributes()
        if len(self.defender_actions) > 0:
            num_attributes = num_attributes + len(self.defender_actions)* self.defender_actions[0].num_attributes()
        if len(self.attacker_actions) > 0:
            num_attributes = num_attributes + len(self.attacker_actions)* self.attacker_actions[0].num_attributes()
        return num_attributes

    @staticmethod
    def get_schema():
        """
        :return: the schema of the DTO
        """
        dto = EmulationTrace(initial_attacker_observation_state=EmulationAttackerObservationState.schema(),
                             initial_defender_observation_state=EmulationDefenderObservationState.schema(),
                             emulation_name="")
        dto.attacker_observation_states = [EmulationAttackerObservationState.schema()]
        dto.defender_observation_states = [EmulationDefenderObservationState.schema()]
        dto.attacker_actions = [EmulationAttackerAction.schema()]
        dto.defender_actions = [EmulationDefenderAction.schema()]
        return dto


class NpEncoder(json.JSONEncoder):
    """
    Encoder for Numpy arrays to JSON
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)