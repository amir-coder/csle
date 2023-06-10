from typing import Dict, Any
from csle_base.json_serializable import JSONSerializable


class NmapHttpEnum(JSONSerializable):
    """
    DTO representing a NMAP HTTP Enum
    """

    def __init__(self, output: str):
        """
        Initializes the DTO

        :param output: the output of the HTTP enum
        """
        self.output = output

    def __str__(self) -> str:
        """
        :return: a string representation of the object
        """
        return f"output:{self.output}"

    def to_dict(self) -> Dict[str, Any]:
        """
        :return: a dict representation of the object
        """
        d = {}
        d["output"] = self.output
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "NmapHttpEnum":
        """
        Converts a dict representation to an instance

        :param d: the dict to convert
        :return: the created instance
        """
        obj = NmapHttpEnum(output=d["output"])
        return obj

    @staticmethod
    def from_json_file(json_file_path: str) -> "NmapHttpEnum":
        """
        Reads a json file and converts it to a DTO

        :param json_file_path: the json file path
        :return: the converted DTO
        """
        import io
        import json
        with io.open(json_file_path, 'r') as f:
            json_str = f.read()
        return NmapHttpEnum.from_dict(json.loads(json_str))
