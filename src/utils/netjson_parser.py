import json
import os

class NetJsonParser:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def load(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"No se encontró el archivo: {self.file_path}")

        with open(self.file_path, 'r', encoding='utf-8') as f:
            try:
                self.data = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Error al parsear JSON: {e}")

        self._validate()

    def _validate(self):
        if not isinstance(self.data, dict):
            raise ValueError("El archivo NetJSON debe contener un objeto JSON.")
        if "nodes" not in self.data or "links" not in self.data:
            raise ValueError("El archivo NetJSON debe contener los campos 'nodes' y 'links'.")
        if not isinstance(self.data["nodes"], list) or not isinstance(self.data["links"], list):
            raise ValueError("'nodes' y 'links' deben ser listas.")

    def get_data(self):
        if self.data is None:
            raise RuntimeError("Primero debes llamar a .load() para cargar la topología.")
        return self.data

    def netjson_to_prompt(self):
        prompt = "Esta es la topología de red:\n"
        for node in self.data["nodes"]:
            prompt += f"- Nodo {node['id']} con direcciones: {', '.join(node['local_addresses'])}\n"
        return prompt


def parse_netjson(input_path):
    parser = NetJsonParser(input_path)
    parser.load()
    print(parser.netjson_to_prompt())

