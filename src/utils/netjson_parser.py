import json

def load_netjson(file_path: str) -> dict:
    with open(file_path) as f:
        return json.load(f)

# Ejemplo de uso:
topologia = load_netjson("data/topologias/red1.netjson")
print(topologia)  # Verifica la estructura