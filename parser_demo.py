from src.utils.netjson_parser import NetJsonParser
from src.llm.zero_shot import ask_question_zero_shot


parser = NetJsonParser("data/topologias/routers.json")
parser.load()
netjson = parser.get_data()

def netjson_to_prompt(netjson):
    prompt = "Esta es la topología de red:\n"
    for node in netjson["nodes"]:
        prompt += f"- Nodo {node['id']} con direcciones: {', '.join(node['local_addresses'])}\n"
    return prompt

prompt = parser.netjson_to_prompt()
pregunta = "¿Cuántos nodos hay en la red?"
respuesta = ask_question_zero_shot(prompt, pregunta)
print(f"\nPREGUNTA: {pregunta}")
print(f"RESPUESTA: {respuesta}")


# def main():
#     file_path = "data/topologias/routers.json"  # Asegúrate de que existe
#     parser = NetJsonParser(file_path)
#
#     try:
#         parser.load()
#         data = parser.get_data()
#         print("Topología cargada correctamente:")
#         print(data)
#     except Exception as e:
#         print(f" Error: {e}")
#
# if __name__ == "__main__":
#     main()
