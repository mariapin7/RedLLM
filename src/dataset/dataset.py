import os
import json
from typing import List
from tqdm import tqdm
from src.llm.ask_llm import LLMWrapper

def normalize_for_storage(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return " ".join(text.replace("\n", " ").replace("\t", " ").split())

class DatasetGenerator:
    def __init__(self, data_path: str = "data/topologias", model = "llama3.2" ):
        self.__data_path = data_path
        self.model = model
        self.__questions = [
            "How many nodes are there in the network? Only answer with the number. Do not add any explanation.",
            "How many IPv4 addresses are assigned to devices? Only answer with the number. Do not add any explanation.",
            "Which devices have the most IP addresses assigned? Do not add any explanation.",
            #"Draw me the graph of my network.If you can't draw it, use ascii art",
            "Are there any devices with special-purpose IP addresses (e.g., multicast addresses)? Do not add any explanation.",
            "Do any devices have multiple IP addresses assigned to them? Do not add any explanation.",
            "Are there any IPv6 addresses assigned? Do not add any explanation.",
            "How many subnetworks are there in my network? Do not add any explanation.",
            "Is it possible to remove one subnetwork but keeping all the devices able to ping each other? Do not add any explanation.",
            "Which is the subnetwork that connects x1 to x2? Do not add any explanation.",
            "Is it possible for x1 to ping x2 without any hop? Answer directly with \"yes\" or \"no\"."

            # "How many nodes are there in the network? Only answer with the number.",
            # "How many IPv4 addresses are assigned to devices? ",
            # "Which devices have the most IP addresses assigned? ",
            # #"Draw me the graph of my network.If you can't draw it, use ascii art",
            # "Are there any devices with special-purpose IP addresses (e.g., multicast addresses)? ",
            # "Do any devices have multiple IP addresses assigned to them? ",
            # "Are there any IPv6 addresses assigned? ",
            # "How many subnetworks are there in my network? Only answer with the number. ",
            # "Is it possible to remove one subnetwork but keeping all the devices able to ping each other? ",
            # "Which is the subnetwork that connects x1 to x2? ",
            # "Is it possible for x1 to ping x2 without any hop? Answer directly with \"yes\" or \"no\". ",

         ]
        self.llm = LLMWrapper(model=model)

    def __get_files(self) -> List[str]:
        return [f for f in os.listdir(self.__data_path) if f.endswith(".json")]



    def generate_dataset(self, name: str, prompts: List[str], file_name: str = None) -> None:
        output_dir = f"data/{name}/"
        os.makedirs(output_dir, exist_ok=True)

        files = [file_name] if file_name else self.__get_files()

        for file in tqdm(files, desc="Generating dataset"):
            topology_name = os.path.splitext(file)[0]
            #output_file = os.path.join(output_dir, f"dataset_{topology_name}_{name}.jsonl")
            output_file = os.path.join(output_dir, f"dataset_{topology_name}_{name}_{self.model.replace(':', '_')}.jsonl")

            # qa_pairs = self.answer_questions_for_netjson(file_name)
            topology_path = os.path.join(self.__data_path, file)
            with open(topology_path, "r") as tf:
                topology_data = json.load(tf)
            topology_str = json.dumps(topology_data, indent=2)

            with open(output_file, "w") as f:
                for i, question in enumerate(self.__questions):
                    prompt = f"{prompts[i]}\n{question}" if prompts[i] else question
                    full_prompt = f"You are a network analyst. Here is the topology:\n{topology_str}\n\n{prompt}"
                    answer = self.llm.ask(full_prompt)
                    answer_normalized = normalize_for_storage(answer)

                    data = {
                        "context": "You are a network analyst and must answer questions about a network topology.",
                        "prompt": prompt,
                        "answer": answer_normalized

                    }
                    f.write(json.dumps(data) + "\n")

    def get_questions(self):
        return self.__questions
