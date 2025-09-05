# import json
# import os
# import string
# from sentence_transformers import SentenceTransformer, util
#
#
# def normalize(text: str) -> str:
#     """Limpia y normaliza una cadena: minúsculas, sin puntuación ni espacios extra."""
#     if not isinstance(text, str):
#         return ""
#     text = text.strip().lower()
#     text = text.replace("\n", " ").replace("\t", " ")
#     text = text.translate(str.maketrans('', '', string.punctuation))
#     text = " ".join(text.split())
#     return text
#
# def normalize_prompt(prompt: str) -> str:
#     #Extrae solo la pregunta real del final del prompt y la normaliza
#     pregunta_real = prompt.strip().split('\\n')[-1].strip()
#     return normalize(pregunta_real)
#
#
# class SBERT_Evaluator:
#     def __init__(self, generated_path: str, ground_truth_path: str):
#         self.model = SentenceTransformer("all-MiniLM-L6-v2")
#         #self.generated = self.__load_jsonl(generated_path)
#         #self.ground_truth = self.__load_jsonl(ground_truth_path)
#         #self.ground_truth_dict = {item["prompt"]: item["answer"] for item in self.ground_truth}
#         self.ground_truth_dict = {normalize(item["prompt"]): item["answer"] for item in self.ground_truth}
#
#     def load_ground_truth(self, path: str):
#         data = {}
#         with open(path, "r", encoding="utf-8") as f:
#             for line in f:
#                 item = json.loads(line)
#                 key = normalize(item["prompt"])
#                 data[key] = normalize(item["answer"])
#         return data
#
#     def evaluate(self, output_path: str = "data/evaluation/evaluation_sbert_normalized.jsonl"):
#         os.makedirs(os.path.dirname(output_path), exist_ok=True)
#
#         with open(output_path, "w") as f:
#             for item in self.generated:
#                 #prompt = item["prompt"]
#                 # Extrae la última pregunta del prompt (después del último ejemplo)
#                 prompt_completo = item["prompt"]
#                 prompt = prompt_completo.split("\n\n")[-1].strip()
#                 respuesta_generada = item.get("generated_output", item.get("answer"))
#                 #respuesta_esperada = self.ground_truth_dict.get(prompt, None)
#                 #respuesta_esperada = self.ground_truth_dict.get(normalize(prompt), None)
#                 # Extraer la última pregunta real del prompt (después del último '\n')
#                 pregunta_real = prompt.strip().split('\n')[-1].strip()
#                 pregunta_normalizada = normalize(pregunta_real)
#                 respuesta_esperada = self.ground_truth_dict.get(pregunta_normalizada, None)
#
#                 if respuesta_esperada is None:
#                     print(f" No se encontró ground truth para: {prompt}")
#                     continue
#
#                 #sim = self.__get_similarity(respuesta_generada, respuesta_esperada)
#                 sim = self.__get_similarity(normalize(respuesta_generada), normalize(respuesta_esperada))
#
#                 f.write(json.dumps({
#                     "prompt": prompt,
#                     "respuesta_generada": normalize(respuesta_generada),
#                     "respuesta_esperada": normalize(respuesta_esperada),
#                     "similaridad_sbert": round(sim, 4)
#                 }) + "\n")
#
#     def __get_similarity(self, a, b):
#         embeddings = self.model.encode([a, b], convert_to_tensor=True)
#         return util.cos_sim(embeddings[0], embeddings[1]).item()
#
#     def __load_jsonl(self, path):
#         with open(path, "r") as f:
#             return [json.loads(line) for line in f]

import json
import os
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

class SBERT_Evaluator:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def __load_jsonl(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    def evaluate(
        self,
        generated_path: str,
        ground_truth_path: str,
        output_path: str
    ):
        print("--------------------")
        print("     RedLLM CLI")
        print("--------------------")

        generated = self.__load_jsonl(generated_path)
        ground_truth = self.__load_jsonl(ground_truth_path)

        # Crear diccionario {pregunta: respuesta} para el ground truth
        gt_dict = {item["prompt"].strip(): item["answer"].strip() for item in ground_truth}

        results = []

        for item in tqdm(generated, desc="Evaluando con SBERT"):
            full_prompt = item["prompt"].strip()
            answer = item["answer"].strip()

            #  IMPORTANTE: extraemos solo la última línea del prompt (la pregunta real)
            question_only = full_prompt.split("\n")[-1].strip()

            if question_only not in gt_dict:
                print(f" No se encontró ground truth para: {question_only}")
                continue

            gt_answer = gt_dict[question_only]

            # Calcular similitud de embeddings
            embeddings = self.model.encode([answer, gt_answer], convert_to_tensor=True)
            similarity = util.cos_sim(embeddings[0], embeddings[1]).item()

            results.append({
                "question": question_only,
                "generated_answer": answer,
                "ground_truth": gt_answer,
                "similarity": similarity
            })

        # Guardar resultados
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        print(f" Evaluación SBERT guardada en {output_path}")
