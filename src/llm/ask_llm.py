

import subprocess

class LLMWrapper:
    def __init__(self, model="llama3.2"):
        self.model = model

    def ask(self, prompt: str) -> str:
        res = subprocess.run(
            ["ollama", "run", self.model],
            input=prompt,
            text=True,
            capture_output=True,
            encoding="utf-8"
        )
        if res.returncode != 0:
            print(" ollama:", res.stderr)
            return ""
        return res.stdout.strip()




# import os
# import requests
# from requests.exceptions import ReadTimeout, RequestException
#
# class LLMWrapper:
#     def __init__(self, model="llama3.2", base_url=None,
#                  num_predict=64, temperature=0.1, top_p=0.9,
#                  connect_timeout=10, read_timeout=900):  # ⬅️ lee hasta 15 min
#         self.model = model
#         self.base_url = (base_url or os.getenv("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
#         self.num_predict = num_predict
#         self.temperature = temperature
#         self.top_p = top_p
#         self.connect_timeout = connect_timeout
#         self.read_timeout = read_timeout
#
#     def ask(self, prompt: str) -> str:
#         payload = {
#             "model": self.model,
#             "prompt": prompt,
#             "stream": False,
#             "options": {
#                 "num_predict": self.num_predict,
#                 "temperature": self.temperature,
#                 "top_p": self.top_p,
#             }
#         }
#         try:
#             r = requests.post(
#                 f"{self.base_url}/api/generate",
#                 json=payload,
#                 timeout=(self.connect_timeout, self.read_timeout)  # (conn, read)
#             )
#             r.raise_for_status()
#             return r.json().get("response", "").strip()
#         except ReadTimeout:
#             print(f" Read timeout con {self.model}. Sube read_timeout o prueba un modelo más ligero.")
#             return ""
#         except RequestException as e:
#             print(" Error al conectar con Ollama:", e)
#             return ""
