import os

import click

from src.dataset.ground_truth import GroundTruthGenerator
from src.utils.netjson_parser import NetJsonParser
from src.dataset.dataset import DatasetGenerator
from src.llm.zero_shot import ask_question_zero_shot
from src.llm_eval.manual_metrics import evaluar_manual



from src.llm_eval.evaluate_sbert import SBERT_Evaluator

#from src.llm_eval.llm_eval import LLM_Evaluator


@click.group()
def redllm():
    click.echo("--------------------")
    click.echo("     RedLLM CLI")
    click.echo("--------------------")


@redllm.command()
@click.option("--file", prompt="data/topologias", help="Path to the NetJSON topology file")
@click.option("--mode", type=click.Choice(["zero_shot", "one_shot", "few_shot", "chain_of_thought"]), default="zero_shot", help="Técnica de prompting")
@click.option("--model", default="llama3.2", help="Nombre del modelo")
def ask(file, mode, model):
    """Haz una pregunta al LLM sobre una topología de red"""
    click.echo("Cargando topología y preguntando...")
    parser = NetJsonParser(file)
    parser.load()
    topology = parser.get_data()
    prompt = parser.netjson_to_prompt()

    while True:
        question = input("Pregunta ('exit' para salir): ")
        if question.lower() == "exit":
            break

        # Selección de técnica
        if mode == "zero_shot":
            response = ask_question_zero_shot(prompt, question, model=model)
        elif mode == "one_shot":
            from src.llm.one_shot import ask_question_one_shot
            response = ask_question_one_shot(prompt, question, model=model)
        elif mode == "few_shot":
            from src.llm.few_shot import ask_question_few_shot
            response = ask_question_few_shot(prompt, question, model=model)
        elif mode == "chain_of_thought":
            from src.llm.chain_of_thought import ask_question_chain_of_thought
            response = ask_question_chain_of_thought(prompt, question, model=model)
        else:
            response = "Invalid mode selected."

        print(f"Respuesta ({mode}): {response}")



@redllm.command()
@click.option("--input_path", default="data/topologias/", help="Ruta del archivo NetJSON")
def ingest(input_path):
    click.echo(f"Ingestando NetJSON desde {input_path}")
    # Aquí llamarías a tu clase que convierte NetJSON en estructura de red interna
    from src.utils.netjson_parser import parse_netjson
    parse_netjson(input_path)





@redllm.command()
@click.option("--model", default="llama3.2", help="Nombre del modelo Ollama a usar")
@click.option("--data_path", default="data/topologias/", help="Path to the data folder")
@click.option("--file_name", default=None, help="Nombre del archivo NetJSON a procesar (opcional)")
@click.option(
    "--zero_shot",
    is_flag=True,
    default=False,
    help="Generate zero shot dataset",
)
@click.option(
    "--one_shot", is_flag=True, default=False, help="Generate one shot dataset"
)

@click.option(
    "--few_shot", is_flag=True, default=False, help="Generate few shot dataset"
)

@click.option(
    "--chain_of_thought",
    is_flag=True,
    default=False,
    help="Generate chain-of-thought dataset",
)

def generate_dataset(data_path, file_name, zero_shot, one_shot, few_shot, chain_of_thought, model):
    click.echo("Generating dataset")
    click.echo(f"Using model: {model}")
    dataset = DatasetGenerator(os.path.join(os.getcwd(), data_path), model=model)
    questions = dataset.get_questions()

    selected_modes = [zero_shot, one_shot, few_shot, chain_of_thought]
    if sum(selected_modes) != 1:
        click.echo("Debes seleccionar una y solo una técnica: --zero_shot, --one_shot o --few_shot")
        return

    if one_shot:
        click.echo("##############################")
        click.echo("Generating one shot dataset")
        click.echo("##############################")
        ejemplos = [
            "Example:\nNetwork with 4 nodes.\n\nQuestion: How many nodes are there in the network? Only answer with the number.\nAnswer: 4\n",
            "Example:\nDevice r1 has 2 IPv4 addresses, r2 has 2, and r3 has 3 → total 7.\n\nQuestion: How many IPv4 addresses are assigned to devices?\nAnswer: 12\n",
            "Example:\nDevice A has 3 IPs, B has 1, C has 2 → A has the most.\n\nQuestion: Which devices have the most IP addresses assigned?\nAnswer: A\n",
            #"Example:\nSimple topology:\nclient -- switch -- server\n\nQuestion: Draw me the graph of my network. If you can't draw it, use ascii art\nAnswer: client -- switch -- server\n",
            "Example:\nAddress 224.0.0.1 is multicast.\n\nQuestion: Are there any devices with special-purpose IP addresses (e.g., multicast addresses)?\nAnswer: Yes\n",
            "Example:\nDevice A: 192.168.1.1, 10.0.0.1 → multiple IPs\n\nQuestion: Do any devices have multiple IP addresses assigned to them?\nAnswer: Yes\n",
            "Example:\nOnly addresses like 2001:db8:: are IPv6.\n\nQuestion: Are there any IPv6 addresses assigned?\nAnswer: No\n",
            "Example:\nDevices use 192.168.1.0/24 and 10.0.0.0/24 → 2 subnets.\n\nQuestion: How many subnetworks are there in my network? Only answer with the number.\nAnswer: 2\n",
            "Example:\nRemoving one subnet breaks ping between A and C.\n\nQuestion: Is it possible to remove one subnetwork but keeping all the devices able to ping each other?\nAnswer: No\n",
            "Example:\nx1 is on 10.0.0.1/24, x2 on 10.0.0.2/24 → both in 10.0.0.0/24\n\nQuestion: Which is the subnetwork that connects x1 to x2?\nAnswer: 10.0.0.0/24\n",
            "Example:\nx1 and x2 are on different networks.\n\nQuestion: Is it possible for x1 to ping x2 without any hop? Answer directly with \"yes\" or \"no\".\nAnswer: no\n",
            "Example:\nx1 and x2 are on same switch.\n\nQuestion: Is it possible for x1 to ping x2 without any hop? Answer directly with \"yes\" or \"no\".\nAnswer: yes\n"
        ]
        dataset.generate_dataset("one_shot", prompts=ejemplos, file_name=file_name)
        click.echo("Dataset one-shot generado correctamente.")

    elif few_shot:
        click.echo("##############################")
        click.echo("Generating few shot dataset")
        click.echo("##############################")
        dataset.generate_dataset("few_shot", prompts=[""] * len(questions), file_name=file_name)
        click.echo("Dataset few-shot generado correctamente.")

    elif zero_shot:
        click.echo("##############################")
        click.echo("Generating zero shot dataset")
        click.echo("##############################")
        dataset.generate_dataset("zero_shot", prompts=[""] * len(questions), file_name=file_name)
        click.echo("Dataset zero-shot generado correctamente.")


    elif chain_of_thought:
        click.echo("##############################")
        click.echo("Generating Chain-of-Thought dataset")
        click.echo("##############################")
        dataset.generate_dataset("chain_of_thought", prompts=[""] * len(questions), file_name=file_name)
        click.echo("Dataset Chain-of-Thought generado correctamente.")



@redllm.command()
@click.option("--data_path", default="data/topologias", help="Carpeta con topologías")
@click.option("--output_path", default="data/ground_truth", help="Carpeta de salida")
@click.option("--file_name", default=None, help="Nombre de archivo específico (opcional)")
def generate_ground_truth(data_path, output_path, file_name):
    generator = GroundTruthGenerator(data_path)
    generator.generate_ground_truth(output_path=output_path, file_name=file_name)
    click.echo("Ground truth generado correctamente.")


@redllm.command()
@click.option("--generated_path", default="data/zero_shot/dataset_routers_2.jsonl", help="Respuestas del modelo")
@click.option("--ground_truth_path", default="data/ground_truth/answers_routers_2.jsonl", help="Respuestas correctas manuales")
@click.option("--output_path", default="data/evaluation/evaluation_sbert_routers_normalized.jsonl", help="Archivo de salida")
def evaluate_sbert(generated_path, ground_truth_path, output_path):
    # evaluator = SBERT_Evaluator(generated_path, ground_truth_path)
    # evaluator.evaluate(output_path)
    evaluator = SBERT_Evaluator()
    evaluator.evaluate(
        generated_path=generated_path,
        ground_truth_path=ground_truth_path,
        output_path=output_path
    )
    click.echo(f"Evaluación SBERT guardada en {output_path}")


@redllm.command()
@click.option("--generated_path", required=True, help="Archivo con respuestas generadas")
@click.option("--ground_truth_path", required=True, help="Archivo con ground truth")
@click.option("--output_path", required=True, help="Ruta donde guardar la evaluación manual")
def evaluate_manual(generated_path, ground_truth_path, output_path):
    evaluar_manual(generated_path, ground_truth_path, output_path)
    click.echo(f" Evaluación manual guardada en {output_path}")



if __name__ == "__main__":
    redllm()
