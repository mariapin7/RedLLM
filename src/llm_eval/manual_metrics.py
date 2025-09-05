import json
import os
import re

def tokenize(text):
    return set(re.findall(r'\b\w+\b', text.lower()))


def calculate_metrics(gt, pred):
    gt_tokens = tokenize(gt)
    pred_tokens = tokenize(pred)

    if not gt_tokens:
        return 1.0 if not pred_tokens else 0.0, 1.0 if not pred_tokens else 0.0, 1.0 if not pred_tokens else 0.0

    true_positives = len(gt_tokens & pred_tokens)
    false_positives = len(pred_tokens - gt_tokens)
    false_negatives = len(gt_tokens - pred_tokens)

    precision = true_positives / (true_positives + false_positives + 1e-8)
    recall = true_positives / (true_positives + false_negatives + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

    return precision, recall, f1

def evaluar_manual(generated_path, ground_truth_path, output_path):
    with open(generated_path, "r", encoding="utf-8") as f:
        generated_data = [json.loads(line) for line in f]

    with open(ground_truth_path, "r", encoding="utf-8") as f:
        ground_truth_data = [json.loads(line) for line in f]

    gt_dict = {}
    for item in ground_truth_data:
        if "question" in item:
            key = item["question"]
        else:
            # Extraer solo la última línea del prompt (la pregunta real)
            key = item["prompt"].strip().split("\n")[-1].strip()
        gt_dict[key] = item["answer"]

    results = []
    for item in generated_data:
        question = item["question"]
        pred = item["generated_answer"]
        gt = gt_dict.get(question)

        if gt is None:
            print(f" Ground truth no encontrado para: {question}")
            continue

        # Antes de calculate_metrics(gt, pred)
        gt_clean = gt.strip().lower()
        pred_clean = pred.strip().lower()

        if gt_clean in ["yes", "no"] and pred_clean.startswith(gt_clean):
            precision, recall, f1 = 1.0, 1.0, 1.0
        else:
            precision, recall, f1 = calculate_metrics(gt, pred)



        item["ground_truth"] = gt
        item["precision"] = precision
        item["recall"] = recall
        item["f1_score"] = f1

        results.append(item)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f" Evaluación manual guardada en {output_path}")
