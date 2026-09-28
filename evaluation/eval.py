
import os
import json
import commentjson
import base64
import time
import requests
from dotenv import load_dotenv
from pathlib import Path


load_dotenv()
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
BASE_DIR = Path(__file__).resolve().parent

GOLDEN_DATASET_PATH = BASE_DIR / "golden_dataset.json"
RESULTS_PATH = BASE_DIR / "eval_results.json"

 
 
def image_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
 
 
def run_pipeline_on_golden_set(golden_entries: list) -> list:
    """Calls /diagnose for each golden entry and collects results, including
    rag_context (needed for the Context Precision metric)."""
    results = []
    for entry in golden_entries:
        print(f"[{entry['id']}/30] {entry['category']} — \"{entry['user_query'][:50]}...\"")
        try:
            # img_b64 = image_to_base64(entry["image_path"])
            response = requests.post(
                f"{BACKEND_URL}/diagnose",
                json={
                    # "images_base64": [img_b64],
                    "user_query": entry["user_query"],
                    "location": entry["location"],
                },
                timeout=200,
            )
            response.raise_for_status()
            data = response.json()
 
            results.append({
                "id": entry["id"],
                "category": entry["category"],
                "user_query": entry["user_query"],
                "reference_answer": entry["reference_answer"],
                "final_answer": data.get("final_answer"),
                "rag_context": data.get("rag_context"),
                "error": data.get("error"),
            })
        except Exception as e:
            print(f"  FAILED: {e}")
            results.append({
                "id": entry["id"],
                "category": entry["category"],
                "user_query": entry["user_query"],
                "reference_answer": entry["reference_answer"],
                "final_answer": None,
                "rag_context": None,
                "error": str(e),
            })
        time.sleep(1) 
 
    return results
 
 
def score_with_ragas(results: list):
    """Scores Faithfulness + Context Precision using RAGAS, with Groq as the judge LLM"""
    from ragas import evaluate
    from ragas.metrics import faithfulness, context_precision
    from ragas.llms import LangchainLLMWrapper
    from datasets import Dataset
    from langchain_groq import ChatGroq
    from ragas import RunConfig
 
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise RuntimeError(
            "GROQ_API_KEY not set — required for the RAGAS judge LLM. "
            "Set it in your environment before running eval."
        )
    
    
    run_config = RunConfig(
    timeout=180,
    max_workers=2
    )
    judge_model_name = os.getenv("RAGAS_JUDGE_MODEL", "openai/gpt-oss-120b")
    judge_llm = ChatGroq(
        model=judge_model_name,
        temperature=0,
        api_key=groq_api_key
        
    )
    ragas_judge = LangchainLLMWrapper(judge_llm)
 
    for metric in (faithfulness, context_precision):
        metric.llm = ragas_judge
 
    scoreable = [r for r in results if r["final_answer"] and r["rag_context"]]
    skipped = len(results) - len(scoreable)
    if skipped:
        print(f"\nSkipping {skipped} entries with missing answer/context (likely pipeline errors).")
 
    if not scoreable:
        print("No scoreable entries — nothing to evaluate.")
        return None
 
    ragas_dataset = Dataset.from_dict({
        "question": [r["user_query"] for r in scoreable],
        "answer": [r["final_answer"] for r in scoreable],
        "contexts": [[r["rag_context"]] for r in scoreable],  # RAGAS expects a list of context chunks
        "ground_truth": [r["reference_answer"] for r in scoreable],
    })
 
    score = evaluate(
        ragas_dataset,
        metrics=[faithfulness, context_precision],
        llm=ragas_judge,
        run_config=run_config
    )
    return score
 
 
def main():
    # with open(GOLDEN_DATASET_PATH) as f:
    #     golden_entries = json.load(f)

    with open(GOLDEN_DATASET_PATH, encoding="utf-8") as f:
        golden_entries = commentjson.load(f)
 
    print(f"Running {len(golden_entries)} golden queries against {BACKEND_URL}/diagnose ...\n")
    results = run_pipeline_on_golden_set(golden_entries)
 
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nRaw results saved to {RESULTS_PATH}")
 
    n_errors = sum(1 for r in results if r["error"])
    print(f"Completed: {len(results) - n_errors}/{len(results)} succeeded, {n_errors} failed")
 
    # Run RAGAS only on successful results
    successful_results = [
        r for r in results
        if not r["error"]
    ]

    if not successful_results:
        print("\nNo successful results available for RAGAS scoring.")
        return

    print("\nScoring successful results with RAGAS...")

    score = score_with_ragas(successful_results)

    if score is not None:
        print("\n=== RAGAS Scores ===")
        print(score)
 
 
if __name__ == "__main__":
    main()
 