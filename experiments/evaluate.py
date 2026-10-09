"""Measure retrieval and answer quality on experiments/eval/dataset.json.

Run from the project root:

    python experiments/evaluate.py          # retrieval and threshold sweep, about a minute
    python experiments/evaluate.py --llm    # also generate answers; ~15 minutes on a CPU

Settings come from the environment and .env like the app's, so a different model or
threshold can be evaluated with e.g. RELEVANCE_THRESHOLD=0.78. Each run writes a JSON
report to experiments/results/.
"""

import argparse
import dataclasses
import json
import re
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_DIRECTORY = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIRECTORY))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_DIRECTORY / ".env")

from app.rag.config import Settings  # noqa: E402
from app.rag.pipeline import INSUFFICIENT_EVIDENCE, RAGPipeline  # noqa: E402
from app.rag.retrieval import retrieve  # noqa: E402


ANSWERABLE = ("answerable", "answerable_english")
UNANSWERABLE = ("unanswerable_related", "unanswerable_offtopic")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=PROJECT_DIRECTORY / "experiments/eval/dataset.json", type=Path)
    parser.add_argument("--llm", action="store_true", help="also generate and score answers")
    parser.add_argument("--threshold", type=float, help="override RELEVANCE_THRESHOLD")
    parser.add_argument("--top-k", type=int, help="override TOP_K")
    args = parser.parse_args()

    settings = Settings()
    overrides = {"relevance_threshold": args.threshold, "top_k": args.top_k}
    settings = dataclasses.replace(settings, **{k: v for k, v in overrides.items() if v is not None})
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    questions = dataset["questions"]

    print(f"Embedding model: {settings.embedding_model_name}")
    print(f"Threshold: {settings.relevance_threshold}  top_k: {settings.top_k}  questions: {len(questions)}\n")

    pipeline = RAGPipeline(settings=settings)
    for document in dataset["documents"]:
        pipeline.index_file(PROJECT_DIRECTORY / document)
    print(f"Indexed {len(dataset['documents'])} documents, {len(pipeline.vector_store.chunks)} chunks\n")

    rows = [_retrieval_row(question, pipeline) for question in questions]
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "settings": dataclasses.asdict(settings),
        "dataset": str(args.dataset.relative_to(PROJECT_DIRECTORY)),
        "retrieval": _retrieval_metrics(rows),
        "threshold_sweep": _threshold_sweep(rows, settings.relevance_threshold),
    }
    _print_retrieval(report, rows, settings)

    if args.llm:
        for row in rows:
            _answer_row(row, pipeline)
        report["answers"] = _answer_metrics(rows)
        _print_answers(report["answers"], rows)

    report["questions"] = rows
    results_directory = PROJECT_DIRECTORY / "experiments/results"
    results_directory.mkdir(exist_ok=True)
    path = results_directory / f"{datetime.now():%Y%m%d-%H%M%S}{'-llm' if args.llm else ''}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nReport written to {path.relative_to(PROJECT_DIRECTORY)}")


def _retrieval_row(question, pipeline):
    settings = pipeline.settings
    results = retrieve(
        question["question"],
        pipeline.vector_store,
        top_k=settings.top_k,
        relevance_threshold=float("-inf"),
        embedding_function=pipeline.embedding_function,
        embedding_model_name=settings.embedding_model_name,
        query_prefix=settings.embedding_query_prefix,
    )
    row = {
        "id": question["id"],
        "category": question["category"],
        "question": question["question"],
        "top_score": results[0]["score"] if results else None,
        "rank": None,
        "relevant_score": None,
    }
    if "document" in question:
        evidence = question["evidence"].lower()
        for rank, result in enumerate(results, start=1):
            if result["filename"] == question["document"] and evidence in result["text"].lower():
                row["rank"], row["relevant_score"] = rank, result["score"]
                break
        row["expected_answer"] = question["answer"]
    return row


def _retrieval_metrics(rows):
    metrics = {}
    for category in ANSWERABLE:
        selected = [row for row in rows if row["category"] == category]
        if not selected:
            continue
        ranks = [row["rank"] for row in selected]
        metrics[category] = {
            "count": len(selected),
            "hit_at_1": _share(rank == 1 for rank in ranks),
            "hit_at_k": _share(rank is not None for rank in ranks),
            "mrr": round(sum(1 / rank for rank in ranks if rank) / len(ranks), 3),
        }
    return metrics


def _threshold_sweep(rows, current):
    answerable = [row for row in rows if row["category"] == "answerable"]
    english = [row for row in rows if row["category"] == "answerable_english"]
    related = [row for row in rows if row["category"] == "unanswerable_related"]
    offtopic = [row for row in rows if row["category"] == "unanswerable_offtopic"]
    candidates = sorted({round(0.70 + step / 100, 2) for step in range(21)} | {round(current, 2)})

    sweep = []
    for threshold in candidates:
        sweep.append(
            {
                "threshold": threshold,
                # The relevant chunk must survive the threshold for the LLM to see it.
                "answerable_kept": _share((row["relevant_score"] or -1) >= threshold for row in answerable),
                "english_kept": _share((row["relevant_score"] or -1) >= threshold for row in english),
                "offtopic_rejected": _share(row["top_score"] < threshold for row in offtopic),
                "related_rejected": _share(row["top_score"] < threshold for row in related),
            }
        )

    relevant_scores = [row["relevant_score"] for row in answerable if row["relevant_score"] is not None]
    lowest_relevant = min(relevant_scores) if relevant_scores else None
    highest_offtopic = max(row["top_score"] for row in offtopic) if offtopic else None
    recommended = None
    if lowest_relevant is not None and highest_offtopic is not None and lowest_relevant > highest_offtopic:
        # The middle of the gap leaves the most room for questions this set doesn't cover;
        # the edges would fit these particular questions too closely.
        recommended = round((lowest_relevant + highest_offtopic) / 2, 3)
    return {
        "lowest_relevant_score": lowest_relevant,
        "highest_offtopic_score": highest_offtopic,
        "recommended_threshold": recommended,
        "sweep": sweep,
    }


def _answer_row(row, pipeline):
    started = time.perf_counter()
    result = pipeline.ask(row["question"])
    row["seconds"] = round(time.perf_counter() - started, 1)
    row["answer"] = result["answer"]
    row["refused"] = result["answer"] == INSUFFICIENT_EVIDENCE
    row["verification"] = result["verification"]

    if row["category"] in ANSWERABLE:
        matched = all(
            any(_contains(result["answer"], alternative) for alternative in group)
            for group in row["expected_answer"]
        )
        row["outcome"] = "refused" if row["refused"] else ("correct" if matched else "wrong")
    else:
        row["outcome"] = "correct" if row["refused"] else "hallucinated"
    print(f"  [{row['seconds']:>5.1f}s] {row['outcome']:<12} {row['id']}", flush=True)


def _contains(text, expected):
    if expected[0].isdigit():
        # Match whole numbers so that "22" doesn't count inside "220" or "2022".
        return re.search(rf"(?<![\d.,]){re.escape(expected)}(?![.,]?\d)", text) is not None
    return expected.lower() in text.lower()


def _answer_metrics(rows):
    answered = [row for row in rows if row.get("verification")]
    metrics = {}
    for category in ANSWERABLE + UNANSWERABLE:
        selected = [row for row in rows if row["category"] == category]
        if selected:
            outcomes = [row["outcome"] for row in selected]
            metrics[category] = {
                "count": len(selected),
                **{outcome: outcomes.count(outcome) for outcome in sorted(set(outcomes))},
                "accuracy": _share(outcome == "correct" for outcome in outcomes),
            }
    metrics["verification"] = {
        "answers_checked": len(answered),
        "with_unsupported_numbers": sum(bool(row["verification"]["unsupported_numbers"]) for row in answered),
        "with_invalid_citations": sum(bool(row["verification"]["invalid_citations"]) for row in answered),
        "without_citations": sum(not row["verification"]["citations"] for row in answered),
    }
    llm_rows = [row for row in rows if row.get("verification") is not None or not row["refused"]]
    metrics["median_seconds_with_llm"] = (
        statistics.median(row["seconds"] for row in llm_rows) if llm_rows else None
    )
    return metrics


def _share(flags):
    flags = list(flags)
    return round(sum(flags) / len(flags), 3) if flags else None


def _print_retrieval(report, rows, settings):
    print("Retrieval (answerable questions)")
    for category, metrics in report["retrieval"].items():
        print(
            f"  {category:<20} n={metrics['count']:<3} hit@1={metrics['hit_at_1']:.2f}  "
            f"hit@{settings.top_k}={metrics['hit_at_k']:.2f}  MRR={metrics['mrr']:.2f}"
        )
    misses = [row for row in rows if row["category"] in ANSWERABLE and row["rank"] != 1]
    for row in misses:
        print(f"    rank={row['rank']}  {row['id']}: {row['question']}")

    sweep = report["threshold_sweep"]
    print("\nThreshold sweep (share of questions)")
    print("  threshold  answerable_kept  english_kept  offtopic_rejected  related_rejected")
    for entry in sweep["sweep"]:
        marker = "  <- current" if entry["threshold"] == round(settings.relevance_threshold, 2) else ""
        print(
            f"  {entry['threshold']:<9.2f}  {entry['answerable_kept']:<15.2f}  {entry['english_kept']:<12.2f}  "
            f"{entry['offtopic_rejected']:<17.2f}  {entry['related_rejected']:.2f}{marker}"
        )
    print(
        f"  lowest relevant score {sweep['lowest_relevant_score']:.3f}, "
        f"highest off-topic score {sweep['highest_offtopic_score']:.3f}"
    )
    if sweep["recommended_threshold"] is None:
        print("  no threshold separates answerable from off-topic questions")
    else:
        print(f"  recommended threshold (middle of the gap): {sweep['recommended_threshold']:.3f}")


def _print_answers(metrics, rows):
    print("\nAnswers")
    for category in ANSWERABLE + UNANSWERABLE:
        if category in metrics:
            values = metrics[category]
            details = ", ".join(f"{k}={v}" for k, v in values.items() if k not in ("count", "accuracy"))
            print(f"  {category:<22} n={values['count']:<3} accuracy={values['accuracy']:.2f}  ({details})")
    verification = metrics["verification"]
    print(
        f"  verification: {verification['answers_checked']} answers checked, "
        f"{verification['with_unsupported_numbers']} with unsupported numbers, "
        f"{verification['with_invalid_citations']} with invalid citations, "
        f"{verification['without_citations']} without citations"
    )
    print(f"  median seconds per LLM answer: {metrics['median_seconds_with_llm']}")

    failures = [row for row in rows if row.get("outcome") != "correct"]
    if failures:
        print("\nFailures")
        for row in failures:
            answer = row["answer"].replace("\n", " ")
            print(f"  [{row['outcome']}] {row['id']}: {row['question']}\n      -> {answer[:200]}")


if __name__ == "__main__":
    main()
