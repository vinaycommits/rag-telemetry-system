# evaluation/harness.py

import json
import re
import time
import statistics
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class QAPair:
    question_id:     str
    question:        str
    expected_answer: str
    answer_type:     str        # "exact" or "open_ended"
    tags:            list[str] = field(default_factory=list)


@dataclass
class EvalResult:
    question_id:      str
    question:         str
    expected:         str
    predicted:        str
    answer_type:      str
    score:            float
    judge_reasoning:  str  = ""
    retrieval_ms:     float = 0.0
    llm_ms:           float = 0.0
    tags:             list[str] = field(default_factory=list)

def exact_match(predicted: str, expected: str) -> float:
    pred = predicted.strip().lower()
    exp  = expected.strip().lower()

    # exact match
    if pred == exp:
        return 1.0

    # expected answer appears anywhere in predicted
    if exp in pred:
        return 1.0

    return 0.0


def llm_judge(
    question:  str,
    expected:  str,
    predicted: str,
    generator,
) -> tuple[float, str]:

    prompt = f"""You are a strict answer evaluator.

Question: {question}
Reference answer: {expected}
Predicted answer: {predicted}

Score the predicted answer from 1 to 5:
5 = fully correct and complete
4 = mostly correct, minor omissions
3 = partially correct
2 = mostly wrong but touches the topic
1 = completely wrong or hallucinated

Respond ONLY in this JSON format:
{{"score": <int 1-5>, "reasoning": "<one sentence>"}}"""

    response = generator.client.chat.completions.create(
        model       = generator.model_name,
        messages    = [{"role": "user", "content": prompt}],
        temperature = 0,
    )

    raw = response.choices[0].message.content.strip()

    try:
        clean  = re.sub(r"```json|```", "", raw).strip()
        parsed = json.loads(clean)
        score  = (parsed["score"] - 1) / 4
        return score, parsed.get("reasoning", "")
    except Exception:
        m = re.search(r'"score"\s*:\s*([1-5])', raw)
        score = (int(m.group(1)) - 1) / 4 if m else 0.0
        return score, raw
    
class EvalHarness:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def run(
        self,
        qa_pairs:    list[QAPair],
        output_path: str = "evaluation/results.jsonl",
    ) -> dict:

        Path("evaluation").mkdir(exist_ok=True)
        results: list[EvalResult] = []

        print(f"\nRunning eval on {len(qa_pairs)} questions...\n")

        for i, qa in enumerate(qa_pairs):
            print(f"[{i+1}/{len(qa_pairs)}] {qa.question[:60]}...")

            # Run full pipeline
            output = self.pipeline.query(qa.question)

            # Score
            if qa.answer_type == "exact":
                score     = exact_match(output["answer"], qa.expected_answer)
                reasoning = "exact match"
            else:
                score, reasoning = llm_judge(
                    question  = qa.question,
                    expected  = qa.expected_answer,
                    predicted = output["answer"],
                    generator = self.pipeline.generator,
                )

            result = EvalResult(
                question_id     = qa.question_id,
                question        = qa.question,
                expected        = qa.expected_answer,
                predicted       = output["answer"],
                answer_type     = qa.answer_type,
                score           = score,
                judge_reasoning = reasoning,
                retrieval_ms    = output["latency"]["retrieval_ms"],
                llm_ms          = output["latency"]["llm_ms"],
                tags            = qa.tags,
            )
            results.append(result)

             

        # Write results
        with open(output_path, "w") as f:
            for r in results:
                f.write(json.dumps(r.__dict__) + "\n")

        return self._summary(results)

    def _summary(self, results: list[EvalResult]) -> dict:
        scores     = [r.score for r in results]
        exact      = [r for r in results if r.answer_type == "exact"]
        open_ended = [r for r in results if r.answer_type == "open_ended"]

        # Per-tag breakdown
        tag_scores: dict[str, list[float]] = {}
        for r in results:
            for tag in r.tags:
                tag_scores.setdefault(tag, []).append(r.score)

        summary = {
            "total":              len(results),
            "overall_score":      round(statistics.mean(scores), 3),
            "exact_match_score":  round(statistics.mean(
                r.score for r in exact), 3) if exact else None,
            "llm_judge_score":    round(statistics.mean(
                r.score for r in open_ended), 3) if open_ended else None,
            "avg_retrieval_ms":   round(statistics.mean(
                r.retrieval_ms for r in results), 1),
            "failures":           [r.question_id for r in results if r.score < 0.4],
            "tag_breakdown":      {
                tag: round(statistics.mean(s), 3)
                for tag, s in tag_scores.items()
            },
        }

        print("\n===== EVAL SUMMARY =====")
        for k, v in summary.items():
            print(f"  {k}: {v}")

        return summary