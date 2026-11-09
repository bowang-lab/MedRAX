"""ChestAgentBench loading, case-level splits and grading for ACE experiments.

Dataset: https://huggingface.co/datasets/wanglab/chest-agent-bench
Expected layout (``data_dir``)::

    data_dir/
      metadata.jsonl          # one question per line
      figures/<case_id>/...   # unzipped figures.zip

Each metadata row has: question (with A-F options), answer (letter), explanation, case_id,
question_id, images (paths relative to data_dir), image_source_urls, categories
(comma-separated string), sections, type.
"""

import json
import os
import random
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional

LETTERS = "ABCDEF"
_OPTION_RE = re.compile(r"^([A-F])\)\s*(.*)$")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_metadata(data_dir: str) -> List[Dict[str, Any]]:
    """Load ChestAgentBench rows from ``data_dir/metadata.jsonl``."""
    path = os.path.join(data_dir, "metadata.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Download the benchmark with:\n"
            "  huggingface-cli download wanglab/chest-agent-bench --repo-type dataset "
            f"--local-dir {data_dir}\n"
            f"  unzip {os.path.join(data_dir, 'figures.zip')} -d {data_dir}"
        )
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_categories(row: Dict[str, Any]) -> List[str]:
    cats = row.get("categories", "")
    if isinstance(cats, list):
        return [c.strip() for c in cats if c.strip()]
    return [c.strip() for c in str(cats).split(",") if c.strip()]


# ---------------------------------------------------------------------------
# Splits (by case, so questions sharing images never cross splits)
# ---------------------------------------------------------------------------


def make_case_splits(
    rows: List[Dict[str, Any]],
    n_train_cases: int,
    n_val_cases: int,
    seed: int = 42,
    n_test_cases: Optional[int] = None,
) -> Dict[str, Any]:
    """Split case IDs into train/val/test. Remaining cases go to test unless n_test_cases is set."""
    case_ids = sorted({str(r["case_id"]) for r in rows})
    if n_train_cases + n_val_cases >= len(case_ids):
        raise ValueError(
            f"n_train_cases + n_val_cases ({n_train_cases + n_val_cases}) must be smaller than "
            f"the number of cases ({len(case_ids)})"
        )
    rng = random.Random(seed)
    rng.shuffle(case_ids)
    train = case_ids[:n_train_cases]
    val = case_ids[n_train_cases : n_train_cases + n_val_cases]
    test = case_ids[n_train_cases + n_val_cases :]
    if n_test_cases is not None:
        test = test[:n_test_cases]
    return {
        "seed": seed,
        "n_cases_total": len(case_ids),
        "train": sorted(train),
        "val": sorted(val),
        "test": sorted(test),
    }


def load_or_create_splits(
    split_file: str,
    rows: List[Dict[str, Any]],
    n_train_cases: int,
    n_val_cases: int,
    seed: int,
) -> Dict[str, Any]:
    """Load a frozen split file, or create and save one if it does not exist yet.

    Freezing the split in a file (committed to git) keeps every experiment on the same
    train/val/test cases.
    """
    if os.path.exists(split_file):
        with open(split_file, "r") as f:
            return json.load(f)
    splits = make_case_splits(rows, n_train_cases, n_val_cases, seed=seed)
    os.makedirs(os.path.dirname(os.path.abspath(split_file)), exist_ok=True)
    with open(split_file, "w") as f:
        json.dump(splits, f, indent=2)
    print(
        f"Created split file {split_file}: {len(splits['train'])} train / "
        f"{len(splits['val'])} val / {len(splits['test'])} test cases"
    )
    return splits


def rows_for_cases(rows: List[Dict[str, Any]], case_ids: List[str]) -> List[Dict[str, Any]]:
    wanted = set(str(c) for c in case_ids)
    return [r for r in rows if str(r["case_id"]) in wanted]


# ---------------------------------------------------------------------------
# Option shuffling (guards against learning answer-letter priors)
# ---------------------------------------------------------------------------


def shuffle_options(question: str, answer: str, seed_key: str):
    """Deterministically permute the A-F options of a question.

    ChestAgentBench answers are skewed (B ~43%, C ~30%), so a learned playbook could pick up
    letter priors ("when unsure, choose B"). Shuffling removes that shortcut. Returns
    ``(question, answer, permutation)``; if the options cannot be parsed reliably the
    question is returned unchanged.
    """
    lines = question.split("\n")
    option_idx = [i for i, line in enumerate(lines) if _OPTION_RE.match(line.strip())]
    if len(option_idx) < 2 or option_idx != list(range(option_idx[0], option_idx[-1] + 1)):
        return question, answer, None
    options = [_OPTION_RE.match(lines[i].strip()) for i in option_idx]
    letters = [m.group(1) for m in options]
    if letters != list(LETTERS[: len(letters)]) or answer not in letters:
        return question, answer, None
    texts = [m.group(2) for m in options]

    order = list(range(len(texts)))
    random.Random(seed_key).shuffle(order)
    new_lines = list(lines)
    for new_pos, old_pos in enumerate(order):
        new_lines[option_idx[new_pos]] = f"{LETTERS[new_pos]}) {texts[old_pos]}"
    new_answer = LETTERS[order.index(letters.index(answer))]
    return "\n".join(new_lines), new_answer, order


# ---------------------------------------------------------------------------
# Answer grading
# ---------------------------------------------------------------------------


def extract_letter(answer: Any) -> str:
    """Extract an A-F choice from a model answer ("B", "B)", "(B) foo", "Answer: B", ...).

    Stricter than experiments/compare_runs.py's last-resort "any A-F letter": an answer
    with no identifiable choice returns "" and is graded incorrect.
    """
    if answer is None:
        return ""
    text = str(answer).strip()
    if len(text) == 1 and text.upper() in LETTERS:
        return text.upper()
    # Letters are matched case-sensitively so prose such as "A patient ..." or
    # "the answer is a lesion" is not mistaken for a choice.
    patterns = [
        r"^\(?([A-F])(?:\)|\.|:|$)",  # "B", "B)", "(B) ...", "B. ...", "B: ..."
        r"(?i:answer|option|choice)\s*(?:is|:)?\s*\(?([A-F])(?:\)|\.|:|\s|$)",
        r"(?:^|\s)\(([A-F])\)",  # "... (B) ..."
    ]
    for pattern in patterns:
        m = re.search(pattern, text)
        if m:
            return m.group(1)
    return ""


# ---------------------------------------------------------------------------
# ACE DataProcessor
# ---------------------------------------------------------------------------


class ChestAgentBenchProcessor:
    """ACE DataProcessor for ChestAgentBench.

    Sample format consumed by ACE: ``{"question", "context", "target", "others"}``.
    ``context`` is a JSON string with the question, image paths and IDs; ACE passes it to the
    generator (MedRAXGenerator decodes it) and to the Curator. It never contains the answer.
    """

    def __init__(
        self,
        data_dir: str,
        use_explanation_feedback: bool = True,
        shuffle_answer_options: bool = False,
    ):
        self.data_dir = os.path.abspath(data_dir)
        self.use_explanation_feedback = use_explanation_feedback
        self.shuffle_answer_options = shuffle_answer_options

    # -- ACE interface ------------------------------------------------------

    def process_task_data(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        samples, skipped = [], defaultdict(int)
        for row in rows:
            image_paths = [
                os.path.normpath(os.path.join(self.data_dir, p))
                for p in (row.get("images") or [])
            ]
            if not image_paths:
                skipped["no_images"] += 1
                continue

            question, answer, permutation = row["question"], row["answer"].strip(), None
            if self.shuffle_answer_options:
                question, answer, permutation = shuffle_options(
                    question, answer, seed_key=str(row["question_id"])
                )

            # ACE also shows `context` to the Curator as "Question Context", so it carries the
            # question text; MedRAXGenerator only reads the IDs and image paths from it.
            context = {
                "case_id": str(row["case_id"]),
                "question_id": str(row["question_id"]),
                "question": question,
                "image_paths": image_paths,
            }
            samples.append(
                {
                    "question": question,
                    "context": json.dumps(context),
                    "target": answer,
                    "others": {
                        "case_id": str(row["case_id"]),
                        "question_id": str(row["question_id"]),
                        "explanation": row.get("explanation", ""),
                        "categories": parse_categories(row),
                        "original_answer": row["answer"].strip(),
                        "option_permutation": permutation,
                        "image_source_urls": row.get("image_source_urls", []),
                    },
                }
            )
        for reason, n in skipped.items():
            print(f"Skipped {n} questions ({reason})")
        return samples

    def answer_is_correct(self, predicted: str, ground_truth: str) -> bool:
        return extract_letter(predicted) == ground_truth.strip().upper()

    def evaluate_accuracy(self, predictions: List[str], ground_truths: List[str]) -> float:
        if not predictions:
            return 0.0
        correct = sum(self.answer_is_correct(p, g) for p, g in zip(predictions, ground_truths))
        return correct / len(predictions)

    # -- Hook used by the vendored ACE (see experiments/ace/NOTICE.md) -------

    def environment_feedback(self, task_dict: Dict[str, Any], is_correct: bool, default: str) -> str:
        if not self.use_explanation_feedback:
            return default
        explanation = (task_dict.get("others") or {}).get("explanation", "")
        if not explanation:
            return default
        return f"{default}.\nExpert explanation of the correct answer: {explanation}"
