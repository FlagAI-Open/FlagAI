import argparse
from human_eval.evaluation import evaluate_functional_correctness


def entry_point(
    question_file: str,
    sample_file: str,
    k: str = "1,10,100",
    n_workers: int = 4,
    timeout: float = 3.0,
):
    """
    Evaluates the functional correctness of generated samples, and writes
    results to f"{sample_file}_results.jsonl.gz"
    """
    k = list(map(int, k.split(",")))
    results = evaluate_functional_correctness(
        sample_file, k, n_workers, timeout, question_file
    )

    return results

def main(args):
    print('test correctness of file:', args.sample_file)
    result = entry_point(
        question_file=args.question_file,
        sample_file=args.sample_file,
    )

    print({k: float(v) for k, v in result.items()})

if __name__ == "__main__":
    args = argparse.ArgumentParser()

    args.add_argument(
        "--question-file",
        type=str,
        default='data/human_eval/question.jsonl'
    )

    args.add_argument(
        "--sample-file",
        type=str,
        default='data/human_eval/model_answer/llama-3-8b-instruct/baseline_correctness.jsonl'
    )


    args = args.parse_args()
    main(args)
