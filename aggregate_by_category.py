"""실험 결과 집계: results/<run>/ep<N>/<pkl_name>/seed0/open_test_results_open_others_msp.csv 를 모아
설정별 지표, 카테고리별 평균, 전체 평균을 하나의 CSV 로 저장한다.

    python aggregate_by_category.py --results_dir results/local_prompt_fewshot16/ep100
"""
import argparse
import os

import pandas as pd

METRICS = ['Acc_Known', 'AUROC', 'Best_F1', 'Unknown_Recall', 'Known_Recall', 'Best_Acc']
TARGET_CSV = "open_test_results_open_others_msp.csv"


def find_result_csv(exp_path, exp_name):
    """결과 CSV 위치 후보: <exp>/<exp>/, <exp>/seed0/, <exp>/ 순으로 탐색"""
    for cand in (os.path.join(exp_path, exp_name, TARGET_CSV),
                 os.path.join(exp_path, 'seed0', TARGET_CSV),
                 os.path.join(exp_path, TARGET_CSV)):
        if os.path.isfile(cand):
            return cand
    return None


def aggregate_by_category(base_dir, output_file):
    if not os.path.isdir(base_dir):
        print(f"Error: results directory not found: {base_dir}")
        return

    rows = []
    for exp_name in sorted(os.listdir(base_dir)):
        exp_path = os.path.join(base_dir, exp_name)
        if not os.path.isdir(exp_path):
            continue
        csv_path = find_result_csv(exp_path, exp_name)
        if csv_path is None:
            print(f"Warning: CSV not found in {exp_path}")
            continue
        df = pd.read_csv(csv_path)
        available = [m for m in METRICS if m in df.columns]
        if not available:
            print(f"Warning: no metric columns in {csv_path}")
            continue
        row = df[available].iloc[0].to_dict()
        row['Experiment'] = exp_name
        row['Category'] = exp_name.split('_')[0]      # 예) bottle_seed0_unknown_contamination -> bottle
        rows.append(row)

    if not rows:
        print("No data found.")
        return

    full_df = pd.DataFrame(rows)
    for m in METRICS:
        if m not in full_df.columns:
            full_df[m] = 0.0
    full_df = full_df[['Experiment', 'Category'] + METRICS]

    category_means = full_df.groupby('Category')[METRICS].mean().reset_index()
    category_means.insert(0, 'Experiment', category_means['Category'])

    overall = full_df[METRICS].mean().to_dict()
    overall.update(Experiment='All', Category='All')
    overall_df = pd.DataFrame([overall])[['Experiment', 'Category'] + METRICS]

    result_df = pd.concat([full_df, category_means, overall_df], ignore_index=True)
    result_df.to_csv(output_file, index=False)
    print(f"Saved to {output_file}\n")

    display_cols = ['Experiment'] + METRICS
    print("=== Category-level averages ===")
    print(category_means[display_cols].to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print("\n=== Overall average ===")
    print(overall_df[display_cols].to_string(index=False, float_format=lambda v: f"{v:.2f}"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results_dir", required=True, help="예: results/local_prompt_fewshot16/ep100")
    parser.add_argument("--output", default=None, help="저장 경로 (기본: <results_dir>/aggregated_by_category.csv)")
    args = parser.parse_args()
    output = args.output or os.path.join(args.results_dir, "aggregated_by_category.csv")
    aggregate_by_category(args.results_dir, output)
