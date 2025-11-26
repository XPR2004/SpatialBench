#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
分析 benchmark_vision_base64.py 脚本输出的 JSON 结果文件，并计算各种维度的准确率。

描述:
    本脚本读取一个 JSON 格式的基准测试结果文件，并按以下维度进行分析：
    1.  模型的整体准确率。
    2.  五种主要问题类别（大类）的准确率。
    3.  十五种次要问题类别（小类）的准确率。
    4.  选择题与数值题的准确率。
    5.  室内与室外场景问题的正确率。

    最后，它会将模型名称和所有统计结果汇总到一个新的 JSON 文件中。

用法:
    python evaluate_benchmark_results.py ai_gen_sample_results_corrected
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
import re
import sys
import math

# --- 问题类别定义 ---

# 15 个小类到 5 个大类的映射
SUBCATEGORY_TO_MAJOR = {
    # Observation and Measurement
    "object_counting": "observation_and_measurement",
    "object_size": "observation_and_measurement",
    "room_size": "observation_and_measurement",
    "absolute_distance": "observation_and_measurement",
    # Topology and Composition
    "appearance_order": "topology_and_composition",
    "relative_distance": "topology_and_composition",
    "relative_direction": "topology_and_composition",
    "appearance_order_on_self-defined_route": "topology_and_composition",
    "relative_counting": "topology_and_composition",
    # Symbolic Visual Reasoning
    "multi-hop_spatial_reasoning": "symbolic_visual_reasoning",
    "affordance": "symbolic_visual_reasoning",
    "landmark-constrained_pose_localization": "symbolic_visual_reasoning",
    # Spatial Causality
    "spatial_causal_reasoning": "spatial_causality",
    # Spatial Planning
    "visual_based_commands": "spatial_planning",
    "route_plan": "spatial_planning",
}

# 新增：定义大类和小类的正确显示顺序
ORDERED_CATEGORIES = [
    {
        "major": "observation_and_measurement",
        "display_name": "Observation",
        "sub_categories": [
            ("object_counting", "Obj.count"),
            ("object_size", "Obj.Size"),
            ("room_size", "Room Size"),
            ("absolute_distance", "Abs. Distance")
        ]
    },
    {
        "major": "topology_and_composition",
        "display_name": "Topology & Relation",
        "sub_categories": [
            ("appearance_order", "App. Order"),
            ("appearance_order_on_self-defined_route", "App. Order (Self-Def-Route)"),
            ("relative_distance", "Rel. Distance"),
            ("relative_direction", "Rel. Direction"),
            ("relative_counting", "Rel. Count")
        ]
    },
    {
        "major": "symbolic_visual_reasoning",
        "display_name": "Symbolic Reasoning",
        "sub_categories": [
            ("multi-hop_spatial_reasoning", "Multi-Hop Reasoning"),
            ("affordance", "Affordance"),
            ("landmark-constrained_pose_localization", "Landmark Constrained Loc.")
        ]
    },
    {
        "major": "spatial_causality",
        "display_name": "Causality",
        "sub_categories": [
            ("spatial_causal_reasoning", "Causal Reasoning")
        ]
    },
    {
        "major": "spatial_planning",
        "display_name": "Planning",
        "sub_categories": [
            ("visual_based_commands", "Visual Based Commands"),
            ("route_plan", "Route Plan")
        ]
    }
]

# 从 ORDERED_CATEGORIES 动态生成
ALL_MAJOR_CATEGORIES = [cat["major"] for cat in ORDERED_CATEGORIES]
ALL_SUBCATEGORIES = [sub[0] for cat in ORDERED_CATEGORIES for sub in cat["sub_categories"]]

# --- 权重配置区 (最终版) ---
# 这是基于 V8 “优美解”模型，在 alpha=0.4, k=0.01 参数下求解得出的最优权重。
# 它在满足排序约束的前提下，平衡了“阶梯均匀性”、“阶梯强度”和“数据现实”。

# 最终混合权重 (W_i)
MAJOR_CATEGORY_WEIGHTS = {
    "observation_and_measurement": 0.0944,
    "topology_and_composition": 0.1564,
    "symbolic_visual_reasoning": 0.1759,
    "spatial_causality": 0.2592,
    "spatial_planning": 0.3141,
}

# --- 以下为脚本核心逻辑，已无需修改 ---

# 获取所有大类和小类的名称
# ALL_MAJOR_CATEGORIES = sorted(list(set(SUBCATEGORY_TO_MAJOR.values())))
# ALL_SUBCATEGORIES = sorted(list(SUBCATEGORY_TO_MAJOR.keys()))


def get_tiered_score(item: dict) -> float:
    """
    根据问题类型计算分数。
    - 'regression' 类型采用 MRA 算法。
    - 其他类型，如果 is_correct 为 True，则得 1.0 分，否则为 0.0 分。
    """
    problem_type = item.get("problem_type")
    
    # 对 'regression' 类型应用 MRA 算法
    if problem_type == "regression":
        try:
            model_ans = float(item.get("model_answer", ""))
            gt_ans_str = item.get("ground_truth", "")
            # 从 <answer>X</answer> 中提取数值
            gt_match = re.search(r"<answer>(.*?)</answer>", gt_ans_str, re.S | re.I)
            gt_text = gt_match.group(1).strip() if gt_match else gt_ans_str.strip()
            gt_ans = float(gt_text)

            if gt_ans == 0:
                # 如果真值为0，退化为绝对误差或简单判断
                return 1.0 if model_ans == 0 else 0.0

            relative_error = abs(model_ans - gt_ans) / abs(gt_ans)
            
            # 定义置信度阈值 C
            confidence_thresholds = [i / 100 for i in range(50, 100, 5)] # 0.5, 0.55, ..., 0.95
            
            total_accuracy = 0.0
            for theta in confidence_thresholds:
                if relative_error < (1 - theta):
                    total_accuracy += 1 # 在该阈值下是正确的

            # 计算平均相对准确率 (MRA)
            return total_accuracy / len(confidence_thresholds)

        except (ValueError, TypeError, ZeroDivisionError):
            # 如果答案无法转换，则按布尔值给分
            is_correct = item.get("is_correct", False)
            return 1.0 if is_correct else 0.0
    
    # 对于所有其他类型 (包括 multiple_choice)
    is_correct = item.get("is_correct", False)
    return 1.0 if is_correct else 0.0


def calculate_average_score(total_score: float, total_count: int) -> float:
    """安全地计算平均分，避免除零错误。"""
    if total_count == 0:
        return 0.0
    return round((total_score / total_count) * 100, 2)


def calculate_weighted_overall_score(major_category_scores: dict, weights: dict) -> float:
    """根据给定的权重计算加权总分。"""
    weighted_score = 0.0
    total_weight = 0.0 # 用于处理可能不完整的分数
    
    for category, score in major_category_scores.items():
        if category in weights:
            weighted_score += score * weights[category]
            total_weight += weights[category]
            
    # 如果总权重不为零，则进行归一化处理
    if total_weight > 0:
        # 重新归一化以防某些类别分数缺失
        return round(weighted_score / total_weight, 2)
    return 0.0


def analyze_results(results_data: list):
    """
    分析结果数据并计算所有维度的统计信息。
    """
    # 初始化用于计数的字典，将 'correct' 改为 'score'
    stats = {
        "overall": defaultdict(float),
        "major_category": {cat: defaultdict(float) for cat in ALL_MAJOR_CATEGORIES},
        "sub_category": {cat: defaultdict(float) for cat in ALL_SUBCATEGORIES},
        "problem_type": {
            "multiple_choice": defaultdict(float),
            "regression": defaultdict(float),
        },
        "scene_type": {
            "indoor": defaultdict(float),
            "outdoor": defaultdict(float),
        }
    }
    # 添加一个用于计数的并行字典
    counts = {
        "overall": 0,
        "major_category": defaultdict(int),
        "sub_category": defaultdict(int),
        "problem_type": defaultdict(int),
        "scene_type": defaultdict(int),
    }

    # 遍历每一条结果
    for item in results_data:
        score = get_tiered_score(item)
        
        # 1. 整体统计
        stats["overall"]["score"] += score
        counts["overall"] += 1

        # 2. 按小类和大类统计
        sub_category = item.get("original_question_type")
        if sub_category and sub_category in SUBCATEGORY_TO_MAJOR:
            major_category = SUBCATEGORY_TO_MAJOR[sub_category]
            
            stats["sub_category"][sub_category]["score"] += score
            stats["major_category"][major_category]["score"] += score
            counts["sub_category"][sub_category] += 1
            counts["major_category"][major_category] += 1

        # 3. 按问题类型统计 (选择题和数值题)
        problem_type = item.get("problem_type")
        if problem_type in stats["problem_type"]:
            stats["problem_type"][problem_type]["score"] += score
            counts["problem_type"][problem_type] += 1

        # 4. 按场景类型（室内/室外）统计
        scene_type = item.get("scene_type")
        if scene_type in stats["scene_type"]:
            stats["scene_type"][scene_type]["score"] += score
            counts["scene_type"][scene_type] += 1
    
    # 计算所有类别的加权准确率 (平均分)
    major_category_scores = {
        cat: calculate_average_score(data["score"], counts["major_category"][cat])
        for cat, data in stats["major_category"].items()
    }

    # --- 新增：按预设顺序重排字典 ---
    ordered_major_scores = {
        cat_info["major"]: major_category_scores.get(cat_info["major"], 0.0)
        for cat_info in ORDERED_CATEGORIES
    }
    
    # 构建一个包含所有小类分数的字典
    all_sub_scores = {
        cat: calculate_average_score(data["score"], counts["sub_category"][cat])
        for cat, data in stats["sub_category"].items()
    }

    # 按顺序重排小类分数
    ordered_sub_scores = {
        sub_key: all_sub_scores.get(sub_key, 0.0)
        for cat in ORDERED_CATEGORIES
        for sub_key, sub_display in cat["sub_categories"]
    }

    scores = {
        "overall_score": calculate_average_score(
            stats["overall"]["score"],
            counts["overall"]
        ),
        "weighted_overall_score": calculate_weighted_overall_score(
            major_category_scores, # 加权分计算仍使用原始数据，避免顺序影响
            MAJOR_CATEGORY_WEIGHTS
        ),
        "major_category_score": ordered_major_scores,
        "sub_category_score": ordered_sub_scores,
        "problem_type_score": {
            ptype: calculate_average_score(data["score"], counts["problem_type"][ptype])
            for ptype, data in stats["problem_type"].items()
        },
        "scene_type_score": {
            stype: calculate_average_score(data["score"], counts["scene_type"][stype])
            for stype, data in stats["scene_type"].items()
        },
    }
    
    return scores


def extract_model_name_from_filename(filename: str) -> str:
    """从 some-model_openai_results.json 中提取模型名称。"""
    # 正则表达式匹配 `_openai_results.json` 之前的部分
    match = re.search(r"^(.*?)_openai_results\.json$", filename)
    if match:
        return match.group(1).replace('_', '/') # 将下划线换回斜杠以还原模型名
    
    # 如果正则匹配失败，提供一个备用名称
    return Path(filename).stem.replace("_openai_results", "")


def parse_args():
    parser = argparse.ArgumentParser(
        description="分析视频问答基准测试的结果目录。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("input_path", type=str, help="包含 JSON 结果文件（例如 ai_gen_sample_results_corrected）的目录路径。")
    parser.add_argument(
        "-o", "--output", type=str, default="evaluation_summary.json",
        help="输出的总的 JSON 文件路径 (默认: evaluation_summary.json)。"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output)

    # 在开始处理前，打印最终使用的权重以供参考
    print("--- 本次运行最终使用的混合权重配置 ---")
    for task, weight in sorted(MAJOR_CATEGORY_WEIGHTS.items(), key=lambda item: item[1], reverse=True):
        print(f"- {task}: {weight:.3f}")
    print("--------------------------------------\n")

    if not input_path.is_dir():
        print(f"错误: 输入路径 '{input_path}' 不是一个有效的目录。")
        sys.exit(1)

    # 约束：最好是处理 ai_gen_sample 的结果目录
    if not input_path.name.startswith("ai_gen_sample_"):
        print(f"警告: 此脚本通常用于处理 'ai_gen_sample' 的结果目录。")
        print(f"      当前目录为: '{input_path.name}'，脚本将继续处理。")

    files_to_process = sorted(list(input_path.glob("*.json")))
    if not files_to_process:
        print(f"错误: 在目录 '{input_path}' 中未找到 JSON 文件。")
        return
        
    print(f"在目录 '{input_path.name}' 中找到 {len(files_to_process)} 个结果文件进行分析...")

    # 1. 读取现有的总报告（如果存在）
    all_reports = []
    if output_path.exists():
        try:
            summary_content = output_path.read_text(encoding="utf-8-sig", errors="replace")
            if summary_content:
                existing_data = json.loads(summary_content)
                if isinstance(existing_data, list):
                    all_reports = existing_data
                else:
                    print(f"警告: 现有报告文件 {output_path} 格式不正确（不是列表），将创建新报告。")
        except (json.JSONDecodeError, IOError) as e:
            print(f"警告: 无法读取或解析现有的报告文件 {output_path} ({e})。将创建一个新报告。")
    
    # 使用字典进行高效更新，键为 (model_name, source_dir)
    report_map = {(report.get("model_name"), report.get("source_dir")): report for report in all_reports}
    
    # 2. 遍历并处理目录中的每个文件
    for file_path in files_to_process:
        print(f"\n--- 正在分析: {file_path.name} ---")
        try:
            content = file_path.read_text(encoding="utf-8-sig", errors="replace")
            results_data = json.loads(content)
        except (json.JSONDecodeError, IOError) as e:
            print(f"错误: 读取或解析 JSON 文件 {file_path} 失败: {e}")
            continue

        # 预处理和筛选空回复
        original_total = len(results_data)
        valid_results = [item for item in results_data if item.get("model_raw_response", "").strip()]
        num_empty = original_total - len(valid_results)

        if num_empty > 0:
            print(f"信息: 在 {original_total} 条记录中检测到 {num_empty} 条空回复，已从准确率计算中排除。")
        
        # 分析数据并生成报告
        weighted_scores = analyze_results(valid_results)
        model_name = extract_model_name_from_filename(file_path.name)
        
        new_report = {
            "model_name": model_name,
            "source_dir": input_path.name, # 指明数据来源目录
            "total_valid_samples": len(valid_results),
            "total_empty_samples": num_empty,
            **weighted_scores
        }
        
        report_key = (model_name, input_path.name)
        if report_key in report_map:
            print(f"信息: 已更新模型 '{model_name}' 的报告 (来源: {input_path.name})。")
        else:
            print(f"信息: 已为新模型 '{model_name}' 添加报告 (来源: {input_path.name})。")
        report_map[report_key] = new_report

    # 3. 保存更新后的总报告
    final_reports = list(report_map.values())
    # 按新的加权总分降序排序
    final_reports.sort(key=lambda r: r.get("weighted_overall_score", 0), reverse=True)
    try:
        output_path.write_text(
            json.dumps(final_reports, indent=4, ensure_ascii=False),
            encoding="utf-8"
        )
        print(f"\n分析完成！汇总报告已更新/保存至: {output_path}")
    except IOError as e:
        print(f"错误: 写入报告到 {output_path} 失败: {e}")


if __name__ == "__main__":
    main()
