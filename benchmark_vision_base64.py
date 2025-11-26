#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Benchmark script for Video Question Answering tasks using Base64 encoded video frames (OpenAI-Compatible API)

Description:
    This script reads a JSON (.txt) file containing video question-answer pairs.
    It performs the following steps:
    1. Extracts frames from each video using OpenCV.
    2. Encodes extracted frames as Base64 strings.
    3. Concurrently submits image data and questions to an OpenAI-compatible API endpoint (e.g., a proxy).
    4. Aggregates results, costs, and accuracy, and outputs them to a JSON file.
    
    This version is adapted for environments requiring Gemini or other models via OpenAI format.

Usage:
    # 1. Install dependencies:
    pip install openai opencv-python numpy tqdm

    # 2. Set environment variables:
    # Linux/MacOS:
    export OPENAI_API_KEY="sk-..."
    export OPENAI_API_BASE="https://api.openai-proxy.org/v1"
    
    # Windows (PowerShell):
    $env:OPENAI_API_KEY="sk-..."
    $env:OPENAI_API_BASE="https://api.openai-proxy.org/v1"

    # 3. Run the script:
    
    # Basic usage (defaults to QA.txt):
    python benchmark_vision_base64.py

    # Specify input file:
    python benchmark_vision_base64.py QA.txt

    # Specify model (defaults to Qwen2.5-VL-72B-Instruct):
    python benchmark_vision_base64.py -m "gpt-4o"

    # Set concurrency (workers):
    python benchmark_vision_base64.py -w 8

    # Resume from interruption (skips completed questions in output file):
    python benchmark_vision_base64.py --resume

    # Override maximum number of frames extracted:
    python benchmark_vision_base64.py --max-frames 128

    # Deep Guide Mode (Video Examples):
    # Automatically activated if input file is "QA_fewshot.txt".
    python benchmark_vision_base64.py QA_fewshot.txt

    # Few-Shot Mode (Text Examples):
    python benchmark_vision_base64.py --few-shot

    # Test a specific problem ID:
    python benchmark_vision_base64.py --test-id 1001

    # Show reasoning process (only with --test-id or --with-reasoning):
    python benchmark_vision_base64.py --test-id 1001 --show-reasoning
"""

import os
import re
import json
import time
import random
import argparse
import base64
import math
import pprint
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from string import Template
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict # Added for simulate_random_choice_answers

# --- Dependency Check ---
try:
    import cv2
    import numpy as np
except ImportError:
    print("Error: Missing 'opencv-python' or 'numpy' library.")
    print("Please run: pip install opencv-python numpy")
    exit(1)

try:
    import httpx
except ImportError:
    print("Error: Missing 'httpx' library.")
    print("Please run: pip install httpx")
    exit(1)

try:
    from openai import OpenAI
except ImportError:
    print("Error: Missing 'openai' library.")
    print("Please run: pip install openai")
    exit(1)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        count = len(iterable) if hasattr(iterable, '__len__') else '...'
        print(f"Info: Processing {count} items (tqdm not installed)...")
        return iterable

# -------------------- Configuration --------------------

# OpenAI-Compatible API Configuration
_api_key_raw = os.getenv("OPENAI_API_KEY")
_api_base_raw = os.getenv("OPENAI_API_BASE")

# Strip quotes that might be included from Windows `set VAR="value"`
OPENAI_API_KEY = _api_key_raw.strip().strip('"') if _api_key_raw else None
OPENAI_API_BASE = _api_base_raw.strip().strip('"') if _api_base_raw else None

if not OPENAI_API_KEY:
    raise RuntimeError("Error: Environment variable 'OPENAI_API_KEY' not found.")
if not OPENAI_API_BASE:
    raise RuntimeError("Error: Environment variable 'OPENAI_API_BASE' not found. Please set your proxy API address.")

# Model Configuration
# Default model
DEFAULT_MODEL = "Qwen2.5-VL-72B-Instruct"

# --- Model Generation Parameters ---
DEFAULT_GEN_CONFIG = {
    'temperature': 0.1,
    'top_p': 0.9,
    'max_tokens': 1024,
}

# --- Video Frame Extraction Configuration (Adapted from user logic) ---
JPEG_QUALITY: int = int(os.getenv("JPEG_QUALITY", "85"))
# New: Compression quality for exemplar frames in Deep Guide mode
EXEMPLAR_JPEG_QUALITY: int = int(os.getenv("EXEMPLAR_JPEG_QUALITY", "30"))

# --- Regression Problem Tolerance ---
# Used to determine if the answer to a numerical regression problem is correct, default 5%
REGRESSION_REL_TOL = float(os.getenv("REGRESSION_REL_TOL", "0.05"))

# API Call Retries
GEN_RETRIES     = int(os.getenv("GEN_RETRIES", "6"))
GEN_BASE_DELAY  = float(os.getenv("GEN_BASE_DELAY", "1.0"))

# QA Concurrency
MAX_QA_WORKERS = int(os.getenv("MAX_QA_WORKERS", "4"))


# --- Formatting & Costing Tools ---

def _fmt_dur(t0: float, t1: float) -> str:
    return f"{(t1 - t0):.2f}s"

def _extract_usage(resp) -> Tuple[int, int]:
    # Extract token usage from OpenAI response object
    if resp and hasattr(resp, 'usage'):
        usage = resp.usage
        return getattr(usage, 'prompt_tokens', 0), getattr(usage, 'completion_tokens', 0)
    return 0, 0

def _resp_text(resp) -> str:
    # Extract model returned text from OpenAI response object
    if resp and hasattr(resp, 'choices') and resp.choices:
        message = resp.choices[0].message
        return getattr(message, 'content', '') or ''
    return ""

# --- Video Processing (Adapted from user logic) ---
def extract_video_frames(
    video_path: str, 
    model_name: str, 
    keyframe_indices: Optional[List[int]] = None,
    override_jpeg_quality: Optional[int] = None,
    override_max_frames: Optional[int] = None
) -> List[str]:
    """
    Extracts frames from a video file at a rate of 1 frame per second and encodes them as Base64 strings.
    Dynamically adjusts JPEG compression quality and frame count based on the model name.
    New: Supports extracting only specified keyframes.
    New: Supports overriding maximum frame limit.
    """
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Unable to open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25 
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    frame_indices = []
    local_jpeg_quality = override_jpeg_quality if override_jpeg_quality is not None else JPEG_QUALITY
    
    # --- New: Keyframe priority logic ---
    if keyframe_indices:
        print(f"[INFO] Keyframe Mode: Extracting {len(keyframe_indices)} specified frames. Quality -> {local_jpeg_quality}")
        frame_indices = [idx for idx in keyframe_indices if 0 <= idx < total_frames]
    
    else:
        # Determine max_frames based on model or override
        if override_max_frames is not None:
             max_frames = override_max_frames
             print(f"[INFO] Manually overriding max frames: {max_frames}")
        else:
            # User requested default 64 frames for all models
            max_frames = 64
            
        # Adjust quality for specific models if not overridden
        if override_jpeg_quality is None:
             if "glm" in model_name.lower(): local_jpeg_quality = 40
             elif "ernie" in model_name.lower(): local_jpeg_quality = 30
        
        # Calculate sample logic
        num_frames_to_sample = min(int(duration), max_frames)
        num_frames_to_sample = min(num_frames_to_sample, total_frames)
        
        print(f"[INFO] Model ({model_name}): Quality -> {local_jpeg_quality}, Max Frames -> {num_frames_to_sample} (Max Limit: {max_frames})")

        if num_frames_to_sample > 0:
            if num_frames_to_sample == 1:
                frame_indices = [0]
            else:
                step = (total_frames - 1) / (num_frames_to_sample - 1)
                frame_indices = [int(round(i * step)) for i in range(num_frames_to_sample)]
        else:
            frame_indices = []

    base64_frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        
        _, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), local_jpeg_quality])
        base64_frames.append(base64.b64encode(buffer).decode("utf-8"))

    cap.release()
    return base64_frames

# --- API Helpers ---
def _build_openai_messages(prompt_text: str, base64_frames: Optional[List[str]] = None, history: Optional[List] = None):
    """Build OpenAI formatted message list"""
    if history:
        # Multi-turn conversation, only add new user prompt
        new_messages = history + [{"role": "user", "content": prompt_text}]
        return new_messages

    # First turn, include images
    content = [{"type": "text", "text": prompt_text}]
    if base64_frames:
        for b64 in base64_frames:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
            })
    return [{"role": "user", "content": content}]

def _build_deep_guide_messages(
    prompt_template: Template,
    exemplar: Dict[str, Any],
    problem: Dict[str, Any],
    exemplar_frames: List[str],
    problem_frames: List[str]
) -> List[Dict[str, Any]]:
    """Build OpenAI formatted message list for Deep Guide mode"""
    
    # Prepare exemplar text
    exemplar_options_text = ""
    if exemplar.get("problem_type") == "multiple_choice" and exemplar.get("options"):
        exemplar_options_text = "Exemplar Options:\n" + "\n".join(exemplar["options"])
    
    # Prepare current problem text
    problem_options_text = ""
    if problem.get("problem_type") == "multiple_choice" and problem.get("options"):
        problem_options_text = "Options:\n" + "\n".join(problem["options"])
        
    prompt_str = prompt_template.substitute(
        problem_type=exemplar.get("problem_type", "N/A"),
        exemplar_problem_text=exemplar.get("problem", "N/A"),
        exemplar_options_text=exemplar_options_text,
        exemplar_reason=exemplar.get("reason", "N/A"),
        exemplar_solution=exemplar.get("solution", "N/A"),
        current_problem_text=problem.get("problem", "N/A"),
        current_options_text=problem_options_text
    )

    # Build content list
    content = []
    # 1. Opening guide text
    content.append({"type": "text", "text": "### BEGIN EXAMPLE ###"})
    # 2. Exemplar images
    for b64 in exemplar_frames:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    # 3. Exemplar problem and current problem text
    content.append({"type": "text", "text": prompt_str})
    # 4. Current problem images
    for b64 in problem_frames:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    
    return [{"role": "user", "content": content}]


def _safe_openai_call(client, model_name, messages, gen_config):
    """OpenAI API call wrapper with retry logic"""
    api_call_func = client.chat.completions.create
    last_err = None

    # --- Model-specific parameter handling ---
    # Models like Claude do not support specifying both temperature and top_p.
    # We prioritize temperature.
    api_params = {
        "model": model_name,
        "messages": messages,
        "temperature": gen_config['temperature'],
    }
    # Only pass max_tokens if explicitly present in gen_config
    if 'max_tokens' in gen_config:
        api_params['max_tokens'] = gen_config['max_tokens']

    if "claude" not in model_name.lower():
        api_params["top_p"] = gen_config['top_p']

    for attempt in range(1, GEN_RETRIES + 1):
        try:
            return api_call_func(**api_params)
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            transient = (
                "timeout" in msg or "connection" in msg or "overloaded" in msg or
                "503" in msg or "502" in msg or "gateway" in msg or
                "resource_exhausted" in msg
            )
            if attempt < GEN_RETRIES and transient:
                sleep_s = GEN_BASE_DELAY * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
                print(f"[RETRY] API call failed (Attempt {attempt}/{GEN_RETRIES}): {e}\n        -> Sleeping {sleep_s:.2f}s before retry")
                time.sleep(sleep_s)
                continue
            raise
    raise last_err

# -------------------- Task Specific Logic --------------------

# --- Prompt 模板 ---
DEEP_GUIDE_PROMPT_TEMPLATE = Template("""
This is an example of how to solve a '${problem_type}' problem.

Exemplar Question: ${exemplar_problem_text}
${exemplar_options_text}

Correct Reasoning Process: ${exemplar_reason}
Final Correct Answer: ${exemplar_solution}

### END EXAMPLE ###

Now, based on the new set of images provided, solve the following new problem.

---

Question: ${current_problem_text}
${current_options_text}

YOUR TASK IS TO PROVIDE ONLY THE FINAL ANSWER.
DO NOT INCLUDE ANY EXPLANATIONS, REASONING, OR THOUGHT PROCESS.
YOUR RESPONSE MUST BE EXTREMELY CONCISE AND CONTAIN ONLY THE ANSWER.

Desired Answer Format:
- For a Multiple choice question, your entire response must be a single letter (e.g., A).
- For a Regression question, your entire response must be a single number (e.g., 240).

Provide the final answer ONLY.
""".strip())

FEW_SHOT_EXAMPLE_TEMPLATE = Template("""
Here is an example of how to solve a problem of type '${problem_type}'. Please follow this reasoning process.

--- BEGIN EXAMPLE ---
Question: ${problem_text}
${options_text}
Correct Reasoning: ${reason}
Final Answer: ${solution}
--- END EXAMPLE ---

Now, based on the video frames provided, solve the following new problem.
""".strip())

SYSTEM_PROMPT_TEMPLATE = Template("""
${few_shot_block}
Analyze the video frames and answer the question.

Question type: ${problem_type}
Question: ${problem_text}
${options_text}

YOUR TASK IS TO PROVIDE ONLY THE FINAL ANSWER.
DO NOT INCLUDE ANY EXPLANATIONS, REASONING, OR THOUGHT PROCESS.
YOUR RESPONSE MUST BE EXTREMELY CONCISE AND CONTAIN ONLY THE ANSWER.

Desired Answer Format:
- For a Multiple choice question, your entire response must be a single letter (e.g., A).
- For a Regression question, your entire response must be a single number (e.g., 240).

Provide the final answer ONLY.
""".strip())

REASONING_SYSTEM_PROMPT_TEMPLATE = Template("""
${few_shot_block}
Analyze the video frames and answer the question. Your primary task is to provide a detailed, step-by-step reasoning process that explains how you arrived at your conclusion. After your reasoning, provide the final answer in the specified format.

Question type: ${problem_type}
Question: ${problem_text}
${options_text}

YOUR TASK:
1.  First, provide a clear, logical, step-by-step "Reasoning" process.
2.  After the reasoning, provide the "Final Answer".

Desired Response Format:
Reasoning:
<Your detailed thought process here>

Final Answer:
<A single letter for multiple choice (e.g., A) or a single number for regression (e.g., 240)>
""".strip())

REASON_PROMPT_BLOCK = Template("""
Here is a reasoning process to guide your thinking, please refer to it to come up with the final answer.
Reasoning: ${reason}
""".strip())

def _parse_gt_solution(solution_str: str) -> str:
    """Extract 'A' from <answer>A</answer>"""
    match = re.search(r"<answer>(.*?)</answer>", solution_str, re.S | re.I)
    return match.group(1).strip() if match else solution_str.strip()

def _clean_model_answer(raw_text: str, problem_type: str, options: Optional[List[str]] = None, reasoning_mode: bool = False) -> str:
    """
    Cleans the raw model response to extract a concise answer.
    """
    if not raw_text:
        return ""
        
    clean_text = raw_text.strip()
    
    # --- Reasoning Mode Handling ---
    if reasoning_mode:
        # In reasoning mode, prioritize looking after "Final Answer:"
        match = re.search(r"Final Answer:\s*(.*)", clean_text, re.IGNORECASE | re.DOTALL)
        if match:
            clean_text = match.group(1).strip()
    
    # --- GLM Special Format ---
    glm_match = re.search(r"<\|begin_of_box\|>(.*?)<\|end_of_box\|>", clean_text, re.DOTALL)
    if glm_match:
        return glm_match.group(1).strip()
    
    norm_problem_type = problem_type.replace("_", " ")

    if "multiple choice" in norm_problem_type:
        # 1. Primary method: Find a single capital letter A-D at the end
        # Look for a letter at the very end of the string, ignoring surrounding non-alphanumeric characters.
        match = re.search(r'[^A-Z0-9a-z]*([A-D])[^A-Z0-9a-z]*\s*$', clean_text, re.I)
        if match:
            return match.group(1).upper()

        # 2. Look for letter at beginning
        match = re.match(r"^\s*[^A-Z0-9a-z]*([A-D])", clean_text, re.I)
        if match:
            return match.group(1).upper()
            
        # 3. Fallback: Match option text if options provided
        if options:
            lines = [line.strip() for line in clean_text.strip().split('\n')]
            last_non_empty_line = ""
            for line in reversed(lines):
                if line:
                    last_non_empty_line = line
                    break
            
            if last_non_empty_line:
                for option_str in options:
                    option_match = re.match(r"^\s*([A-D])\.\s*(.*?)\s*$", option_str)
                    if option_match:
                        letter = option_match.group(1)
                        text = option_match.group(2).strip('. ')
                        if re.search(r'\b' + re.escape(text) + r'\b', last_non_empty_line, re.IGNORECASE):
                            return letter.upper()

    elif "regression" in norm_problem_type or "object counting" in norm_problem_type:
        all_numbers = re.findall(r"\b(\d+(?:\.\d+)?)\b", clean_text)
        if all_numbers:
            return all_numbers[-1]
            
    return clean_text.rstrip(".").strip()

def _check_correctness(model_ans: str, gt_solution: str, problem_type: str) -> bool:
    """
    Check if the model's answer is correct.
    """
    gt_text = _parse_gt_solution(gt_solution)
    norm_problem_type = problem_type.replace("_", " ")

    if "regression" in norm_problem_type:
        try:
            model_num = float(model_ans)
            gt_num = float(gt_text)
            return math.isclose(model_num, gt_num, rel_tol=REGRESSION_REL_TOL)
        except (ValueError, TypeError):
            return model_ans.lower() == gt_text.lower()
    
    elif "object counting" in norm_problem_type:
        try:
            model_num = float(model_ans)
            gt_num = float(gt_text)
            return model_num == gt_num and model_num == int(model_num)
        except (ValueError, TypeError):
            return False
            
    else:
        return model_ans.lower() == gt_text.lower()


# --- Categories, Weights, and Analysis Logic Imported from evaluation_summary.py ---
SUBCATEGORY_TO_MAJOR = {
    "object_counting": "observation_and_measurement",
    "object_size": "observation_and_measurement",
    "room_size": "observation_and_measurement",
    "absolute_distance": "observation_and_measurement",
    "appearance_order": "topology_and_composition",
    "relative_distance": "topology_and_composition",
    "relative_direction": "topology_and_composition",
    "appearance_order_on_self-defined_route": "topology_and_composition",
    "relative_counting": "topology_and_composition",
    "multi-hop_spatial_reasoning": "symbolic_visual_reasoning",
    "affordance": "symbolic_visual_reasoning",
    "landmark-constrained_pose_localization": "symbolic_visual_reasoning",
    "spatial_causal_reasoning": "spatial_causality",
    "visual_based_commands": "spatial_planning",
    "route_plan": "spatial_planning",
}
MAJOR_CATEGORY_WEIGHTS = {
    "observation_and_measurement": 0.0944,
    "topology_and_composition": 0.1564,
    "symbolic_visual_reasoning": 0.1759,
    "spatial_causality": 0.2592,
    "spatial_planning": 0.3141,
}
ALL_MAJOR_CATEGORIES = sorted(list(set(SUBCATEGORY_TO_MAJOR.values())))
ALL_SUBCATEGORIES = sorted(list(SUBCATEGORY_TO_MAJOR.keys()))

def _sim_get_score(item: dict) -> float:
    # In simulation, we only care about the is_correct field
    return 1.0 if item.get("is_correct", False) else 0.0

def _sim_calculate_avg_score(total_score: float, total_count: int) -> float:
    if total_count == 0: return 0.0
    return (total_score / total_count) * 100

def _sim_calculate_weighted_score(major_scores: dict, weights: dict) -> float:
    score = sum(major_scores.get(cat, 0) * w for cat, w in weights.items())
    total_w = sum(weights[cat] for cat, s in major_scores.items() if cat in weights and s > 0)
    return score / total_w if total_w > 0 else 0.0

def analyze_simulation_results(results_data: list):
    stats = {
        "major_category": defaultdict(float), "sub_category": defaultdict(float),
        "scene_type": defaultdict(float), "overall": 0.0
    }
    counts = {
        "major_category": defaultdict(int), "sub_category": defaultdict(int),
        "scene_type": defaultdict(int), "overall": 0
    }

    for item in results_data:
        score = _sim_get_score(item)
        sub_cat = item.get("original_question_type")
        major_cat = SUBCATEGORY_TO_MAJOR.get(sub_cat)
        scene_type = item.get("scene_type")

        stats["overall"] += score
        counts["overall"] += 1
        if major_cat:
            stats["major_category"][major_cat] += score
            counts["major_category"][major_cat] += 1
        if sub_cat:
            stats["sub_category"][sub_cat] += score
            counts["sub_category"][sub_cat] += 1
        if scene_type in ["indoor", "outdoor"]:
            stats["scene_type"][scene_type] += score
            counts["scene_type"][scene_type] += 1
            
    major_scores = {
        cat: _sim_calculate_avg_score(stats["major_category"][cat], counts["major_category"][cat])
        for cat in ALL_MAJOR_CATEGORIES
    }
    
    final_scores = {
        "overall_score": _sim_calculate_avg_score(stats["overall"], counts["overall"]),
        "weighted_overall_score": _sim_calculate_weighted_score(major_scores, MAJOR_CATEGORY_WEIGHTS),
        "major_category_score": major_scores,
        "sub_category_score": {
            cat: _sim_calculate_avg_score(stats["sub_category"][cat], counts["sub_category"][cat])
            for cat in ALL_SUBCATEGORIES
        },
        "scene_type_score": {
            cat: _sim_calculate_avg_score(stats["scene_type"][cat], counts["scene_type"][cat])
            for cat in ["indoor", "outdoor"]
        }
    }
    return final_scores
# --- Logic Integration End ---


def simulate_random_choice_answers(problems: List[Dict[str, Any]]):
    """
    Perform 100 iterations of random guessing for multiple-choice questions in the test data,
    and calculate average scores according to the evaluation script logic.
    """
    print("\n--- Starting Random Guess Simulation (Average Final Score Mode) ---")
    
    choice_problems = [p for p in problems if p.get("problem_type") == "multiple_choice" and p.get("options")]
    if not choice_problems:
        print("Error: No multiple choice questions found in data.")
        return

    print(f"Found {len(choice_problems)} multiple choice questions. Running 100 simulations...")

    all_simulation_scores = []
    
    for _ in tqdm(range(100), desc="Simulating Random Answers", ncols=100):
        # 1. Generate a result set for this simulation
        current_run_results = []
        for problem in choice_problems:
            options_count = len(problem["options"])
            possible_answers = [chr(ord('A') + i) for i in range(options_count)]
            random_answer = random.choice(possible_answers)
            ground_truth = _parse_gt_solution(problem.get("solution", ""))
            
            sim_result_item = {
                **problem,  # Include all original fields
                "model_answer": random_answer,
                "is_correct": (random_answer.lower() == ground_truth.lower())
            }
            current_run_results.append(sim_result_item)
        
        # 2. Analyze results for this simulation
        scores = analyze_simulation_results(current_run_results)
        all_simulation_scores.append(scores)

    # 3. Calculate average score over 100 simulations
    final_avg_scores = {
        "overall_score": np.mean([s["overall_score"] for s in all_simulation_scores]),
        "weighted_overall_score": np.mean([s["weighted_overall_score"] for s in all_simulation_scores]),
        "major_category_score": {
            cat: np.mean([s["major_category_score"][cat] for s in all_simulation_scores])
            for cat in ALL_MAJOR_CATEGORIES
        },
        "sub_category_score": {
            cat: np.mean([s["sub_category_score"][cat] for s in all_simulation_scores])
            for cat in ALL_SUBCATEGORIES
        },
        "scene_type_score": {
            cat: np.mean([s["scene_type_score"][cat] for s in all_simulation_scores])
            for cat in ["indoor", "outdoor"]
        }
    }

    # 4. Print final average report
    print("\n--- Random Simulation Average Score Report (100 runs) ---")
    print(f"\n[Overall Scores]")
    print(f"  - Average Overall Score: {final_avg_scores['overall_score']:.2f}")
    print(f"  - Average Weighted Overall Score: {final_avg_scores['weighted_overall_score']:.2f}")
    
    print("\n[By Major Category]")
    for cat, score in final_avg_scores["major_category_score"].items():
        if score > 0: print(f"  - {cat}: {score:.2f}")
        
    print("\n[By Sub Category]")
    for cat, score in final_avg_scores["sub_category_score"].items():
        if score > 0: print(f"  - {cat}: {score:.2f}")

    print("\n[By Scene Type]")
    for cat, score in final_avg_scores["scene_type_score"].items():
        if score > 0: print(f"  - {cat}: {score:.2f}")
    
    print("\n-----------------------------------------")


def _process_video_chat_task(
    client: OpenAI,
    model_name: str,
    gen_config: Dict,
    video_path: str,
    problems_for_video: List[Dict[str, Any]],
    args: argparse.Namespace,
    independent_questions: bool = True,
    exemplars: Optional[Dict[str, Any]] = None,
    deep_guide_mode: bool = False
) -> List[Dict[str, Any]]:
    """
    Process a multi-turn conversation session for a single video (using OpenAI compatible API).
    
    Args:
        independent_questions: If True, treat each question as an independent session,
                               rather than a continuous multi-turn conversation. Saves tokens.
    """
    if not problems_for_video:
        return []

    all_results = []

    # --- Special Logic for Deep Guide Mode ---
    if deep_guide_mode:
        if not exemplars:
            # Should not happen theoretically as main function provides it
            raise ValueError("Deep guide mode requires an exemplar library, but none provided.")

        # In this mode, each question is independent and paired with an exemplar
        for problem in problems_for_video:
            t0_single = time.time()
            try:
                problem_type = problem.get("problem_type")
                if not problem_type or problem_type not in exemplars:
                    raise ValueError(f"Problem {problem.get('problem_id')} cannot find matching exemplar type.")

                exemplar = exemplars[problem_type]
                
                # 1. Load video frames for current problem (regular quality)
                t0_frames_prob = time.time()
                problem_frames = extract_video_frames(
                    video_path, 
                    model_name,
                    override_max_frames=args.max_frames
                )
                t1_frames_prob = time.time()
                
                # 2. Load keyframes for exemplar (high compression quality)
                t0_frames_ex = time.time()
                exemplar_path = exemplar.get("path")
                exemplar_keyframes = exemplar.get("keyframes")
                if not exemplar_path or not exemplar_keyframes:
                    raise ValueError(f"Exemplar {exemplar.get('problem_id')} missing path or keyframes field.")
                
                exemplar_frames = extract_video_frames(
                    exemplar_path, 
                    model_name, 
                    keyframe_indices=exemplar_keyframes,
                    override_jpeg_quality=EXEMPLAR_JPEG_QUALITY
                )
                t1_frames_ex = time.time()
                
                print(f"[Frame Processing] Problem: {len(problem_frames)} frames ({_fmt_dur(t0_frames_prob, t1_frames_prob)}). "
                      f"Exemplar: {len(exemplar_frames)} keyframes ({_fmt_dur(t0_frames_ex, t1_frames_ex)}).")

                # 3. Build and send request
                messages = _build_deep_guide_messages(
                    DEEP_GUIDE_PROMPT_TEMPLATE,
                    exemplar,
                    problem,
                    exemplar_frames,
                    problem_frames
                )
                
                local_gen_config = gen_config.copy()
                resp = _safe_openai_call(client, model_name, messages, local_gen_config)
                t1_single = time.time()

                # 4. Process and record results (similar to independent mode)
                model_raw_response = _resp_text(resp)
                error_msg = None
                if not model_raw_response:
                    finish_reason = resp.choices[0].finish_reason if (resp and resp.choices) else "Unknown"
                    error_msg = f"Empty response received. Finish reason: {finish_reason}"
                
                model_answer = _clean_model_answer(model_raw_response, problem_type, options=problem.get("options"))
                is_correct = _check_correctness(model_answer, problem.get("solution", ""), problem_type)
                in_tok, out_tok = _extract_usage(resp)

                result_item = {**problem} # Copy to avoid modifying original dict
                result_item.update({
                    "question": result_item.pop("problem", "N/A"),
                    "video_path": result_item.pop("path", "N/A"),
                    "ground_truth": result_item.pop("solution", "N/A"),
                    "model_raw_response": model_raw_response,
                    "model_answer": model_answer,
                    "is_correct": is_correct,
                    "input_tokens": in_tok,
                    "output_tokens": out_tok,
                    "elapsed_sec": t1_single - t0_single,
                    "error": error_msg,
                    "used_exemplar_id": exemplar.get("problem_id")
                })
                all_results.append(result_item)

            except Exception as e_single:
                 result_item = {**problem}
                 result_item.update({
                    "question": result_item.pop("problem", "N/A"),
                    "video_path": result_item.pop("path", "N/A"),
                    "ground_truth": result_item.pop("solution", "N/A"),
                    "model_raw_response": "", "model_answer": "", "is_correct": False, 
                    "input_tokens": 0, "output_tokens": 0, "elapsed_sec": 0,
                    "error": str(e_single)
                })
                 all_results.append(result_item)
        return all_results

    # --- Original Independent/Multi-turn Logic ---
    try:
        # --- 1. Video Processing ---
        t0_frame = time.time()
        base64_frames = extract_video_frames(
            video_path, 
            model_name,
            override_max_frames=args.max_frames
        )
        t1_frame = time.time()
        print(f"[Frame Processing] Extracted {len(base64_frames)} frames for {os.path.basename(video_path)}, took {_fmt_dur(t0_frame, t1_frame)}")
        
        if not base64_frames:
            raise ValueError("Failed to extract any frames from video.")

        # --- 2. Process problems based on mode ---
        if independent_questions:
            # --- Independent Question Mode ---
            for i, problem in enumerate(problems_for_video):
                t0_single = time.time()
                try:
                    problem_text = problem.get("problem", "N/A")
                    problem_type = problem.get("problem_type", "N/A")
                    options = problem.get("options")
                    options_text = ""
                    if problem_type == "multiple_choice" and options:
                        options_text = "Options:\n" + "\n".join(options)

                    # --- Few-shot Logic ---
                    few_shot_block_str = ""
                    if exemplars and problem_type in exemplars:
                        exemplar = exemplars[problem_type]
                        if exemplar.get("problem_id") != problem.get("problem_id"):
                            exemplar_options_text = ""
                            if exemplar.get("problem_type") == "multiple_choice" and exemplar.get("options"):
                                exemplar_options_text = "Options:\n" + "\n".join(exemplar["options"])
                            
                            few_shot_block_str = FEW_SHOT_EXAMPLE_TEMPLATE.substitute(
                                problem_type=exemplar.get("problem_type", ""),
                                problem_text=exemplar.get("problem", ""),
                                options_text=exemplar_options_text,
                                reason=exemplar.get("reason", ""),
                                solution=exemplar.get("solution", "")
                            )
                    
                    # --- New: Choose Prompt based on mode ---
                    prompt_template_to_use = SYSTEM_PROMPT_TEMPLATE
                    is_reasoning_mode = (args.test_id and args.show_reasoning) or args.rerun_incorrect or args.with_reasoning
                    if is_reasoning_mode:
                        print("[INFO] Reasoning Mode enabled. Requesting model to output thought process.")
                        prompt_template_to_use = REASONING_SYSTEM_PROMPT_TEMPLATE

                    prompt_str = prompt_template_to_use.substitute(
                        few_shot_block=few_shot_block_str,
                        problem_type=problem_type,
                        problem_text=problem_text, 
                        options_text=options_text
                    )
                    
                    # Dynamically adjust max_tokens
                    local_gen_config = gen_config.copy()
                    if is_reasoning_mode:
                        local_gen_config['max_tokens'] = 4096
                        print(f"[INFO] Reasoning Mode: 'max_tokens' set to {local_gen_config['max_tokens']}.")
                        
                    messages = _build_openai_messages(prompt_str, base64_frames)
                    resp = _safe_openai_call(client, model_name, messages, local_gen_config)
                    
                    t1_single = time.time()
                    model_raw_response = _resp_text(resp)
                    
                    error_msg = None
                    if not model_raw_response:
                        finish_reason = resp.choices[0].finish_reason if (resp and resp.choices) else "Unknown"
                        error_msg = f"Empty response received. Finish reason: {finish_reason}"
                        problem_id = problem.get("problem_id", "N/A")
                        print(f"[WARN] Received empty response (Problem ID: {problem_id}). Reason: {finish_reason}")

                    model_answer = _clean_model_answer(model_raw_response, problem_type, options=problem.get("options"), reasoning_mode=is_reasoning_mode)
                    is_correct = _check_correctness(model_answer, problem.get("solution", ""), problem_type)
                    in_tok, out_tok = _extract_usage(resp)
                    
                    result_item = {}
                    for key, value in problem.items():
                        if key == "problem": result_item["question"] = value
                        elif key == "path": result_item["video_path"] = value
                        elif key == "solution": result_item["ground_truth"] = value
                        else: result_item[key] = value
                    
                    result_item.update({
                        "model_raw_response": model_raw_response,
                        "model_answer": model_answer,
                        "is_correct": is_correct,
                        "input_tokens": in_tok,
                        "output_tokens": out_tok,
                        "elapsed_sec": t1_single - t0_single,
                        "frame_extraction_sec": t1_frame - t0_frame if i == 0 else 0, # Record only on first item
                        "error": error_msg
                    })
                    all_results.append(result_item)

                except Exception as e_single:
                    result_item = {}
                    for key, value in problem.items():
                        if key == "problem": result_item["question"] = value
                        elif key == "path": result_item["video_path"] = value
                        elif key == "solution": result_item["ground_truth"] = value
                        else: result_item[key] = value
                    result_item.update({
                        "model_raw_response": "", "model_answer": "", "is_correct": False, "input_tokens": 0, "output_tokens": 0,
                        "elapsed_sec": 0, 
                        "frame_extraction_sec": t1_frame - t0_frame if i == 0 else 0,
                        "error": str(e_single)
                    })
                    all_results.append(result_item)
                
                # --- New: Add sleep for ERNIE model after each independent question ---
                if "ernie" in model_name.lower():
                    time.sleep(2.0)
            
            return all_results

        # --- Default: Multi-turn Mode (if --keep-context is specified) ---
        message_history = []
        total_in_tok, total_out_tok = 0, 0
        
        # --- 2.1 First Question (Includes video frames) ---
        first_problem = problems_for_video[0]
        t0_first = time.time()

        problem_text = first_problem.get("problem", "N/A")
        problem_type = first_problem.get("problem_type", "N/A")
        options = first_problem.get("options")
        options_text = ""
        if problem_type == "multiple_choice" and options:
            options_text = "Options:\n" + "\n".join(options)
        
        # --- Few-shot Logic (Multi-turn) ---
        few_shot_block_str_first = ""
        if exemplars and problem_type in exemplars:
            exemplar = exemplars[problem_type]
            if exemplar.get("problem_id") != first_problem.get("problem_id"):
                exemplar_options_text = ""
                if exemplar.get("problem_type") == "multiple_choice" and exemplar.get("options"):
                    exemplar_options_text = "Options:\n" + "\n".join(exemplar["options"])
                
                few_shot_block_str_first = FEW_SHOT_EXAMPLE_TEMPLATE.substitute(
                    problem_type=exemplar.get("problem_type", ""),
                    problem_text=exemplar.get("problem", ""),
                    options_text=exemplar_options_text,
                    reason=exemplar.get("reason", ""),
                    solution=exemplar.get("solution", "")
                )

        # --- New: Prompt Selection for Multi-turn First Round ---
        prompt_template_to_use_first = SYSTEM_PROMPT_TEMPLATE
        is_reasoning_mode = (args.test_id and args.show_reasoning) or args.rerun_incorrect or args.with_reasoning
        if is_reasoning_mode:
            print("[INFO] Reasoning Mode enabled (Multi-turn First Round). Requesting model to output thought process.")
            prompt_template_to_use_first = REASONING_SYSTEM_PROMPT_TEMPLATE

        prompt_str = prompt_template_to_use_first.substitute(
            few_shot_block=few_shot_block_str_first,
            problem_type=problem_type,
            problem_text=problem_text, 
            options_text=options_text
        )
        
        # Dynamically adjust max_tokens for efficiency
        local_gen_config = gen_config.copy()
        if is_reasoning_mode:
            local_gen_config['max_tokens'] = 4096
            print(f"[INFO] Reasoning Mode: 'max_tokens' set to {local_gen_config['max_tokens']}.")
            
        first_messages = _build_openai_messages(prompt_str, base64_frames)
        resp_first = _safe_openai_call(client, model_name, first_messages, local_gen_config)
        
        t1_first = time.time()
        model_raw_response_first = _resp_text(resp_first)
        
        # --- Diagnostic Logic ---
        error_msg_first = None
        if not model_raw_response_first:
            finish_reason = resp_first.choices[0].finish_reason if (resp_first and resp_first.choices) else "Unknown"
            error_msg_first = f"Empty response received. Finish reason: {finish_reason}"
            problem_id = first_problem.get("problem_id", "N/A")
            print(f"[WARN] Received empty response (Problem ID: {problem_id}). Reason: {finish_reason}")

        model_answer_first = _clean_model_answer(model_raw_response_first, problem_type, options=first_problem.get("options"), reasoning_mode=is_reasoning_mode)
        is_correct_first = _check_correctness(model_answer_first, first_problem.get("solution", ""), problem_type)
        in_tok_f, out_tok_f = _extract_usage(resp_first)
        total_in_tok += in_tok_f
        total_out_tok += out_tok_f
        
        # Rebuild the dictionary to preserve original order and append new fields
        result_item = {}
        for key, value in first_problem.items():
            if key == "problem":
                result_item["question"] = value
            elif key == "path":
                result_item["video_path"] = value
            elif key == "solution":
                result_item["ground_truth"] = value
            else:
                result_item[key] = value
        
        result_item.update({
            "model_raw_response": model_raw_response_first,
            "model_answer": model_answer_first,
            "is_correct": is_correct_first,
            "input_tokens": in_tok_f,
            "output_tokens": out_tok_f,
            "elapsed_sec": t1_first - t0_first,
            "frame_extraction_sec": t1_frame - t0_frame,
            "error": error_msg_first
        })
        all_results.append(result_item)
        
        # Update history for next turn
        message_history.extend(first_messages)
        message_history.append({"role": "assistant", "content": model_raw_response_first})

        # --- 2.2 Subsequent Questions (Text Only) ---
        for problem in problems_for_video[1:]:
            t0_sub = time.time()
            try:
                problem_text = problem.get("problem", "N/A")
                problem_type = problem.get("problem_type", "N/A")
                options = problem.get("options")
                options_text = ""
                if problem_type == "multiple_choice" and options:
                    options_text = "Options:\n" + "\n".join(options)

                # --- Few-shot Logic (Multi-turn) ---
                few_shot_block_str_sub = ""
                if exemplars and problem_type in exemplars:
                    exemplar = exemplars[problem_type]
                    if exemplar.get("problem_id") != problem.get("problem_id"):
                        exemplar_options_text = ""
                        if exemplar.get("problem_type") == "multiple_choice" and exemplar.get("options"):
                            exemplar_options_text = "Options:\n" + "\n".join(exemplar["options"])
                        
                        few_shot_block_str_sub = FEW_SHOT_EXAMPLE_TEMPLATE.substitute(
                            problem_type=exemplar.get("problem_type", ""),
                            problem_text=exemplar.get("problem", ""),
                            options_text=exemplar_options_text,
                            reason=exemplar.get("reason", ""),
                            solution=exemplar.get("solution", "")
                        )

                # --- New: Prompt Selection for Subsequent Turns ---
                prompt_template_to_use_sub = SYSTEM_PROMPT_TEMPLATE
                if is_reasoning_mode: # is_reasoning_mode defined in first turn
                    prompt_template_to_use_sub = REASONING_SYSTEM_PROMPT_TEMPLATE

                prompt_str_sub = prompt_template_to_use_sub.substitute(
                    few_shot_block=few_shot_block_str_sub,
                    problem_type=problem_type,
                    problem_text=problem_text, 
                    options_text=options_text
                )
                
                # Dynamically adjust max_tokens for subsequent turns
                local_gen_config_sub = gen_config.copy()
                if is_reasoning_mode:
                    local_gen_config_sub['max_tokens'] = 4096

                subsequent_messages = _build_openai_messages(prompt_str_sub, history=message_history)
                resp_sub = _safe_openai_call(client, model_name, subsequent_messages, local_gen_config_sub)

                t1_sub = time.time()
                model_raw_response_sub = _resp_text(resp_sub)
                
                # --- Diagnostic Logic ---
                error_msg_sub = None
                if not model_raw_response_sub:
                    finish_reason_sub = resp_sub.choices[0].finish_reason if (resp_sub and resp_sub.choices) else "Unknown"
                    error_msg_sub = f"Empty response received. Finish reason: {finish_reason_sub}"
                    problem_id_sub = problem.get("problem_id", "N/A")
                    print(f"[WARN] Received empty response (Problem ID: {problem_id_sub}). Reason: {finish_reason_sub}")

                model_answer_sub = _clean_model_answer(model_raw_response_sub, problem_type, options=problem.get("options"), reasoning_mode=is_reasoning_mode)
                is_correct_sub = _check_correctness(model_answer_sub, problem.get("solution", ""), problem_type)
                in_tok_s, out_tok_s = _extract_usage(resp_sub)
                
                # Rebuild the dictionary to preserve order
                result_item = {}
                for key, value in problem.items():
                    if key == "problem":
                        result_item["question"] = value
                    elif key == "path":
                        result_item["video_path"] = value
                    elif key == "solution":
                        result_item["ground_truth"] = value
                    else:
                        result_item[key] = value

                result_item.update({
                    "model_raw_response": model_raw_response_sub,
                    "model_answer": model_answer_sub,
                    "is_correct": is_correct_sub,
                    "input_tokens": in_tok_s,
                    "output_tokens": out_tok_s,
                    "elapsed_sec": t1_sub - t0_sub,
                    "frame_extraction_sec": 0,
                    "error": error_msg_sub
                })
                all_results.append(result_item)
                # Update history
                message_history.append({"role": "user", "content": prompt_str_sub})
                message_history.append({"role": "assistant", "content": model_raw_response_sub})

            except Exception as e_sub:
                result_item = {}
                for key, value in problem.items():
                    if key == "problem":
                        result_item["question"] = value
                    elif key == "path":
                        result_item["video_path"] = value
                    elif key == "solution":
                        result_item["ground_truth"] = value
                    else:
                        result_item[key] = value

                result_item.update({
                    "model_raw_response": "", "model_answer": "", "is_correct": False, "input_tokens": 0, "output_tokens": 0,
                    "elapsed_sec": 0, "frame_extraction_sec": 0, "error": str(e_sub)
                })
                all_results.append(result_item)

    except Exception as e_chat:
        print(f"[Session Failed] Processing session for video {video_path} failed completely: {e_chat}")
        all_results = []
        for p in problems_for_video:
            result_item = {}
            for key, value in p.items():
                if key == "problem":
                    result_item["question"] = value
                elif key == "path":
                    result_item["video_path"] = value
                elif key == "solution":
                    result_item["ground_truth"] = value
                else:
                    result_item[key] = value
            
            result_item.update({
                "model_raw_response": "", "model_answer": "", "is_correct": False, "input_tokens": 0, "output_tokens": 0,
                "elapsed_sec": 0, "frame_extraction_sec": 0, "error": str(e_chat)
            })
            all_results.append(result_item)

    return all_results


# -------------------- Main Function --------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Video Question Answering tasks via OpenAI-compatible API.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("input_file", type=str, nargs='?', default="QA.txt", help="Input JSON (.txt) file path (default: QA.txt)")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output JSON file path. Auto-generated if not specified.")
    parser.add_argument("-w", "--workers", type=int, default=MAX_QA_WORKERS, help=f"Number of concurrent API worker threads (default: {MAX_QA_WORKERS})")
    parser.add_argument("-m", "--model", type=str, default=DEFAULT_MODEL, help=f"Model name to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--keep-context", action="store_true", help="Enable multi-turn conversation mode, keeping context for questions of the same video. Disabled by default.")
    parser.add_argument("--test-id", type=int, default=None, help="Enable test mode, run only the specified problem_id.")
    parser.add_argument("--show-reasoning", action="store_true", help="In test mode, request the model to show reasoning process. Must be used with --test-id.")
    parser.add_argument("--few-shot", action="store_true", help="[Text Mode] Provide one 'reason' example per problem type as in-context learning.")
    parser.add_argument("--simulate-random", action="store_true", help="Run 100 random guess simulations for multiple choice questions and output statistics, skipping API calls.")
    parser.add_argument("--rerun-incorrect", type=str, default=None, help="Provide a JSON file path containing incorrect question IDs to rerun only those questions.")
    parser.add_argument("--resume", action="store_true", help="[Resume] Read existing output file, skip completed questions, and append new results to the file.")
    parser.add_argument("--with-reasoning", action="store_true", help="[Main Feature] Force model to output thought process and save reasoning and answer separately.")
    parser.add_argument("--max-frames", type=int, default=None, help="Force set maximum frames extracted from video. Auto-adjusted based on model type if not set.")
    return parser.parse_args()

def main():
    # 1. Parse command line arguments
    args = parse_args()

    if args.show_reasoning and not args.test_id:
        print("Error: --show-reasoning argument must be used with --test-id.")
        return

    # --- Load problem data (early execution) ---
    input_path = Path(args.input_file)
    try:
        # Try utf-8-sig (handle BOM), fallback to utf-8 on failure
        data = json.loads(input_path.read_text("utf-8-sig"))
        problems = [item['sample'] for item in data if 'sample' in item]
        print(f"Successfully loaded {len(problems)} problems.")
    except Exception as e:
        print(f"Error: Failed to read or parse JSON file {input_path}: {e}")
        return
    
    if not problems:
        print("Error: 'sample' entry not found in JSON file.")
        return

    # --- New: Rerun incorrect questions logic ---
    if args.rerun_incorrect:
        try:
            with open(args.rerun_incorrect, 'r', encoding='utf-8') as f:
                incorrect_data = json.load(f)
            
            incorrect_ids = set()
            for id_list in incorrect_data.values():
                incorrect_ids.update(id_list)
            
            original_count = len(problems)
            problems = [p for p in problems if p.get("problem_id") in incorrect_ids]
            print(f"\n--- Rerun Incorrect Mode ---")
            print(f"Loaded {len(incorrect_ids)} incorrect IDs from {args.rerun_incorrect}.")
            print(f"Matched {len(problems)} problems (Original total: {original_count}). Will process only these.")
            print(f"--------------------------\n")

        except Exception as e:
            print(f"Error: Failed to read or process incorrect questions JSON file {args.rerun_incorrect}: {e}")
            return
            
    # --- Simulation Mode Check ---
    # If simulation mode, run simulation and exit, skipping API checks
    if args.simulate_random:
        simulate_random_choice_answers(problems)
        return

    # If not simulation mode, run API runner
    main_api_runner(args, problems)
    

def main_api_runner(args: argparse.Namespace, problems: List[Dict[str, Any]]):
    """Main logic for handling actual API calls."""
    
    # --- API Mode Initialization ---
    _api_key_raw = os.getenv("OPENAI_API_KEY")
    _api_base_raw = os.getenv("OPENAI_API_BASE")
    OPENAI_API_KEY = _api_key_raw.strip().strip('"') if _api_key_raw else None
    OPENAI_API_BASE = _api_base_raw.strip().strip('"') if _api_base_raw else None

    if not OPENAI_API_KEY:
        raise RuntimeError("Error: Environment variable 'OPENAI_API_KEY' not found.")
    if not OPENAI_API_BASE:
        raise RuntimeError("Error: Environment variable 'OPENAI_API_BASE' not found. Please set your proxy API address.")

    actual_model_name = args.model
    input_path = Path(args.input_file)
    
    # --- New: Automatic Mode Detection ---
    deep_guide_mode = False
    if input_path.name == "QA_fewshot.txt":
        print("Info: Input file 'QA_fewshot.txt' detected. Automatically activating Deep Guide mode.")
        deep_guide_mode = True
    
    if args.output:
        output_path = Path(args.output)
    elif args.rerun_incorrect:
        sanitized_model_name = actual_model_name.replace('/', '_')
        output_filename = f"rerun_incorrect_results_{sanitized_model_name}.json"
        output_path = Path(output_filename)
        print(f"Info: Rerun mode activated. Output will be saved to: {output_path}")
    else:
        # Create result directory based on input sample filename (e.g., 'QA_results')
        output_dir = Path(f"{input_path.stem}_results")

        # Create result filename based on model name
        sanitized_model_name = actual_model_name.replace('/', '_')
        output_filename = f"{sanitized_model_name}_openai_results.json"
        
        output_path = output_dir / output_filename

    # Ensure result directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # --- Resume Logic (Moved before filtering logic could be affected by other things) ---
    existing_results = []
    if args.resume and output_path.exists():
        try:
            print(f"[Resume] Detected output file: {output_path}")
            text = output_path.read_text(encoding='utf-8')
            if text.strip():
                existing_results = json.loads(text)
                if not isinstance(existing_results, list):
                     print(f"[Warning] Output file format incorrect (not a list), cannot resume. Will overwrite file.")
                     existing_results = []
                else:
                     print(f"[Resume] Loaded {len(existing_results)} existing records.")
            else:
                print(f"[Resume] Output file is empty, starting fresh.")
        except Exception as e:
            print(f"[Warning] Failed to read existing output file: {e}. Starting fresh.")
            existing_results = []

    # Filter problems based on resume logic
    if args.resume and existing_results:
        finished_ids = set(item.get("problem_id") for item in existing_results if item.get("problem_id") is not None)
        original_count = len(problems)
        problems = [p for p in problems if p.get("problem_id") not in finished_ids]
        print(f"[Resume] Filtered {len(finished_ids)} completed problems. Remaining {len(problems)} to process.")
        
        if not problems:
            print("[Resume] All problems completed. No need to run.")
            return

    total_start_time = time.time()

    print(f"--- Video Frame QA Process (OpenAI-Compatible API) ---")
    print(f"Model: {args.model} ({actual_model_name})")

    # --- Debugging: Print loaded environment variables ---
    api_key_display = f"{OPENAI_API_KEY[:5]}...{OPENAI_API_KEY[-4:]}" if OPENAI_API_KEY and len(OPENAI_API_KEY) > 9 else "Not Set or Too Short"
    print(f"DEBUG: Loaded API Key: {api_key_display}")
    print(f"DEBUG: Loaded API Base: {OPENAI_API_BASE or 'Not Set'}")
    # --- End Debugging ---

    print(f"API Base: {OPENAI_API_BASE}")
    print(f"Input File: {input_path}")
    print(f"Output File: {output_path}")

    # --- New: Force concurrency to 1 for ERNIE ---
    workers = args.workers
    if "ernie" in actual_model_name.lower():
        if workers != 1:
            print(f"[INFO] ERNIE model detected. Forcing concurrency to 1 (was {workers}) to avoid rate limits.")
            workers = 1

    print(f"Concurrency: {workers}")
    print(f"Frame Extraction Rate: 1 frame/sec")
    if args.keep_context:
        print("Mode: Multi-turn Conversation (Keep Context)")
    else:
        print("Mode: Independent Questions (Save Tokens, Default)")
    print(f"------------------------------------------------")

    # 1. Initialize Client
    # Warning: Disabling SSL verification poses security risks. Use only when network environment is secure and necessary.
    try:
        # Check system proxy settings (compatible with old httpx)
        proxy_url = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy") or os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")
        
        client_kwargs = {
            'verify': False,
            'timeout': httpx.Timeout(120.0, connect=60.0) # Extend total timeout to 120 seconds
        }

        if proxy_url:
            # Compatible with old httpx which only accepts 'proxy' argument
            print(f"DEBUG: System proxy detected, using legacy 'proxy' argument: {proxy_url}")
            client_kwargs['proxy'] = proxy_url

        custom_http_client = httpx.Client(**client_kwargs)
        client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE, http_client=custom_http_client)
    except Exception as e:
        print(f"Error: Failed to initialize OpenAI client: {e}")
        return
    
    gen_config = DEFAULT_GEN_CONFIG.copy()

    # Increase default token limit for GLM-4.5V model
    if "glm-4.5v" in actual_model_name.lower():
        gen_config['max_tokens'] = 2048
        print(f"[INFO] GLM-4.5V Model: Default max_tokens -> {gen_config['max_tokens']}")

    # 2. Load and Parse Problems
    try:
        # Try utf-8-sig (handle BOM), fallback to utf-8 on failure
        data = json.loads(input_path.read_text("utf-8-sig"))
        problems = [item['sample'] for item in data if 'sample' in item]
        print(f"Successfully loaded {len(problems)} problems.")
    except Exception as e:
        print(f"Error: Failed to read or parse JSON file {input_path}: {e}")
        return

    # --- Argument Conflict Check ---
    if args.few_shot and deep_guide_mode:
        print("Error: --few-shot (Text Exemplars) and Deep Guide Mode (Triggered by filename 'QA_fewshot.txt') cannot be used together.")
        return

    # --- Exemplar Library Construction (Select based on mode) ---
    exemplars = {}
    exemplar_ids = set()

    if args.few_shot:
        # --- Text Exemplar Mode ---
        print("Info: --few-shot (Text Exemplars) mode enabled.")
        for p in problems:
            ptype = p.get("problem_type")
            if ptype and p.get("reason") and ptype not in exemplars:
                exemplars[ptype] = p
        print(f"Text exemplar library constructed, total {len(exemplars)} types.")
    
    elif deep_guide_mode:
        # --- Deep Guide (Video Exemplar) Mode ---
        print(f"Info: Deep Guide (Video Exemplar) mode enabled.")
        # Exemplar file is the input file itself
        exemplar_file_path = input_path
        
        print(f"Loading exemplars from '{exemplar_file_path}'...")
        try:
            # Since exemplar file and problem file are the same, we can use loaded 'problems'
            all_exemplars = problems
            
            for p in all_exemplars:
                ptype = p.get("problem_type")
                # Must have reason and non-empty keyframes list
                if ptype and p.get("reason") and p.get("keyframes") and ptype not in exemplars:
                    exemplars[ptype] = p
                    exemplar_ids.add(p.get("problem_id"))
            
            print(f"Video exemplar library constructed, found {len(exemplars)} types of valid exemplars.")
            if not exemplars:
                print("Warning: Failed to find any valid exemplars containing both 'reason' and 'keyframes' in the exemplar file.")

        except Exception as e:
            print(f"Error: Failed to construct exemplar library: {e}")
            return
        
        # Exclude problems used as exemplars from the main problem list
        original_count = len(problems)
        problems = [p for p in problems if p.get("problem_id") not in exemplar_ids]
        print(f"Excluded {original_count - len(problems)} problems used as exemplars from the test set.")


    # --- New: Test Mode Logic ---
    if args.test_id:
        print(f"\n--- Test Mode Enabled ---")
        print(f"Searching for Problem ID: {args.test_id}")
        target_problem = next((p for p in problems if p.get("problem_id") == args.test_id), None)
        
        if not target_problem:
            print(f"Error: Problem ID {args.test_id} not found in input file.")
            return
            
        problems = [target_problem]
        print("Problem found, will process only this task.\n")
    # --- End Test Mode Logic ---

    if not problems:
        print("Error: 'sample' entry not found in JSON file.")
        return

    # Group problems by video path
    problems_by_video: Dict[str, List[Dict[str, Any]]] = {}
    for p in problems:
        video_path = p.get('path')
        if not video_path: continue
        if video_path not in problems_by_video:
            problems_by_video[video_path] = []
        problems_by_video[video_path].append(p)
    print(f"Grouped into {len(problems_by_video)} independent video sessions.")

    # 3. Concurrent QA Processing
    qa_t0 = time.time()
    # Initialize results with existing ones if resuming
    results: List[Dict[str, Any]] = list(existing_results) if args.resume else []
    
    tasks_to_run = list(problems_by_video.items())
            
    print(f"\n[Processing Started] Starting {workers} worker threads for {len(tasks_to_run)} video sessions...")

    def save_current_results(current_results):
        """Helper to save results immediately to disk"""
        current_results.sort(key=lambda r: (r.get("problem_id", 0) or 0))
        try:
            temp_output_path = output_path.with_suffix(".tmp")
            temp_output_path.write_text(
                json.dumps(current_results, indent=2, ensure_ascii=False),
                encoding="utf-8"
            )
            if temp_output_path.exists():
                if output_path.exists():
                    output_path.unlink()
                temp_output_path.rename(output_path)
        except Exception as e:
            print(f"Warning: Failed to save intermediate results: {e}")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _process_video_chat_task, 
                client, actual_model_name, gen_config, 
                video_path, problem_list, 
                args,
                not args.keep_context, 
                exemplars,
                deep_guide_mode=deep_guide_mode
            ): video_path
            for video_path, problem_list in tasks_to_run
        }
        
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing Video Sessions", ncols=100):
            try:
                video_results = fut.result()
                results.extend(video_results)
                
                # Real-time saving
                if not args.test_id:
                    save_current_results(results)

            except Exception as e:
                video_path = futures[fut]
                print(f"[Fatal Error] Session {video_path} raised unhandled exception: {e}")
            
    qa_t1 = time.time()
    print(f"[Processing Complete] QA processing phase finished. Time elapsed: {_fmt_dur(qa_t0, qa_t1)}")

    # --- Modified: Output based on mode ---
    if args.test_id:
        print("\n--- Test Mode Results ---")
        if results:
            pprint.pprint(results[0])
        else:
            print("Test produced no results (error might have occurred during processing).")
        print("--------------------")
        
        total_end_time = time.time()
        print(f"Total process time: {_fmt_dur(total_start_time, total_end_time)}")
    else:
        # 4. Final save (just to be sure and print final status)
        print(f"\n[Saving Results] Saving final results...")
        save_current_results(results)
        print(f"Detailed results saved to: {output_path}")
        
        total_end_time = time.time()
        print(f"Total process time: {_fmt_dur(total_start_time, total_end_time)}")


if __name__ == "__main__":
    main()
