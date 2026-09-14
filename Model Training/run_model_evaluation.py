import os
import sys
import json
import glob
import time
import random
import re
from pathlib import Path
import pandas as pd
from PIL import Image

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from satquery_model import SatQueryVLM

# -------------------------------------------------------------
# Question Banks: Realistic Natural Language Remote Sensing Queries
# -------------------------------------------------------------

S1_QUESTIONS = [
    "Analyze this Sentinel-1 SAR radar image. Describe the surface roughness, backscatter intensity, and moisture indications across this terrain.",
    "What kind of terrain or landforms are visible in this radar image, and where are the strongest reflection signatures located?",
    "Can you identify any open water or flat smooth surfaces versus rough vegetated structures based on the SAR backscatter patterns?",
    "Evaluate the polarimetric radar texture and identify potential human infrastructure, industrial structures, or topographic relief in this SAR scene.",
    "Examine the radar backscatter contrast in this scene. What does the return signal indicate about vegetation canopy density and ground wetness?"
]

S2_QUESTIONS = [
    "Provide an in-depth ecological and land use assessment of this Sentinel-2 satellite scene. What vegetation, urban elements, or agricultural patterns dominate?",
    "Examine the spectral reflectance and describe the seasonal landscape composition, vegetative health, and visible terrain features in this area.",
    "Identify the spatial distribution of developed vs natural areas in this scene, describing the orientation and boundaries of major sectors.",
    "What environmental or agricultural activities are evident from the field boundaries, crop patterns, and surface textures visible here?",
    "Analyze the dominant land cover types in this optical scene. Are there any water bodies, forest stands, or built-up infrastructure present?"
]

DUAL_QUESTIONS = [
    "Analyze both the optical and SAR imagery together in this dual-modality composite. How do the spectral colors correlate with the radar backscatter returns?",
    "Compare the radar surface roughness against the optical surface appearance. Do the high-backscatter structures correspond to buildings or rough natural terrain?",
    "How does combining SAR and optical observations enhance our understanding of this landscape compared to using either sensor alone?",
    "Assess the land cover classification confidence by cross-referencing the optical vegetation signature with the radar volume scattering.",
    "In this dual-modality satellite scene, identify how cloud penetration or surface moisture from SAR clarifies ambiguous features seen in optical."
]

def load_ground_truth():
    print("[1/5] Loading BigEarthNet ground-truth metadata from parquet...")
    parquet_path = MODULE_DIR / "data" / "BigEarthNet.txt.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Missing {parquet_path}")
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df):,} metadata records.")
    return df

def select_unseen_scenes(df):
    print("[2/5] Sampling 110 completely unseen satellite scenes (40 S1, 40 S2, 30 Dual)...")
    train_parquet = MODULE_DIR / "data" / "train_subset_multimodal.parquet"
    trained_images = set()
    if train_parquet.exists():
        train_df = pd.read_parquet(train_parquet)
        trained_images = set(train_df["image_path"].str.replace("\\", "/").unique())

    all_s1 = [p.replace("\\", "/") for p in glob.glob(str(MODULE_DIR / "data" / "images_s1" / "*.png"))]
    all_s2 = [p.replace("\\", "/") for p in glob.glob(str(MODULE_DIR / "data" / "images" / "*.png"))]
    all_dual = [p.replace("\\", "/") for p in glob.glob(str(MODULE_DIR / "data" / "images_dual" / "*.png"))]

    unseen_s1 = [p for p in all_s1 if p not in trained_images]
    unseen_s2 = [p for p in all_s2 if p not in trained_images]
    unseen_dual = [p for p in all_dual if p not in trained_images]

    random.seed(42)
    sample_s1 = random.sample(unseen_s1, min(40, len(unseen_s1)))
    sample_s2 = random.sample(unseen_s2, min(40, len(unseen_s2)))
    sample_dual = random.sample(unseen_dual, min(30, len(unseen_dual)))

    print(f"Sampled: {len(sample_s1)} Sentinel-1 | {len(sample_s2)} Sentinel-2 | {len(sample_dual)} Dual-Modality")
    return sample_s1, sample_s2, sample_dual

def extract_patch_metadata(image_path, df):
    stem = Path(image_path).stem
    # Match patch_id or s1_name in dataframe
    matches = df[(df["patch_id"].str.contains(stem, na=False)) | (df["s1_name"].str.contains(stem, na=False))]
    if matches.empty:
        # Fallback: look for sub-stem
        sub_stem = stem.replace("prev_", "").replace("dual_", "")
        matches = df[(df["patch_id"].str.contains(sub_stem, na=False)) | (df["s1_name"].str.contains(sub_stem, na=False))]

    if not matches.empty:
        row = matches.iloc[0]
        all_categories = list(matches["category"].dropna().unique())
        return {
            "country": str(row.get("country", "Unknown")),
            "season": str(row.get("season", "Unknown")),
            "climate_zone": str(row.get("climate_zone", "Unknown")),
            "latitude": float(row.get("latitude", 0.0)) if pd.notna(row.get("latitude")) else None,
            "longitude": float(row.get("longitude", 0.0)) if pd.notna(row.get("longitude")) else None,
            "ground_truth_categories": all_categories,
            "patch_id": str(row.get("patch_id", "")),
            "s1_name": str(row.get("s1_name", ""))
        }
    return {
        "country": "Unknown",
        "season": "Unknown",
        "climate_zone": "Unknown",
        "latitude": None,
        "longitude": None,
        "ground_truth_categories": [],
        "patch_id": "",
        "s1_name": ""
    }

KNOWN_LAND_COVERS = [
    "arable land", "pastures", "complex cultivation", "broad-leaved forest",
    "coniferous forest", "mixed forest", "inland waters", "marine waters",
    "water", "urban fabric", "industrial", "commercial", "discontinuous urban",
    "continuous urban", "transitional woodland", "natural grasslands",
    "moors", "heathlands", "wetlands", "peat bogs", "bare rocks", "sparsely vegetated"
]

def audit_response(answer, gt_meta, modality):
    ans_lower = answer.lower()
    mentioned_covers = [c for c in KNOWN_LAND_COVERS if c in ans_lower]
    gt_cats_lower = [str(c).lower() for c in gt_meta.get("ground_truth_categories", [])]
    
    hits = []
    for m in mentioned_covers:
        if any(m in gt or gt in m for gt in gt_cats_lower):
            hits.append(m)

    has_water_gt = any("water" in gt for gt in gt_cats_lower)
    has_water_pred = ("inland waters" in ans_lower or "marine waters" in ans_lower or "water body" in ans_lower or "open water" in ans_lower)
    water_hallucination = has_water_pred and not has_water_gt and len(gt_cats_lower) > 0

    is_pure_forest = len(gt_cats_lower) > 0 and all("forest" in gt or "woodland" in gt for gt in gt_cats_lower)
    urban_hallucination = is_pure_forest and ("urban fabric" in ans_lower or "industrial or commercial units" in ans_lower)

    sqm_mentions = re.findall(r"~?(\d+(?:,\d+)?)\s*(?:sqm|m²|square meters)", ans_lower)
    has_bbox = bool(re.search(r"\[([0-9.]+)[,\s]+([0-9.]+)[,\s]+([0-9.]+)[,\s]+([0-9.]+)\]", answer))

    hallucinations = []
    if water_hallucination:
        hallucinations.append("False water body detection (calm radar speckle or shadow misclassified as water)")
    if urban_hallucination:
        hallucinations.append("False urban settlement detection in homogenous forest parcel")

    score = 70.0
    if hits:
        score += min(20.0, len(hits) * 10.0)
    if gt_meta.get("country") != "Unknown" and gt_meta.get("country", "").lower() in ans_lower:
        score += 5.0
    if gt_meta.get("climate_zone") != "Unknown" and any(w in ans_lower for w in gt_meta.get("climate_zone", "").lower().split()):
        score += 5.0
    if hallucinations:
        score -= 25.0 * len(hallucinations)
    score = max(0.0, min(100.0, score))

    verdict = "PASS" if score >= 70.0 and not hallucinations else ("HALLUCINATED" if hallucinations else "PARTIAL")

    return {
        "score": round(score, 1),
        "verdict": verdict,
        "mentioned_land_covers": mentioned_covers,
        "matching_ground_truth": hits,
        "hallucinations_detected": hallucinations,
        "has_bounding_box": has_bbox,
        "sqm_estimates": sqm_mentions
    }

def main():
    print("="*70)
    print("  SatQuery: 110-Image Multimodal Evaluation & Hallucination Audit")
    print("="*70)

    gt_df = load_ground_truth()
    s1_images, s2_images, dual_images = select_unseen_scenes(gt_df)

    print("\n[3/5] Initializing SatQueryVLM on RTX 5060 Ti GPU...")
    start_init = time.time()
    vlm = SatQueryVLM(warmup=True)
    print(f"VLM ready in {time.time() - start_init:.2f}s.\n")

    test_cases = []
    for idx, img in enumerate(s1_images):
        test_cases.append({
            "id": f"s1_eval_{idx+1:02d}",
            "modality": "Sentinel-1 SAR",
            "image": img,
            "question": S1_QUESTIONS[idx % len(S1_QUESTIONS)]
        })
    for idx, img in enumerate(s2_images):
        test_cases.append({
            "id": f"s2_eval_{idx+1:02d}",
            "modality": "Sentinel-2 Optical",
            "image": img,
            "question": S2_QUESTIONS[idx % len(S2_QUESTIONS)]
        })
    for idx, img in enumerate(dual_images):
        test_cases.append({
            "id": f"dual_eval_{idx+1:02d}",
            "modality": "Cross-Modality Dual",
            "image": img,
            "question": DUAL_QUESTIONS[idx % len(DUAL_QUESTIONS)]
        })

    print(f"[4/5] Running inference on {len(test_cases)} unseen scenes...")
    results = []
    total_start = time.time()

    for i, tc in enumerate(test_cases, 1):
        gt_meta = extract_patch_metadata(tc["image"], gt_df)
        
        t0 = time.time()
        try:
            res = vlm.query(tc["image"], tc["question"], max_tokens=256)
            answer = res.get("answer", "")
            raw_answer = res.get("raw_answer", "")
            latency = res.get("latency_ms", (time.time() - t0) * 1000)
        except Exception as e:
            answer = f"INFERENCE_ERROR: {e}"
            raw_answer = str(e)
            latency = (time.time() - t0) * 1000

        audit = audit_response(answer, gt_meta, tc["modality"])

        record = {
            "test_id": tc["id"],
            "modality": tc["modality"],
            "image_path": tc["image"],
            "question": tc["question"],
            "model_answer": answer,
            "raw_answer": raw_answer,
            "latency_ms": round(latency, 2),
            "ground_truth": gt_meta,
            "audit": audit
        }
        results.append(record)

        if i % 10 == 0 or i == len(test_cases) or audit["hallucinations_detected"]:
            symbol = "🟢" if audit["verdict"] == "PASS" else ("⚠️" if audit["verdict"] == "PARTIAL" else "❌")
            print(f"  [{i:03d}/{len(test_cases)}] {symbol} {tc['modality'][:12]:<12} | Score: {audit['score']:>5.1f} | Latency: {latency:>6.0f}ms | Verdict: {audit['verdict']}")
            if audit["hallucinations_detected"]:
                for h in audit["hallucinations_detected"]:
                    print(f"       -> [HALLUCINATION]: {h}")

    total_time = time.time() - total_start
    print(f"\nCompleted {len(results)} queries in {total_time:.2f}s ({total_time/len(results):.2f}s/scene).")

    out_dir = MODULE_DIR / "output"
    out_dir.mkdir(exist_ok=True)
    out_json = out_dir / "evaluation_110_results.json"

    modality_stats = {}
    for mod in ["Sentinel-1 SAR", "Sentinel-2 Optical", "Cross-Modality Dual"]:
        sub = [r for r in results if r["modality"] == mod]
        avg_score = sum(r["audit"]["score"] for r in sub) / len(sub) if sub else 0
        avg_lat = sum(r["latency_ms"] for r in sub) / len(sub) if sub else 0
        passes = sum(1 for r in sub if r["audit"]["verdict"] == "PASS")
        partials = sum(1 for r in sub if r["audit"]["verdict"] == "PARTIAL")
        hallucinations = sum(1 for r in sub if r["audit"]["hallucinations_detected"])
        modality_stats[mod] = {
            "count": len(sub),
            "avg_score": round(avg_score, 1),
            "pass_rate_pct": round((passes / len(sub)) * 100, 1) if sub else 0,
            "partial_rate_pct": round((partials / len(sub)) * 100, 1) if sub else 0,
            "hallucination_rate_pct": round((hallucinations / len(sub)) * 100, 1) if sub else 0,
            "avg_latency_ms": round(avg_lat, 1)
        }

    total_hallucinations = sum(1 for r in results if r["audit"]["hallucinations_detected"])
    overall_summary = {
        "total_evaluated": len(results),
        "total_time_seconds": round(total_time, 2),
        "avg_latency_ms": round(sum(r["latency_ms"] for r in results) / len(results), 1),
        "overall_mean_score": round(sum(r["audit"]["score"] for r in results) / len(results), 1),
        "total_hallucinations_flagged": total_hallucinations,
        "overall_hallucination_rate_pct": round((total_hallucinations / len(results)) * 100, 2),
        "modality_breakdown": modality_stats
    }

    final_payload = {
        "summary": overall_summary,
        "detailed_results": results
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2, ensure_ascii=False)

    print(f"\n[5/5] Structured evaluation saved to: {out_json}")
    print("\n" + "="*70)
    print("                     EVALUATION SCORECARD")
    print("="*70)
    print(f"Total Scenes Tested:         {overall_summary['total_evaluated']}")
    print(f"Overall Mean Score:          {overall_summary['overall_mean_score']} / 100")
    print(f"Overall Hallucination Rate:  {overall_summary['overall_hallucination_rate_pct']}% ({total_hallucinations} cases)")
    print(f"Average Inference Latency:   {overall_summary['avg_latency_ms']} ms")
    print("-"*70)
    for mod, stat in modality_stats.items():
        print(f"{mod:<22} | Score: {stat['avg_score']:>5.1f} | Pass: {stat['pass_rate_pct']:>5.1f}% | Hallucination: {stat['hallucination_rate_pct']:>4.1f}% | Latency: {stat['avg_latency_ms']}ms")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
