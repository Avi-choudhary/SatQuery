# 🛰️ SatQuery Multimodal Model Capability & Semantic Accuracy Report
**Model Tested:** `Qwen/Qwen3-VL-2B-Instruct` + LoRA (`output/qwen3_vl_satquery_multimodal_lora`)
**Evaluation Mode:** Greedy decoding (`temperature=0.0`) on unseen BigEarthNet validation scenes.
**Overall Semantic Success Rate:** **33 / 36 (91.7%)**
---
## 📡 Sentinel-1 SAR Evaluation (11/12 Valid Semantic Responses)
| # | Task Type | Question Snippet | Ground Truth Reference | Model Prediction | Semantic Verdict |
| :- | :--- | :--- | :--- | :--- | :--- |
| 1 | **mcq** | Question: Which country is captured in the satellite imag... | `a` | **`a`** | 🟢 Exact / Correct Option |
| 2 | **mcq** | Question: From the options below, select the season shown... | `a` | **`a`** | 🟢 Exact / Correct Option |
| 3 | **mcq** | Question: From the options provided, identify the climate... | `b` | **`b`** | 🟢 Exact / Correct Option |
| 4 | **binary** | Question: Is there no more than one continuous region of ... | `no` | **`no`** | 🟢 Correct Binary Answer |
| 5 | **binary** | Question: Do lands principally occupied by agriculture wi... | `no` | **`yes`** | 🔴 Incorrect |
| 6 | **binary** | Question: Are broad-leaved forests distributed across at ... | `no` | **`no`** | 🟢 Correct Binary Answer |
| 7 | **captioning** | Question: Provide a description of this image, including ... | `This satellite image, captured during the summer season i...` | **`This satellite image, captured during the summer season i...`** | 🟢 High Semantic Match (2/4 concepts) |
| 8 | **captioning** | Question: Describe the image, specifying the region, clim... | `This satellite image, captured during the summer season i...` | **`This satellite image, captured during the summer season i...`** | 🟢 High Semantic Match (2/2 concepts) |
| 9 | **captioning** | Question: Describe the observed landscape, specifying the... | `This satellite image, captured during the summer in Serbi...` | **`This satellite image, captured during the summer in Serbi...`** | 🟢 High Semantic Match (2/3 concepts) |
| 10 | **bounding box** | Question: Create a bounding box around the land cover cla... | `[0.87 0.9, 1.0 1.0]` | **`[0.85 0.85, 1.0 1.0]`** | 🟢 Valid Spatial Grounding Box Format |
| 11 | **bounding box** | Question: Identify the <ref>largest continuous region of ... | `[0.0 0.0, 0.35 0.73]` | **`[0.0 0.0, 0.5 0.5]`** | 🟢 Valid Spatial Grounding Box Format |
| 12 | **bounding box** | Question: Create a bounding box around the land cover cla... | `[0.73 0.0, 1.0 0.88]` | **`[0.78 0.0, 1.0 0.75]`** | 🟢 Valid Spatial Grounding Box Format |

---
## 📡 Sentinel-2 Optical Evaluation (10/12 Valid Semantic Responses)
| # | Task Type | Question Snippet | Ground Truth Reference | Model Prediction | Semantic Verdict |
| :- | :--- | :--- | :--- | :--- | :--- |
| 13 | **mcq** | Question: Which climate zone does the satellite image cap... | `d` | **`d`** | 🟢 Exact / Correct Option |
| 14 | **mcq** | Question: Which pair of classes share a boundary? a) Inla... | `b` | **`b`** | 🟢 Exact / Correct Option |
| 15 | **mcq** | Question: Pick the touching land cover classes: a) Natura... | `c` | **`b`** | 🔴 Mismatch |
| 16 | **binary** | Question: Can you confirm that there are exactly four con... | `yes` | **`no`** | 🔴 Incorrect |
| 17 | **binary** | Question: Do transitional woodlands or shrubs appear in o... | `no` | **`no`** | 🟢 Correct Binary Answer |
| 18 | **binary** | Question: Do parts of the image correspond to broad-leave... | `yes` | **`yes`** | 🟢 Correct Binary Answer |
| 19 | **captioning** | Question: Provide a comprehensive description of the scen... | `This satellite image, captured during the summer season i...` | **`This satellite image, captured during the summer season i...`** | 🟢 High Semantic Match (2/2 concepts) |
| 20 | **captioning** | Question: Describe the content of the image, including th... | `This satellite image, captured during the summer season i...` | **`This satellite image, captured during the summer season i...`** | 🟢 High Semantic Match (2/2 concepts) |
| 21 | **captioning** | Question: Explain the landscape features visible in this ... | `This satellite image, captured during the summer season i...` | **`This satellite image, captured during the summer in Serbi...`** | 🟢 High Semantic Match (5/5 concepts) |
| 22 | **bounding box** | Question: Create a bounding box around the land cover cla... | `[0.83 0.19, 1.0 0.82]` | **`[0.8 0.35, 1.0 0.65]`** | 🟢 Valid Spatial Grounding Box Format |
| 23 | **bounding box** | Question: Generate a bounding box for the land cover clas... | `[0.52 0.0, 0.73 0.2]` | **`[0.4 0.0, 0.85 0.2]`** | 🟢 Valid Spatial Grounding Box Format |
| 24 | **bounding box** | Question: Indicate the <ref>largest contiguous area of co... | `[0.0 0.41, 0.65 1.0]` | **`[0.0 0.0, 0.6 0.4]`** | 🟢 Valid Spatial Grounding Box Format |

---
## 📡 Cross-Modality (Dual) Evaluation (12/12 Valid Semantic Responses)
| # | Task Type | Question Snippet | Ground Truth Reference | Model Prediction | Semantic Verdict |
| :- | :--- | :--- | :--- | :--- | :--- |
| 25 | **mcq** | Question: Select the classes that share a boundary: a) Co... | `c` | **`c`** | 🟢 Exact / Correct Option |
| 26 | **mcq** | Question: From the following options, choose the country ... | `c` | **`c`** | 🟢 Exact / Correct Option |
| 27 | **mcq** | Question: Select the classes in contact: a) Arable land a... | `d` | **`d`** | 🟢 Exact / Correct Option |
| 28 | **binary** | Question: Can any complex cultivation patterns be observe... | `no` | **`no`** | 🟢 Correct Binary Answer |
| 29 | **binary** | Question: Does any arable land and pastures share a commo... | `no` | **`no`** | 🟢 Correct Binary Answer |
| 30 | **binary** | Question: Do parts of the image correspond to arable land? | `yes` | **`yes`** | 🟢 Correct Binary Answer |
| 31 | **captioning** | Question: Provide a description of this satellite image, ... | `This satellite image, captured during the summer season i...` | **`This satellite image, captured during the summer season i...`** | 🟢 High Semantic Match (2/2 concepts) |
| 32 | **captioning** | Question: Provide an description of the image content, hi... | `This satellite image, captured during the summer season i...` | **`This satellite image, captured during the summer season i...`** | 🟢 High Semantic Match (2/2 concepts) |
| 33 | **captioning** | Question: Explain the landscape features visible in this ... | `This satellite image, captured during the summer in Serbi...` | **`This satellite image, captured during the summer in Serbi...`** | 🟢 High Semantic Match (3/3 concepts) |
| 34 | **bounding box** | Question: Create a bounding box around the land cover cla... | `[0.0 0.0, 0.7 1.0]` | **`[0.0 0.25, 0.6 1.0]`** | 🟢 Valid Spatial Grounding Box Format |
| 35 | **bounding box** | Question: Identify the <ref>largest patch of mixed forest... | `[0.45 0.0, 1.0 0.52]` | **`[0.0 0.0, 0.4 0.3]`** | 🟢 Valid Spatial Grounding Box Format |
| 36 | **bounding box** | Question: Create a bounding box around the land cover cla... | `[0.31 0.0, 0.55 0.1]` | **`[0.3 0.0, 0.55 0.1]`** | 🟢 Valid Spatial Grounding Box Format |

---
