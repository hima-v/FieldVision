<p align="center">
  <img src="docs/readme/hero.svg" alt="FieldVision: Grand Prize, 1st place, Hack the Harvest 2026, USDA NASS challenge" width="100%">
</p>

<p align="center">
<a href="#quick-start"><img alt="Quick start" src="https://img.shields.io/badge/-Quick%20start-2F6B4F?style=for-the-badge"></a>
  <a href="handoff/presentation/FieldVision.pdf"><img alt="Slides" src="https://img.shields.io/badge/-Slides-D4AF37?style=for-the-badge"></a>
  <a href="#graded-against-usda-nass-2025"><img alt="Results" src="https://img.shields.io/badge/-Results-2F6B4F?style=for-the-badge"></a>
  <a href="#the-pipeline-9-stages"><img alt="Pipeline" src="https://img.shields.io/badge/-Pipeline-2F6B4F?style=for-the-badge"></a>
  <a href="#key-slides"><img alt="Key slides" src="https://img.shields.io/badge/-Key%20slides-2F6B4F?style=for-the-badge"></a>
  <a href="#skills-and-tech-stack"><img alt="Skills" src="https://img.shields.io/badge/-Skills-2F6B4F?style=for-the-badge"></a>
</p>

# FieldVision

**An agentic AI system that tells, field by field from free satellite data, whether US corn was cut for silage or
harvested for grain, months before USDA publishes the official number.**

Grand Prize (1st place) at Hack the Harvest 2026, the USDA NASS challenge at UC Davis, judged by USDA, NVIDIA,
Niantic Spatial and UC Davis. Built in 22.5 hours by Team FieldVision: Hetvi Bhadani, Hima Verma, Krishi Shah,
Priyanka Ramachandran and Siyun Chen.

<p align="center">
  <a href="#graded-against-usda-nass-2025"><img src="docs/readme/charts/highlights.svg" width="100%" alt="4 of 5 states within 5 points of NASS 2025; 95.5% field accuracy; 13.3 to 1.4 point gap with Bayes and USDA history; 13.1 to 1.8 point miss with cloud-gap Bayes; 0 unsupported LLM numbers"></a>
</p>

- **Machine learning + Bayesian inference + LLM agents in one pipeline.** A 6-feature classifier calls each field,
  Bayes' theorem fixes what the satellite cannot see, a RAG critic checks every county against USDA reports, and
  NVIDIA Nemotron agents orchestrate the run and explain it.
- **Field level, Maryland BARC (Beltsville) labels:** 42 of 44 field-years right (95.5%), with no US labels in training.
- **State level, USDA NASS 2025 state acres:** 11 states within 5 points using end to end architecture, each run once.
- **Explainable and auditable:** every number comes from code, every field call comes with its evidence, and the
  LLM never writes a number it cannot trace to a tool result.

Every figure FieldVision produces is a model estimate, not a NASS statistic.

## Quick start

Python 3.10+.

```bash
git clone https://github.com/ChenSiyun1234/FieldVision && cd FieldVision
uv venv && uv pip install -e ".[agents]"
cp .env.example .env          # optional: add NVIDIA_API_KEY to use Nemotron
```

| Command                                                                    | What it does                                                |
| -------------------------------------------------------------------------- | ----------------------------------------------------------- |
| `fieldvision serve`                                                        | The app at <http://127.0.0.1:8765> (analyst or dairy owner) |
| `fieldvision run 24033 2022 --report`                                      | One county-season through the agent, with a written report  |
| `fieldvision run 24021 2022 --offline`                                     | Same, with a rule-based agent and no LLM                    |
| `python -m fieldvision_agents.watch.watch 24021 2022 --data data/counties` | Replay a season in 5-day steps                              |
| `fieldvision test` · `pytest -q tests`                                     | 106 agent tests (offline) · 184 model tests                 |
| `python docs/readme/charts/make_charts.py`                                 | Redraw the charts in this README from the logged numbers    |

Without an NVIDIA key everything still runs. County data is not in git: put it in `data/counties/<fips>/<year>/`
or set `FIELDVISION_COUNTY_DATA` (see [`data/README.md`](data/README.md)).

## Data

All data sources used are public:

- **Satellite imagery:** Sentinel-2 L2A and Landsat 8/9 (Microsoft Planetary Computer)
- **USDA data:** Crop Sequence Boundaries, Cropland Data Layer, and QuickStats (2012-2025)
- **Crop reports:** NASS Crop Progress reports
- **European data:** Maize declarations from Austria, France, and the Netherlands
- **Reference fields:** The organisers' 44 labelled Beltsville fields
- **Environmental and dairy records:** Maryland dairy permits and EPA ECHO

## Beyond Classification Baseline

Our first model had 135 features in a boosted-tree model trained on European farm data, got 30 of 44 USDA-labelled
fields right. Feature engineering to 6 physical harvest features and switching
to a class-balanced logistic regression, picked by leave-one-country-out validation, took it to 42 of 44 (95.5%).

That fixed the field-by-field call. But adding fields up to county and state totals exposed two problems that no
amount of model tuning could fix:

1. **Clouds hide the harvest.** Silage is chopped over a few weeks in September. When those weeks are cloudy, the
   satellite never sees the cut, the field cannot be judged, and in dairy country those hidden fields are mostly
   silage. Wyoming County, NY read 13 points low for exactly this reason.
2. **Images mislead in some places.** Early-harvested Texas grain looks like silage from space: the model alone read
   Texas at 34% silage against USDA's 10%.

So we investigated by EDA on the dataset then on the cloud cover and tackled both problems probabilistically, with good old Bayes' theorem and the
USDA reports:

- **See more.** Landsat 8/9 as a second satellite, and a cloud check that only judges fields whose harvest was seen.
- **Bayes for the hidden fields.** p(silage) = πS / (πS + (1−π)G), using when the county's silage (S) and grain (G)
  harvests happened and the last official share (π).
- **Bayes with USDA history.** p′ = pπ / (pπ + (1−p)(1−π)) moves each field's probability toward the state's recent
  USDA share.
- **USDA reports as a critic.** A RAG pipeline over QuickStats and NASS Crop Progress reports checks every county,
  cites its sources, and sends implausible runs back.

<p align="center">
  <img src="docs/readme/charts/gap_closing.svg" width="100%" alt="Miss against USDA before and after each stage">
</p>

## Graded against USDA NASS 2025

The organisers graded teams on USDA NASS 2025 state acres of corn for grain and corn for silage
([QuickStats table](https://quickstats.nass.usda.gov/results/ACC45317-B978-3390-B0A1-BE0EC02A9D63)). The table was
read before any 2025 imagery was downloaded; the state method was written down first and nothing was tuned on a 2025
figure ([`handoff/results/ACCESS_LOG.md`](handoff/results/ACCESS_LOG.md)).

<p align="center">
  <img src="docs/readme/charts/state_2025.svg" width="100%" alt="Graded against USDA NASS 2025: Michigan 14.4 vs 14.7, New York 48.7 vs 53.2, Maryland 4.75 vs 8.9, Wisconsin 16.6 vs 21.6, Pennsylvania 15.2 vs 32.8">
</p>

County checks run the same way: Lancaster County, PA 2023, 26.6% against Census 27.4%, and Branch County, MI
2022, 2.1% against 1.8%.

<p align="center">
  <img src="docs/readme/charts/us_tests.svg" width="100%" alt="Learned from 8,000 European fields, adapted with US harvest signals, tested in 13 US states">
</p>

## The pipeline: 9 stages

Each stage handles one way the satellite signal can mislead, and each has a measured result. Click a stage name in
the table to open its code.

<p align="center">
  <img src="docs/readme/charts/pipeline.svg" width="100%" alt="FieldVision pipeline in 9 stages, orchestrated by a Nemotron supervisor">
</p>

<details>
<summary><b>The nine stages, with code links</b></summary>

| #   | Stage                                                                | What it does                                                                                                                                                                                                           | What it bought us                                                                                      |
| --- | -------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| 1   | [**Ingest**](src/fieldvision/s2/)                                    | Per-field time series from Sentinel-2 (every 5 days) and Landsat 8/9 on USDA Crop Sequence Boundaries, pulled from STAC catalogs                                                                                       | Fields hidden by cloud at harvest: 217 → 136 across 8 states. Corn in view in Kent County MD: 9% → 94% |
| 2   | [**Cloud check**](docs/module.md)                                    | A field is judged only when two clear images bracket the cut within 12 days. Otherwise it is flagged, never guessed                                                                                                    | Queen Anne's County MD: 22.4% → 4.0% (Census 1.3%)                                                     |
| 3   | [**Classify**](src/fieldvision/model/)                               | Class-balanced logistic regression on 6 physical features (moisture before the cut, greenness before the cut as a share of the field's own peak, browning rate, dry-down length, days from peak to cut, residue after) | 42 of 44 US fields (95.5%); a 135-feature boosted model got 37                                         |
| 4   | [**Cloud-gap Bayes**](src/fieldvision_agents/gap/)                   | p(silage) = πS / (πS + (1−π)G) for fields cut inside a cloud gap                                                                                                                                                       | Wyoming County NY miss: −13.1 → −1.8 points                                                            |
| 5   | [**USDA-history prior**](src/fieldvision/model/state_estimate.py)    | p′ = pπ / (pπ + (1−p)(1−π)), then area-weighted sums to county and state acres with a range                                                                                                                            | TX, CA, CO average gap: 13.3 → 1.4 points                                                              |
| 6   | [**RAG critic**](rag-pipeline/)                                      | Checks each county against QuickStats history (2012-2025), retrieves NASS Crop Progress passages when a check fails, sends the run back once                                                                           | Every flag comes with a cited source                                                                   |
| 7   | [**Silage Match**](recommender/)                                     | Min-cost max-flow matching dairies short on feed to standing corn within 20 road miles                                                                                                                                 | 2,000 t from 3 fields within 1.6 road miles, $28.69 to $33.03 per ton                                  |
| 8   | [**Orchestration + decision checks**](src/fieldvision_agents/react/) | Nemotron Lightning picks each tool in a ReAct loop; code enforces step order, one rerun, critic before accept, abstain over guess, and checks every number the LLM writes                                              | 5/5 routing, 4/4 evidence Q&A, 0 unsupported numbers                                                   |
| 9   | [**Watch loop**](src/fieldvision_agents/watch/)                      | Polls for new scenes every 5 days, reclassifies only when 5%+ of fields got a new clear look, wakes the agent only when the estimate moves 1+ point                                                                    | Agent ran 4 times instead of 21; Frederick's count settled by Oct 10, USDA publishes in January        |

</details>

## Key slides

From the final presentation ([full deck](handoff/presentation/FieldVision.pdf)). Click a slide to open the deck.

|                                                                                                                  |                                                                                                           |
| ---------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| **Data and feature engineering**<br>Five public sources, six physical features per field                         | **Model alone**<br>Within 5 points of USDA in 7 states                                                    |
| [![Data and feature engineering](docs/readme/slides/04_data_features.jpg)](handoff/presentation/FieldVision.pdf) | [![Model alone](docs/readme/slides/05_model_alone.jpg)](handoff/presentation/FieldVision.pdf)             |
| **Model + USDA history (Bayes)**<br>Where images mislead, USDA history steers                                    | **Cloud gaps**<br>Landsat as a second satellite: hidden fields 217 → 136                                  |
| [![Model plus USDA history](docs/readme/slides/06_bayes_usda_history.jpg)](handoff/presentation/FieldVision.pdf) | [![Cloud gaps](docs/readme/slides/08_cloud_gaps.jpg)](handoff/presentation/FieldVision.pdf)               |
| **Counties against USDA**<br>From Branch County MI to Wyoming County NY                                          | **Field-level test**<br>42 of 44 labelled US fields, precision and recall 91.7%                           |
| [![Counties against USDA](docs/readme/slides/07_counties_vs_usda.jpg)](handoff/presentation/FieldVision.pdf)     | [![Field-level test](docs/readme/slides/09_field_level_test.jpg)](handoff/presentation/FieldVision.pdf)   |
| **Agent architecture (NVIDIA Nemotron)**<br>Nemotron decides and explains; code makes every number               | **What broke and how we fixed it**<br>Reasoning tokens, prompt injection, unsupported numbers             |
| [![Agent architecture](docs/readme/slides/10_agent_architecture.jpg)](handoff/presentation/FieldVision.pdf)      | [![Agent reliability](docs/readme/slides/11_agent_reliability.jpg)](handoff/presentation/FieldVision.pdf) |

## What it delivers

| Who                            | What they get                                                                                                                                                                                                    |
| ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **USDA NASS analyst**          | In-season silage and grain acres for any county with a range, the reason behind each field's call, a check against USDA history with cited NASS reports, a written report, and a US map from state down to field |
| **Market analyst**             | A dashboard of USDA corn production, yield, stocks and price, plus a 2026-28 price scenario showing how silage acres move grain supply (+1 point of national silage share = 971,350 acres = 181M fewer bushels)  |
| **Dairy owner**                | Standing corn near the dairy that can still be bought for silage: tons, road miles and delivered price, on a map                                                                                                 |
| **Anyone monitoring a season** | A watch loop that checks for new Sentinel-2 scenes every 5 days and reruns the agent only when the estimate moves                                                                                                |

## Skills and tech stack

| Area                                       | What we built                                                         | Keywords                                                                                                                                                                                                                                                                                       |
| ------------------------------------------ | --------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Machine learning**                       | Silage vs grain classifier, model selection, evaluation               | Supervised classification, logistic regression, gradient boosting (LightGBM), feature engineering, transfer learning, class imbalance, cross-validation (spatial blocks, leave-one-country-out, leave-one-year-out), precision/recall/F1, ROC-AUC, model registry and versioning, scikit-learn |
| **Probabilistic modelling and statistics** | Cloud-gap probability, USDA-history prior, uncertainty ranges         | Bayesian inference, Bayes' theorem, prior updating, calibration, sampling ranges, pre-registered evaluation                                                                                                                                                                                    |
| **LLMs and agentic AI**                    | Nemotron supervisor, critic, report writer, vision check              | NVIDIA Nemotron, ReAct agent, tool calling, multi-agent orchestration, RAG, embeddings (sentence-transformers), vision-language models, LLM evaluation and benchmarking, guardrails, prompt-injection defense, hallucination checks                                                            |
| **Geospatial and remote sensing**          | Per-field satellite time series for 13 states                         | Sentinel-2, Landsat 8/9, Sentinel-1 SAR, STAC APIs (Microsoft Planetary Computer), GeoPandas, Shapely, rasterio, pyproj, NDVI / NDMI / NDTI, time-series analysis                                                                                                                              |
| **Data engineering**                       | Extraction and ingestion pipelines for imagery, USDA data and reports | ETL pipelines, Parquet and PyArrow, batch extraction on Google Colab, HTTP data ingestion, PDF parsing (PyMuPDF), data validation                                                                                                                                                              |
| **Optimization**                           | Dairy-to-field matching, decision thresholds                          | Min-cost max-flow (NetworkX), balanced-accuracy threshold optimization                                                                                                                                                                                                                         |
| **Backend and full stack**                 | Analyst and dairy app with maps, dashboard and chat                   | Python, FastAPI, Uvicorn, Pydantic, REST APIs, JavaScript, Leaflet, HTML/CSS, packaged CLI                                                                                                                                                                                                     |
| **Software engineering**                   | Installable package with 290 offline tests                            | Python packaging (pyproject), pytest, modular architecture, Git and pull requests, reproducible experiments, technical documentation                                                                                                                                                           |

## How the classifier was built

<p align="center">
  <img src="docs/readme/charts/field_check.svg" width="100%" alt="Field-level check on 44 labelled US fields: always grain 32, boosted trees 30 and 37, timing rule 38, 6-feature logistic 42, with Beltsville labels 43">
</p>

- **Labels from Europe.** The US has no field-level silage or grain labels, but Austria, France and the Netherlands
  record what farmers harvest. We trained on 8,000 of those fields.
- **US signals shaped the features.** Harvest dates don't carry across regions (Texas cuts in summer, the north in
  October), so the features describe the crop at the cut: how wet and green it was, how fast it browned, and what
  was left on the ground after.
- **Simple beat big.** The 135-feature boosted model fit Europe better and the US worse. The 6-feature logistic
  regression is the one we ship, and every call can be explained.
- **Optimization for an imbalanced problem.** About 94% of US corn is grain, so the model is fit with balanced class
  weights, rule thresholds are chosen by maximising balanced accuracy, and estimates are shifted to the real
  prevalence afterwards.
- **Beltsville as a check.** The 44 labelled Beltsville fields are the only labelled US set, and the team looked at
  them during development. A version with those labels in training scores 43 of 44, cross-validated by season.

![Fields called grain or silage from Sentinel-2 images and greenness curves](docs/readme/strips_sheet.png)

<details>
<summary><b>Model figures</b>: features, validation, confusion matrix, Landsat and evidence tiers</summary>

|                                                                                 |                                                                                       |
| ------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| ![The six features and their weights](docs/readme/02_features.png)              | ![Transfer to a new country and to labelled US fields](docs/readme/03_validation.png) |
| ![Confusion matrix on 44 labelled US field-years](docs/readme/04_confusion.png) | ![Landsat as a second satellite](docs/readme/06_landsat.png)                          |

![Why the evidence tiers exist](docs/readme/08_evidence_tiers.png)

</details>

## Results in detail

The full record, including which results are clean tests, is in [`docs/RESULTS.md`](docs/RESULTS.md).

<details>
<summary><b>Field level: 42 of 44 on USDA-labelled fields (Beltsville, MD, 2021-23)</b></summary>

The shipped model, `eu2_logistic_v1`, trained on European fields:

|                  | Predicted silage | Predicted grain |
| ---------------- | ---------------- | --------------- |
| **Truly silage** | 11               | 1               |
| **Truly grain**  | 1                | 31              |

| Accuracy | Precision | Recall | F1    | Speed                                  |
| -------- | --------- | ------ | ----- | -------------------------------------- |
| 95.5%    | 91.7%     | 91.7%  | 91.7% | 14 ms per field (3,463 fields in 47 s) |

`eu_barc_logistic_v1` adds the Beltsville labels to training: 43 of 44 (97.7%), each season predicted by a model
trained on the other two. Every scoring of this set is listed in [`handoff/results/ACCESS_LOG.md`](handoff/results/ACCESS_LOG.md).

</details>

<details>
<summary><b>Model alone across 7 states: average gap 2.0 points</b></summary>

300 fields per state, Sentinel-2, used while building the method (not clean tests).

| State        | NASS 2025 | FieldVision | Gap |
| ------------ | --------- | ----------- | --- |
| Iowa         | 1.9%      | 6.2%        | 4.3 |
| Minnesota    | 4.1%      | 6.4%        | 2.3 |
| Kansas       | 3.4%      | 1.3%        | 2.1 |
| Michigan     | 14.7%     | 12.7%       | 2.0 |
| South Dakota | 5.8%      | 4.3%        | 1.5 |
| Nebraska     | 2.1%      | 0.7%        | 1.4 |
| Wisconsin    | 21.6%     | 22.2%       | 0.6 |

</details>

<details>
<summary><b>Bayesian updates: where images mislead, math steers</b></summary>

**USDA-history prior** (state level, not clean tests). π = the state's USDA silage share in 2021, 2023 and 2024.
Texas, π = 10.5%: a field at p = 0.70 becomes 0.22.

| State      | NASS 2025 | Model alone | With prior | Gap        |
| ---------- | --------- | ----------- | ---------- | ---------- |
| Texas      | 10.3%     | 34.1%       | 10.2%      | 23.8 → 0.1 |
| California | 83.1%     | 75.0%       | 86.0%      | 8.1 → 2.9  |
| Colorado   | 14.0%     | 21.9%       | 12.7%      | 7.9 → 1.3  |

**Cloud-gap timing probability** (county level). In Wyoming County NY, 110 fields shared one cloud gap from Sep 10 to
Oct 5: green before (NDVI 0.84), bare after (0.22). There S = 0.632, G = 0.067, π = 0.559, so p = 0.92.

| Counting rule                               | Wyoming NY 2022 | Frederick MD 2022 |
| ------------------------------------------- | --------------- | ----------------- |
| Hidden fields dropped                       | −13.1           | −3.0              |
| Timing probability, π = last official share | −1.8            | −1.7              |

Both county answers were known before the rule was written, so these are not clean tests; the agent reports the
result as a range beside the estimate. Details: [`docs/hima/cloud_gaps.md`](docs/hima/cloud_gaps.md).

</details>

<details>
<summary><b>Second satellite and radar</b></summary>

|                                     | Sentinel-2 only | + Landsat 8/9 |
| ----------------------------------- | --------------- | ------------- |
| Fields hidden at harvest (8 states) | 217             | 136           |
| Model-alone miss, in points         | 3.5             | 2.8           |
| Iowa silage share (NASS 1.9%)       | 6.2%            | 2.5%          |

Sentinel-1 radar (pilot, France replay with a 3-5 week optical gap): silage recall 0.33 → 0.54. Built and tested,
not in the final runs.

</details>

<details>
<summary><b>Agents (NVIDIA Nemotron)</b></summary>

| Model                  | Size                   | Role                                              | Run time            |
| ---------------------- | ---------------------- | ------------------------------------------------- | ------------------- |
| Nemotron 3.5 Lightning | 30B total, 3B active   | Supervisor: picks each tool                       | 15.3 s per county   |
| Nemotron 3 Ultra       | 550B total, 55B active | Explains the critic's verdict, writes the report  | 37.6 s per county   |
| Nemotron 3.5 Super VL  | 120B total, 12B active | Reads before and after images of cloud-gap fields | about 6 s per field |
| Nemotron 3.5 Bolt      | 74B total, 9B active   | Benchmarked for fast Q&A                          | 6.9 s per county    |

What broke and how we fixed it: empty replies when reasoning used the token budget (900 → 2,048 tokens);
Lightning followed an instruction injected into a tool result (next steps now enforced in code); Ultra added
numbers not in the evidence in 2 of 8 reasons (number check with template fallback); vision right on 15 of 28
fields at 10 m (used as context, never as the label). Full notes: [`src/fieldvision_agents/README.md`](src/fieldvision_agents/README.md).

</details>

<details>
<summary><b>Corn price scenario (2026-28)</b></summary>

Regression on USDA stocks-to-use, last year's price and yield (`src/fieldvision_agents/market/`). It sizes what
silage acres do to the grain market: +1 point of national silage share is about 971,350 acres and 181M fewer
bushels, a price effect of +0.85% to +9.4% depending on the model. The 2026-28 path rests on assumed acres, yield
and use, so the dashboard labels it a scenario, not a forecast. See
[`src/fieldvision_agents/market/README.md`](src/fieldvision_agents/market/README.md).

</details>

<details>
<summary><b>Where it still reads low</b></summary>

Pennsylvania 2025 and Marathon County WI read low. Two causes: cloud hides the September silage harvest, and
northern silage cut after the crop yellows can look like grain to a model trained in Europe. Landsat and the
cloud-gap range fix the first; the second needs labelled northern fields. See
[`handoff/results/hima/COMBINED_GAP_FINDINGS.md`](handoff/results/hima/COMBINED_GAP_FINDINGS.md).

</details>

| Member                    | Contribution                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | Code                                                                                                                                                                                                                                                                                                                                                                                                       |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Hetvi Bhadani**         | **Data collection, EDA, Analysis, Fine Tuning, Dataset collection, Literature Review.** USDA NASS acreage scripts; Sentinel-2 time series with cloud filtering; multi-source data pipeline (CSB, CDL, Sentinel-1, weather)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | [`src/`](https://github.com/ChenSiyun1234/FieldVision/tree/dataset-analysis/src), [`download_corn_acres.py`](https://github.com/ChenSiyun1234/FieldVision/blob/dataset-analysis/notebooks/dataset/download_corn_acres.py), [`DATA_LOG.md`](https://github.com/ChenSiyun1234/FieldVision/blob/dataset-analysis/DATA_LOG.md) on `dataset-analysis`                                                           |
| **Priyanka Ramachandran** | **Data research, EDA, Geospatial data preparation, Dataset collection, Literature Review.** Beltsville field boundaries and labels; Cropland Data Layer analysis and corn maps; dairy location dataset from KML; coordinate systems and field geometries; weather data; presentation design                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | [BARC explorer](https://github.com/ChenSiyun1234/FieldVision/blob/dataset-analysis/notebooks/dataset/BARC%20Fields%20Explorer.ipynb), [CDL EDA](https://github.com/ChenSiyun1234/FieldVision/blob/dataset-analysis/notebooks/dataset/CDL%20Basic%20EDA.ipynb), [dairy farms](https://github.com/ChenSiyun1234/FieldVision/blob/dataset-analysis/notebooks/dataset/dairy_farms.ipynb) on `dataset-analysis` |
| **Hima Verma**            | **Probabilistic modeling, RAG, LLM agent optimization, Recommendation System, Ablation Study, ML Model optimization, Literature Review, Model Prototype, Team Lead.** Bayesian prior correction and cloud-gap timing model (county miss 13.1 → 1.8 points); uncertainty range for cloud-hidden fields (r = 0.69 over 11 places); RAG pipeline over USDA reports with data-leakage fix; agent tools, monitoring loop and LLM evaluation (NVIDIA Nemotron); optimization-based recommender (min-cost max-flow); led the repo, SDLC and team communication                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | [`feature/hima`](https://github.com/ChenSiyun1234/FieldVision/tree/feature/hima), [`rag-pipeline/`](rag-pipeline/), [`recommender/`](recommender/), [`gap/`](src/fieldvision_agents/gap/), [`watch/`](src/fieldvision_agents/watch/), [`docs/hima/`](docs/hima/)                                                                                                                                           |
| **Krishi Shah**           | **Agentic AI system, Benchmarking, Stock Price Prediction, Full-stack app.** Nemotron ReAct agent with guardrails; LLM benchmarking; FastAPI backend and frontend (map, dashboard); co-maintained the repo; final deck                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | [`feature/agent-app`](https://github.com/ChenSiyun1234/FieldVision/tree/feature/agent-app)                                                                                                                                                                                                                                                                                                                 |
| **Siyun Chen**            | **Classification model.** Satellite time-series pipeline; logistic regression model trained on EU field labels; model toolbox and tests | [`src/fieldvision/`](src/fieldvision/), [`tests/`](tests/), [`docs/RESULTS.md`](docs/RESULTS.md) |


<p align="center">
  <sub>Hosted by the Human Machine Collaboration Institute with USDA, UC Davis, NVIDIA, Solidigm, FarmGPU and Keeboh.</sub>
</p>

<details>
<summary><b>Repository map</b></summary>

```
src/fieldvision/          the model: extraction (s2/ ls/ s1/), features/, model/, toolbox.py
src/fieldvision_agents/   supervisor (react/), critic bridge, gap/, watch/, reporter/, market/, app/
rag-pipeline/             RAG critic: QuickStats checks, NASS report index
recommender/              Silage Match
registry/                 frozen model versions
docs/                     method, results, experiment log; docs/readme/ holds every image on this page
handoff/                  slides, figures, result files
colab/ notebooks/ scripts/ tests/
```

| I want to...                                    | Read                                                                                                 |
| ----------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| See every result and whether it is a clean test | [`docs/RESULTS.md`](docs/RESULTS.md)                                                                 |
| Understand the model                            | [`docs/module.md`](docs/module.md), [`docs/features.md`](docs/features.md)                           |
| Understand the agents and app                   | [`src/fieldvision_agents/README.md`](src/fieldvision_agents/README.md)                               |
| Understand the cloud-gap work                   | [`docs/hima/cloud_gaps.md`](docs/hima/cloud_gaps.md)                                                 |
| Use the RAG critic or recommender               | [`rag-pipeline/README.md`](rag-pipeline/README.md), [`recommender/README.md`](recommender/README.md) |
| See what was tried and failed                   | [`docs/log.md`](docs/log.md)                                                                         |

</details>
