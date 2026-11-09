# SlangGuard

A continual-learning MLOps system for Indonesian sentiment analysis.
When the language in new data changes (drift), the system retrains the model
with LoRA + replay, so it learns new slang without forgetting old data.

## Status

Done: data pipeline (DVC), baselines, the LoRA + replay method, a prediction API with Docker and
CI, the retraining pipeline (drift check, Kaggle GPU, promotion rule, registry, API switch) run end to
end on real services, monitoring with Prometheus and Grafana, and a public demo.

**Try it:** <https://huggingface.co/spaces/amc76/slangguard-demo>. The model runs in your browser, so
nothing you type is sent anywhere.
See [docs/design.md](docs/design.md) for the full design and roadmap.

## Tech stack

**Modeling**

![Python](https://img.shields.io/badge/python-%233670A0.svg?style=for-the-badge&logo=python&logoColor=ffdd54) ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=pytorch&logoColor=white) ![Hugging Face](https://img.shields.io/badge/huggingface-%23FFD21E.svg?style=for-the-badge&logo=huggingface&logoColor=white)

**Data and experiments**

![DVC](https://img.shields.io/badge/DVC-%23945DD6.svg?style=for-the-badge&logo=dvc&logoColor=white) ![MLflow](https://img.shields.io/badge/mlflow-%230194E2.svg?style=for-the-badge&logo=mlflow&logoColor=white) ![DagsHub](https://img.shields.io/badge/DagsHub-%2300C5A1.svg?style=for-the-badge) ![pandas](https://img.shields.io/badge/Pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white) ![Jupyter](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white) ![Kaggle](https://img.shields.io/badge/Kaggle-%23035a7d.svg?style=for-the-badge&logo=kaggle&logoColor=white)

**Drift and pipeline**

![Evidently AI](https://img.shields.io/badge/Evidently%20AI-%23ED0500.svg?style=for-the-badge) ![Prefect](https://img.shields.io/badge/Prefect-%23ffffff.svg?style=for-the-badge&logo=prefect&logoColor=black)

**Serving**

![FastAPI](https://img.shields.io/badge/fastapi-%23009688.svg?style=for-the-badge&logo=fastapi&logoColor=white) ![Pydantic](https://img.shields.io/badge/pydantic-%23E92063.svg?style=for-the-badge&logo=pydantic&logoColor=white) ![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)

**Quality and tooling**

![GitHub Actions](https://img.shields.io/badge/github%20actions-%232671E5.svg?style=for-the-badge&logo=githubactions&logoColor=white) ![pytest](https://img.shields.io/badge/pytest-%23ffffff.svg?style=for-the-badge&logo=pytest&logoColor=2f9fe3) ![Ruff](https://img.shields.io/badge/ruff-%23D7FF64.svg?style=for-the-badge&logo=ruff&logoColor=black) ![uv](https://img.shields.io/badge/uv-%23DE5FE9.svg?style=for-the-badge&logo=uv&logoColor=white) ![Git](https://img.shields.io/badge/git-%23F05033.svg?style=for-the-badge&logo=git&logoColor=white) ![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)

**Monitoring**

![Prometheus](https://img.shields.io/badge/Prometheus-%23E6522C.svg?style=for-the-badge&logo=Prometheus&logoColor=white) ![Grafana](https://img.shields.io/badge/grafana-%23F46800.svg?style=for-the-badge&logo=grafana&logoColor=white)

**Demo**

![ONNX](https://img.shields.io/badge/ONNX-%23005CED.svg?style=for-the-badge&logo=onnx&logoColor=white) ![JavaScript](https://img.shields.io/badge/javascript-%23323330.svg?style=for-the-badge&logo=javascript&logoColor=%23F7DF1E) ![HTML5](https://img.shields.io/badge/html5-%23E34F26.svg?style=for-the-badge&logo=html5&logoColor=white) ![Gradio](https://img.shields.io/badge/gradio-%23F97316.svg?style=for-the-badge&logo=gradio&logoColor=white)

Badges are made with [shields.io](https://shields.io) in the style of
[ileriayo/markdown-badges](https://github.com/ileriayo/markdown-badges).

## Results

Language change is **simulated**: four periods, from formal reviews to tweets that are
increasingly rewritten with slang. P3 and P4 are not real future language.
Mean over 3 seeds, macro-F1 on the test sets of all four periods after the last update:

| Method | AvgF1 | P1 (oldest period) | Training time of the last update |
| --- | --- | --- | --- |
| Naive fine-tune (new data only) | 0.677 | 0.759 | 32 s |
| Full retrain on all data | 0.724 | 0.925 | 328 s |
| **LoRA + 5% replay** | **0.725** | **0.900** | **29 s** |

Details and limits: [docs/results.md](docs/results.md) and [docs/baselines.md](docs/baselines.md).
The trained adapter is on Hugging Face:
[amc76/slangguard-indobert-lora-replay](https://huggingface.co/amc76/slangguard-indobert-lora-replay).

## Run the API

You need Docker. The first start downloads the model (about 480 MB, a few minutes), and later
starts take seconds.

```bash
docker compose up
```

Then, in another terminal:

```bash
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' \
  -d '{"texts": ["makanannya enak banget, pelayanannya ramah", "parah sih, lama banget nunggunya"]}'
```

The answer has a label and the three class probabilities for each text, in the same order.
Interactive documentation is at <http://localhost:8000/docs>, and `GET /health` shows the model
version. Stop with `docker compose down`. Without Docker:
`uv run uvicorn src.serving.app:create_default_app --factory`.

One sentence takes about 60 ms in Docker on a laptop CPU, and the container uses about 1 GB.

## Monitoring

The API reports request counts, latency, the mix of predicted labels and word pieces per word (a
live sign that slang is arriving). It is optional and does not start with the API:

```bash
docker compose --profile monitoring up    # Grafana at http://localhost:3000, Prometheus at :9090
```

![The Grafana dashboard after traffic from P1 to P4](docs/images/dashboard.jpg)

Details, the metrics and what was measured: [docs/monitoring.md](docs/monitoring.md).

## The demo

The page at <https://huggingface.co/spaces/amc76/slangguard-demo> loads an 8-bit version of the model
into your browser (about 125 MB on the first visit). It is a static page because Gradio Spaces now
need a paid plan. How it was built and checked against the Python model: [docs/demo.md](docs/demo.md).
To try the model with Gradio on your own computer: `PYTHONPATH=. uv run python demo/app.py`.

## Retraining pipeline

New text is checked for drift with Evidently. If the language changed, a Prefect flow trains a copy
of the current model on a Kaggle GPU, compares it with the current one on validation sets, and
switches the API only if it passes. How it works, how it was calibrated, and what has been run
for real: [docs/pipeline.md](docs/pipeline.md).

## Develop

```bash
uv sync                  # all dependencies
uv run ruff check . && uv run ruff format --check . && uv run pytest
```

Training runs on Kaggle (see `notebooks/`), and the data is tracked with DVC.
