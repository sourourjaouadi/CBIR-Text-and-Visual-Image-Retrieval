# CBIR: Text and Visual Image Retrieval

A client/server image search application that lets you find images using keywords, visual similarity, or both. Select an image already in the collection or upload a query image; the backend compares its visual descriptors against indexed images and returns ranked results.

The repository includes a small `objects` dataset workflow and an integration path for externally supplied MPEG-7 descriptor vectors. It does **not** include a production image corpus: you generate the demo data or provide your own images.

## Features

- Search images by keyword, with AND/OR matching.
- Search by a selected collection image or an uploaded query image.
- Combine visual descriptors with user-selected weights.
- Browse paginated image results and repeat a search from a result.
- Index image descriptors with a VP-Tree; store vectors in NumPy files and memory-map them when loading indexes.
- Run locally with SQLite or use Docker Compose with PostgreSQL.
- Evaluate retrieval quality with precision@k, recall@k, AP/mAP, and latency metrics.

## How it works

1. Image files are ingested into a database. The current ingestion script derives labels and keywords from underscore-separated filename tokens.
2. Descriptor scripts compute feature vectors for the images. The default handcrafted descriptors are HSV color histograms, Local Binary Patterns (LBP), and Histogram of Oriented Gradients (HOG).
3. Index scripts build a per-descriptor VP-Tree.
4. The API filters candidates by keyword, searches the selected descriptor indexes, combines standardized distances using the supplied weights, and returns ranked image URLs.

VGG16 descriptors are optional and require PyTorch and torchvision. MPEG-7 descriptors can be loaded from supported numeric text exports, but this path needs an explicit row-aligned image-ID manifest and is not a one-click import of the MIRFLICKR downloads.

## Repository layout

```text
backend/
  app/
    api/                 FastAPI endpoints
    core/
      descriptors/       HSV, LBP, HOG, VGG16, and MPEG-7 loader
      search/            Search engine and VP-Tree implementation
    db/                  SQLAlchemy database models and sessions
    schemas.py           Request and response schemas
  scripts/               Demo generation, ingestion, descriptor and index builds,
                         evaluation, and synthetic benchmarks
  tests/                 API, pipeline, loader, and index tests
frontend/                React + Vite + TypeScript application
data/
  images/                Runtime image files
  descriptors/           Descriptor matrices and image-ID maps
  indexes/               Search indexes
docker-compose.yml       PostgreSQL, seed job, API, and frontend
Makefile                 Common local commands
```

Generated data and image collections are runtime assets, not source files. Keep large datasets out of Git.

## Requirements

### Local development

- Python 3.11
- Node.js 20 or newer (npm included)
- Windows PowerShell, macOS, or Linux

### Docker

- Docker Desktop or Docker Engine with the Docker Compose plugin

The full MIRFLICKR-25k image archive is about 2.9 GB before extraction. Allow additional disk space for extracted images, descriptors, indexes, and temporary build files.

## Quick start: small demo

The demo creates 18 synthetic images. It is a quick way to verify the full application without downloading a dataset.

### 1. Set up the backend

Run these commands from the repository root. PowerShell commands are shown:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r backend\requirements.txt
Copy-Item .env.example .env
```

If PowerShell blocks virtual-environment activation, use a terminal where activation is permitted, or run the environment's Python directly as `.venv\Scripts\python.exe`.

### 2. Generate and index the demo images

```powershell
python -m backend.scripts.generate_demo --output data\images\objects
python -m backend.scripts.ingest_images --dataset objects --directory data\images\objects
python -m backend.scripts.compute_descriptors --dataset objects
python -m backend.scripts.build_indexes --dataset objects
```

Run these commands again after changing or adding collection images: ingest them, recompute descriptors, and rebuild the indexes.

### 3. Start the API

In the same terminal, from the repository root:

```powershell
uvicorn backend.app.main:app --reload
```

The API creates its database tables when it starts. With the default `.env`, it uses a local SQLite database (`cbir.db`).

### 4. Start the frontend

Open a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open the frontend at <http://localhost:5173>. The API health check is at <http://localhost:8000/health>, and interactive API documentation is at <http://localhost:8000/docs>.

To stop either local service, press `Ctrl+C` in its terminal.

## Use the MIRFLICKR-25k images

For a real-image experiment, download **[mirflickr25k.zip](http://press.liacs.nl/mirflickr/mirflickr25k.v3b/mirflickr25k.zip)**, the MIRFLICKR-25k image collection. It is approximately 2.9 GB. Extract the images under:

```text
data/images/objects/mirflickr25k/
```

The exact extracted folder layout can vary; the ingestion script searches recursively. Make sure the image files remain somewhere inside `data/images/`, because the API serves images from that directory.

Then, from the repository root with the backend virtual environment active, run:

```powershell
python -m backend.scripts.ingest_images --dataset objects --directory data\images\objects\mirflickr25k
python -m backend.scripts.compute_descriptors --dataset objects
python -m backend.scripts.build_indexes --dataset objects
```

Descriptor computation and index construction can take considerably longer for 25,000 images than for the 18-image demo.

### About MIRFLICKR annotations and tags

The optional [MIRFLICKR-25k annotations archive](http://press.liacs.nl/mirflickr/mirflickr25k.v3b/mirflickr25k_annotations_v080.zip) is not required to build visual indexes. The current ingestion script does **not** parse the MIRFLICKR annotation files or import their tags. It derives keywords from image filenames instead, so MIRFLICKR keyword search will not provide the collection's official tags without additional annotation-import work.

## Run with Docker Compose

Docker Compose starts PostgreSQL, generates and indexes the small demo dataset, then starts the API and frontend:

```powershell
docker compose up --build
```

Open <http://localhost:8080>. The API and its documentation are available at <http://localhost:8000> and <http://localhost:8000/docs>. Compose uses PostgreSQL; local development uses SQLite by default. The Compose seed job always uses the synthetic demo images, not MIRFLICKR.

Stop the containers with `Ctrl+C`. To stop and remove the containers while preserving the named database volume, run:

```powershell
docker compose down
```

## Configuration

Copy `.env.example` to `.env` for local development. Settings are read by the backend:

| Variable | Default | Purpose |
| --- | --- | --- |
| `DATABASE_URL` | `sqlite:///./cbir.db` | SQLAlchemy database connection. |
| `DATA_DIR` | `./data` | Root directory for images, descriptors, indexes, and runtime outputs. |
| `PUBLIC_BASE_URL` | `http://localhost:8000` | Base URL used when returning image links. |
| `API_CORS_ORIGINS` | `http://localhost:5173` | Comma-separated origins allowed to call the API from a browser. |
| `LOG_LEVEL` | `INFO` | Backend logging level. |
| `VGG_LAYER` | `fc7` | VGG16 feature layer (`fc7` or `conv5_pool`). |
| `VGG_NUM_WORKERS` | `0` | DataLoader worker count for VGG16 feature extraction. |

SQLite is created automatically when the API or ingestion scripts initialize the database. PostgreSQL is optional for local development; Docker Compose configures it for the containerized demo.

## Search API

The main endpoints are:

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Check that the API is running. |
| `GET` | `/api/descriptors?dataset=objects` | List available descriptors for a dataset. |
| `GET` | `/api/images?dataset=objects&page=1&size=20` | Browse a page of images; an optional `keyword` filters by keyword. |
| `GET` | `/api/images/{image_id}?dataset=objects` | Get one image and its keywords. |
| `POST` | `/api/search` | Search with JSON fields, optionally using an indexed query image. |
| `POST` | `/api/search/upload` | Search using an uploaded query image (supported for `objects`). |

Use <http://localhost:8000/docs> to try endpoints interactively. Example JSON request:

```json
{
  "dataset": "objects",
  "keywords": ["red"],
  "keyword_mode": "and",
  "query_image_id": null,
  "descriptors": [
    { "name": "color_hsv", "weight": 1.0 },
    { "name": "hog", "weight": 0.5 }
  ],
  "k": 10
}
```

`keywords` and `descriptors` may be empty. For image-to-image search, pass a `query_image_id` belonging to the selected dataset. Each descriptor weight must be greater than zero; `k` must be between 1 and 500. The response includes ranked results, distances, result image URLs, and the elapsed time.

Uploads are supported for the `objects` dataset using the locally implemented extractors. An uploaded query image is not currently supported for `mpeg7_1m`, because the project does not include a compatible MPEG-7 query-image extractor.

## Descriptors and indexes

The default object descriptor set is:

| Descriptor | Description | Default distance |
| --- | --- | --- |
| `color_hsv` | HSV color histogram | L1 |
| `lbp` | Local Binary Patterns texture descriptor | L1 |
| `hog` | Histogram of Oriented Gradients shape descriptor | L2 |
| `vgg16` | Optional pretrained VGG16 feature vector | L2 |

Compute the default handcrafted descriptors and build indexes with:

```powershell
python -m backend.scripts.compute_descriptors --dataset objects
python -m backend.scripts.build_indexes --dataset objects
```

Index metric overrides can be applied globally or per descriptor:

```powershell
python -m backend.scripts.build_indexes --dataset objects --metric l2
python -m backend.scripts.build_indexes --dataset objects --descriptor-metric color_hsv=l2 --descriptor-metric lbp=l1
```

Descriptor matrices and their row-to-image mappings are stored under `data/descriptors/<dataset>/`. The ID mapping is essential: search never assumes that a descriptor matrix row number is the database image ID. Index metadata is stored under `data/indexes/<dataset>/`. Keyword filtering happens before visual ranking. Small candidate sets (up to 50,000 images) use brute-force kNN; larger sets use VP-Tree over-fetch. Selected descriptor distances are standardized and combined using their weights.

## Optional VGG16 features

VGG16 extraction requires PyTorch and torchvision and may download pretrained weights on first use. On a CPU-only Windows/Linux/macOS environment, install the matching CPU wheels before installing the rest of the requirements if needed:

```powershell
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r backend\requirements.txt
```

For GPU, install compatible PyTorch and torchvision wheels using the [official PyTorch installation selector](https://pytorch.org/get-started/locally/). Then select VGG16 explicitly:

```powershell
python -m backend.scripts.compute_descriptors --dataset objects --descriptor vgg16 --limit 1000 --batch-size 64 --device cpu
python -m backend.scripts.build_indexes --dataset objects
```

Remove `--limit 1000` to process all images. Extraction can be resumed and its checkpoint data is stored under `data/descriptors/objects/vgg16/`. VGG16 indexes use PCA by default; `--pca-dim` and `--pca-sample-size` can tune that build.

## MPEG-7 descriptor imports

The project provides a streaming loader for **row-oriented numeric text or CSV** descriptor exports. It needs a separate text manifest containing one unique database image ID per descriptor row, in exactly the same order. For example:

```text
data/descriptors/mpeg7_1m/ColorLayout.txt   # one numeric vector per line
data/descriptors/mpeg7_1m/ColorLayout.ids   # one image ID per line, same row order
```

Convert a whitespace-separated export:

```powershell
python -m backend.app.core.descriptors.mpeg7_loader `
  data\descriptors\mpeg7_1m\ColorLayout.txt `
  data\descriptors\mpeg7_1m\ColorLayout.npy `
  --ids data\descriptors\mpeg7_1m\ColorLayout.ids
```

For comma-separated rows, add `--delimiter ','`. Repeat for each descriptor and build indexes:

```powershell
python -m backend.scripts.build_indexes --dataset mpeg7_1m
```

The MIRFLICKR download page's descriptor archives are not automatically compatible with this importer. Confirm their file format and create a correct image-ID manifest before attempting an import. XML/binary inputs, descriptors with row headers, or mismatched IDs need a format-specific adapter; do not infer IDs from row positions. Building a VP-Tree for one million vectors can require substantial memory, disk space, and time, so benchmark the target machine first.

## Evaluation and tests

Run the automated backend tests from the repository root:

```powershell
pytest backend/tests
```

Evaluate the available descriptors against labeled images:

```powershell
python -m backend.scripts.evaluate --dataset objects --queries 100 -k 20
```

Evaluation writes per-query and aggregate precision@k, recall@k, AP/mAP, and latency reports to `data/evaluation/<dataset>` by default. It evaluates each built descriptor and combinations of descriptors. A separate synthetic scaling benchmark reports index build time, memory, archive size, query time, and recall against brute force:

```powershell
python -m backend.scripts.benchmark --sizes 10000 100000 1000000 --dim 64 --queries 20 -k 10
```

The one-million-vector benchmark includes brute-force comparisons and can be compute-intensive.

## Troubleshooting

- **No images or results:** check that ingestion completed and that `data/images/` contains your images. Confirm the selected dataset name matches the one used with `--dataset`.
- **Missing descriptor or index:** rerun `compute_descriptors` and `build_indexes` after ingesting images.
- **Image URL returns 404:** the image must remain inside `DATA_DIR/images`, and the API must be able to access that same data directory.
- **Frontend cannot reach the API:** verify the backend is listening on port 8000 and that `API_CORS_ORIGINS` includes the frontend origin (normally `http://localhost:5173`).
- **MIRFLICKR keyword results are unhelpful:** the importer currently derives keywords from filenames; it does not load MIRFLICKR annotations.
- **MPEG-7 upload search is rejected:** this is expected until a compatible query-image extractor is implemented.

## Known limitations and next steps

- MIRFLICKR annotations/tags are not imported; keyword extraction currently uses filename tokens.
- MPEG-7 import currently supports row-oriented numeric text/CSV plus explicit IDs, not arbitrary XML or binary exports.
- The MIRFLICKR 1M dataset and its feature files require significant storage and compute resources; end-to-end compatibility has not been established.
- Production deployments should add upload size/retention controls, authentication, database migrations, and monitoring.

"# CBIR-Text-and-Visual-Image-Retrieval" 
