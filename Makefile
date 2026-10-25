.PHONY: demo seed api web test evaluate

demo:
	docker compose up --build

seed:
	python -m backend.scripts.generate_demo --output data/images/objects
	python -m backend.scripts.ingest_images --dataset objects --directory data/images/objects
	python -m backend.scripts.compute_descriptors --dataset objects
	python -m backend.scripts.build_indexes --dataset objects

api:
	uvicorn backend.app.main:app --reload

web:
	cd frontend && npm run dev

evaluate:
	python -m backend.scripts.evaluate --dataset objects --queries 100 -k 20

test:
	pytest backend/tests
