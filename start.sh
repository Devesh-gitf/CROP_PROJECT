#!/bin/bash
# start.sh
uvicorn ai_crop_recommender_backend:app --host 0.0.0.0 --port $PORT
