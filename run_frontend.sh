#!/usr/bin/env bash
cd "$(dirname "$0")/frontend" && exec streamlit run app.py --server.port 8501 "$@"
