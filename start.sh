#!/usr/bin/env bash
# Hosted deployment: ONE process (Streamlit). With CA_MODE=embedded the dashboard runs the backend code itself,
# so there is no separate API server. Without it, an API server is started on 127.0.0.1 as before.
set -e
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1   # keep native thread pools small on small hosts
PORT="${PORT:-8501}"

if [ "${CA_MODE:-}" != "embedded" ]; then
  export CA_API_URL="${CA_API_URL:-http://127.0.0.1:8000}"
  export CA_API_KEY="${CA_API_KEY:-dev-sensor-key}"
  (cd backend && python -m uvicorn cropadvisor.main:app --host 127.0.0.1 --port 8000) &
fi
cd frontend
exec python -m streamlit run app.py --server.port "$PORT" --server.address 0.0.0.0 --server.headless true
