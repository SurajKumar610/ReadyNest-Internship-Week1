# Dataset Analysis & Reporting Platform (DARP)

DARP is a complete, modular, and professional dataset analysis and reporting platform built with Python (FastAPI + Streamlit). It has been refactored into a decoupled **Frontend-Backend architecture** to allow professional deployment, caching, and scalable analytical processing.

---

## 🔗 Live Deployment Links

- **Frontend Application (Streamlit)**: [https://sun-darp.streamlit.app/](https://sun-darp.streamlit.app/)
- **Backend API (FastAPI)**: [https://readynest-internship-week1-2.onrender.com](https://readynest-internship-week1-2.onrender.com)

---

## 🚀 Architecture Reorganization

The project is structured into separate frontend and backend directories:

- **[`backend/`](file:///e:/Week1%20Dataset%20analysis%20platform/backend/)**: Exposes a REST API using FastAPI. It handles files uploads, data cleansing, descriptive statistics, time series forecasting, geographic mapping, machine learning model training, and A/B test evaluations.
  - **SQLite Database**: Automatically stores uploaded files details (original filename, row/column shapes, column names, missing values per column, inferred data types, and cleaning status) to maintain a persistent state.
- **[`frontend/`](file:///e:/Week1%20Dataset%20analysis%20platform/frontend/)**: A responsive Streamlit dashboard interface. It makes HTTP REST requests to the backend API and renders Plotly charts and Folium map components directly from JSON response payloads.

---

## 🛠️ Project Directory Layout

```text
Dataset-Analysis-Reporting/
│
├── backend/
│   ├── app/
│   │   ├── src/                   # Core analytical modules
│   │   ├── main.py                # FastAPI endpoints
│   │   ├── database.py            # SQLite session configuration
│   │   ├── models.py              # SQLAlchemy database tables
│   │   ├── schemas.py             # Pydantic validation schemas
│   │   └── crud.py                # CRUD queries
│   ├── tests/                     # Unit test files
│   ├── uploads/                   # Stored dataset CSV/Excel/JSON files
│   ├── reports/                   # Stored PDF, Excel, and PPT reports
│   ├── Dockerfile
│   └── requirements.txt
│
├── frontend/
│   ├── streamlit_app.py           # Streamlit app communicating with backend API
│   ├── Dockerfile
│   └── requirements.txt
│
├── docker-compose.yml             # Container orchestrator
├── LICENSE                        # MIT License
└── README.md                      # Reorganized setup and run manual
```

---

## ⚙️ Running the Services Locally

We recommend using virtual environments to run both components locally.

### 1. Start the Backend API (FastAPI)

1. Open a new terminal and navigate to the `backend/` directory:
   ```bash
   cd backend
   ```
2. Install the backend dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the FastAPI server:
   ```bash
   python -m uvicorn app.main:app --reload --port 8000
   ```
   *The interactive Swagger documentation is available at [http://localhost:8000/docs](http://localhost:8000/docs).*

### 2. Start the Frontend Dashboard (Streamlit)

1. Open a separate terminal and navigate to the `frontend/` directory:
   ```bash
   cd frontend
   ```
2. Install the frontend dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch the Streamlit dashboard:
   ```bash
   streamlit run streamlit_app.py --server.port 8501
   ```
   *The frontend dashboard will automatically open in your browser at [http://localhost:8501](http://localhost:8501).*

---

## 🐳 Running with Docker Compose (Recommended for Deployment)

You can launch both the frontend and backend servers together inside containers with a single command:

```bash
docker-compose up --build
```

- **Backend API**: Accessible at `http://localhost:8000`
- **Frontend App**: Accessible at `http://localhost:8501`

---

## 🧪 Running Unit Tests

To run the suite of 13 automated checks verifying the backend's core analytic engines:

```bash
cd backend
python -m unittest tests/test_all.py
```

---

## 🛠️ System Prerequisites

To run this platform locally or via containers, ensure your system meets these requirements:
* **Python**: Version `3.10` or higher (tested on `3.14.3`).
* **Docker & Compose**: Required for containerized orchestration (Docker Compose v5+).
* **Package Installer**: Standard `pip` (pre-bundled with Python).

---

## 🔒 Security & Configuration

### Hardcoded Credentials
> [!WARNING]
> **Demo Security Notice:**
> The frontend application uses hardcoded credentials (`admin` / `admin`) for quick demo login. For production or public deployment, implement proper authentication (such as JWT/OAuth2 flows) and override the credentials checks in `frontend/streamlit_app.py`.

### Environment Configuration
The platform components can be configured using environment variables:

| Component | Environment Variable | Default Value | Description |
| :--- | :--- | :--- | :--- |
| **Frontend** | `BACKEND_URL` | `https://readynest-internship-week1-2.onrender.com` | The endpoint URL of the FastAPI backend service. |
| **Backend** | `PORT` | `8000` | The port the FastAPI server binds to. |

---

## 📖 API Documentation

The FastAPI backend automatically hosts interactive API documentation.

### 🌐 Deployed Production API
* **Swagger UI:** [https://readynest-internship-week1-2.onrender.com/docs](https://readynest-internship-week1-2.onrender.com/docs)
* **ReDoc UI:** [https://readynest-internship-week1-2.onrender.com/redoc](https://readynest-internship-week1-2.onrender.com/redoc)

### 💻 Local Development API
* **Swagger UI:** [http://localhost:8000/docs](http://localhost:8000/docs)
* **ReDoc UI:** [http://localhost:8000/redoc](http://localhost:8000/redoc)


