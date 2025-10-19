# AI-Powered Furniture Recommendation System

**An intelligent recommendation engine combining Machine Learning, Natural Language Processing, Computer Vision, and Generative AI to deliver personalized furniture recommendations through a conversational interface.**

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Backend Setup](#backend-setup)
  - [Frontend Setup](#frontend-setup)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Data Analytics](#data-analytics)
- [Model Training](#model-training)
- [Deployment](#deployment)
- [Performance Metrics](#performance-metrics)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This project implements a complete AI-powered furniture recommendation system that leverages multiple AI domains to provide intelligent, context-aware product recommendations. Users interact through a natural chat interface, receiving personalized suggestions based on semantic understanding of their queries.

**Key Capabilities:**
- Semantic search using state-of-the-art NLP embeddings
- Real-time recommendations with sub-50ms latency
- Conversational AI interface powered by GenAI
- Rich analytics dashboard with interactive visualizations
- Scalable vector database architecture
- Production-ready FastAPI backend with async support

---

## Features

### Core Functionality

**1. Intelligent Recommendations**
- Semantic similarity search using sentence-transformers
- Context-aware product matching
- Price range filtering
- Category-based organization
- Metadata-rich results with images, prices, and descriptions

**2. Conversational Interface**
- Natural language query processing
- AI-generated conversational responses
- Query history and context preservation
- Real-time chat interaction

**3. Analytics Dashboard**
- Price distribution visualization
- Category breakdown (pie/bar charts)
- Material analysis
- Brand insights
- Color distribution
- Summary statistics

**4. Data Processing**
- Comprehensive EDA and preprocessing
- Missing value imputation
- Text normalization and enrichment
- Combined field generation for optimal embeddings

---

## Tech Stack

### Backend
- **Framework:** FastAPI 0.104.1 (Python 3.9+)
- **ML/NLP:** 
  - sentence-transformers (all-MiniLM-L6-v2)
  - scikit-learn
  - PyTorch
- **GenAI:** 
  - HuggingFace Transformers (Flan-T5-small)
  - LangChain (RAG implementation)
- **Vector Database:** Pinecone (serverless)
- **Data Processing:** pandas, numpy
- **Validation:** Pydantic

### Frontend
- **Framework:** React 18.2.0
- **Build Tool:** Vite 5.0.8
- **Routing:** React Router DOM 6.20.0
- **HTTP Client:** Axios 1.6.2
- **Visualizations:** Recharts 2.10.3
- **Styling:** Custom CSS

### Development Tools
- **Notebooks:** Jupyter (data analysis & model training)
- **Environment:** Python venv, Node.js 18+
- **Version Control:** Git

---

## System Architecture

```
┌─────────────────┐
│   React Frontend│
│  (Vite + React) │
└────────┬────────┘
         │ HTTP/REST
         │
┌────────▼────────────────────────────────────────┐
│              FastAPI Backend                    │
│  ┌──────────────────────────────────────────┐  │
│  │  Routes: /recommend, /search, /analytics │  │
│  └──────────────┬───────────────────────────┘  │
│                 │                                │
│  ┌──────────────▼───────────────────────────┐  │
│  │      Service Layer                        │  │
│  │  - Vector DB Service (Pinecone)          │  │
│  │  - Text Generator Service (GenAI)        │  │
│  │  - LangChain Service (RAG)               │  │
│  └──────────────┬───────────────────────────┘  │
└─────────────────┼──────────────────────────────┘
                  │
         ┌────────▼────────┐
         │  Pinecone       │
         │  Vector DB      │
         │  (384-dim)      │
         └─────────────────┘
```

**Data Flow:**
1. User enters query in React chat interface
2. Frontend sends POST request to `/recommend` endpoint
3. Backend generates query embedding (sentence-transformers)
4. Vector similarity search in Pinecone
5. GenAI generates conversational response
6. Results returned with product metadata
7. Frontend displays recommendations with images

---

## Project Structure

```
furniture-recommendation-system/
│
├── backend/                          # FastAPI backend
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                  # Application entry point
│   │   ├── config.py                # Configuration management
│   │   ├── services/
│   │   │   ├── vector_db_service.py      # Pinecone integration
│   │   │   ├── text_generator_service.py # GenAI service
│   │   │   └── langchain_service.py      # LangChain RAG
│   │   └── schemas/
│   │       └── schemas_product.py   # Pydantic models
│   ├── data/
│   │   ├── cleaned_products.csv     # Processed dataset
│   │   └── analytics_data.json      # Pre-computed analytics
│   ├── requirements.txt             # Python dependencies
│   ├── .env.example                 # Environment template
│   └── Dockerfile                   # Docker configuration
│
├── frontend/                        # React frontend
│   ├── src/
│   │   ├── components/
│   │   │   ├── ChatInterface.jsx
│   │   │   ├── ProductCard.jsx
│   │   │   └── AnalyticsDashboard.jsx
│   │   ├── services/
│   │   │   └── api.js              # API client
│   │   ├── styles/
│   │   │   └── App.css             # Styling
│   │   ├── App.jsx                 # Main component
│   │   └── index.jsx               # Entry point
│   ├── public/
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
│
├── notebooks/                       # Jupyter notebooks
│   ├── 01_data_analytics_detailed.ipynb
│   └── 02_model_training_evaluation.ipynb
│
├── docs/                           # Documentation
│   ├── API_DOCUMENTATION.md
│   ├── DEPLOYMENT_GUIDE.md
│   └── ARCHITECTURE.md
│
├── .gitignore
├── README.md
└── LICENSE
```

---

## Installation

### Prerequisites

**Required Software:**
- Python 3.9 or higher
- Node.js 18.0 or higher
- npm 9.0 or higher
- Git

**API Keys:**
- Pinecone API key (free tier available at [pinecone.io](https://www.pinecone.io/))

### Backend Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/furniture-recommendation-system.git
   cd furniture-recommendation-system/backend
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your credentials
   ```

5. **Run the backend server:**
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

   Backend will be available at: `http://localhost:8000`

### Frontend Setup

1. **Navigate to frontend directory:**
   ```bash
   cd ../frontend
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Configure API URL:**
   Create `.env.local`:
   ```
   VITE_API_URL=http://localhost:8000
   ```

4. **Start development server:**
   ```bash
   npm run dev
   ```

   Frontend will be available at: `http://localhost:3000`

---

## Configuration

### Backend Configuration (.env)

```env
# Pinecone Configuration
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=us-west1-gcp-free
PINECONE_INDEX_NAME=furniture-products

# Server Configuration
BACKEND_PORT=8000
FRONTEND_URL=http://localhost:3000

# Model Configuration
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
GENAI_MODEL=google/flan-t5-small

# Optional: Logging
LOG_LEVEL=INFO
```

### Frontend Configuration (.env.local)

```env
VITE_API_URL=http://localhost:8000
```

---

## Usage

### Starting the System

1. **Start Backend:**
   ```bash
   cd backend
   source venv/bin/activate
   uvicorn app.main:app --reload
   ```

2. **Start Frontend:**
   ```bash
   cd frontend
   npm run dev
   ```

3. **Access the application:**
   Open browser to `http://localhost:3000`

### Using the Chat Interface

1. Enter furniture queries in natural language:
   - "Show me comfortable office chairs under $200"
   - "I need a wooden dining table for 6 people"
   - "Modern minimalist desk with storage"

2. View recommendations with:
   - Product images
   - Prices and ratings
   - Detailed descriptions
   - Similarity scores

3. Apply filters:
   - Category selection
   - Price range
   - Material/color preferences

### Viewing Analytics

Navigate to `/analytics` to view:
- Price distribution charts
- Category breakdown
- Popular materials and colors
- Brand insights
- Summary statistics

---

## API Documentation

### Base URL
```
http://localhost:8000
```

### Endpoints

#### 1. Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0"
}
```

#### 2. Get Recommendations
```http
POST /recommend
```

**Request Body:**
```json
{
  "query": "comfortable office chair",
  "filters": {
    "category": "Office Furniture",
    "min_price": 50,
    "max_price": 300
  },
  "top_k": 5
}
```

**Response:**
```json
{
  "query": "comfortable office chair",
  "recommendations": [
    {
      "uniq_id": "prod_001",
      "title": "Ergonomic Office Chair",
      "description": "...",
      "price": 199.99,
      "category": "Office Furniture",
      "image_url": "https://...",
      "similarity_score": 0.89
    }
  ],
  "response_text": "Here are some great office chairs...",
  "processing_time_ms": 45
}
```

#### 3. Search Products
```http
POST /search
```

**Request Body:**
```json
{
  "query": "wooden table",
  "limit": 20
}
```

#### 4. Get Analytics
```http
GET /analytics
```

**Response:**
```json
{
  "price_distribution": [...],
  "category_breakdown": [...],
  "material_analysis": [...],
  "summary_stats": {...}
}
```

---

## Data Analytics

### Running the Analytics Notebook

1. **Install Jupyter:**
   ```bash
   pip install jupyter notebook
   ```

2. **Start Jupyter:**
   ```bash
   jupyter notebook notebooks/01_data_analytics_detailed.ipynb
   ```

3. **Run all cells** to perform:
   - Exploratory Data Analysis (EDA)
   - Missing value analysis
   - Price distribution analysis
   - Category/brand/material analysis
   - Data preprocessing
   - Analytics data generation

### Key Insights from Analysis

- **Dataset Size:** 312 furniture products
- **Missing Data:** 49% descriptions, 31% prices (handled via imputation)
- **Price Range:** $0.60 - $349.00 (avg: $67.63)
- **Top Category:** Home & Kitchen (81% of products)
- **Common Materials:** Wood, Engineered Wood, Metal

---

## Model Training

### Running the Training Notebook

1. **Start Jupyter:**
   ```bash
   jupyter notebook notebooks/02_model_training_evaluation.ipynb
   ```

2. **Run all cells** to:
   - Load sentence-transformer model
   - Generate embeddings (384-dim vectors)
   - Compute similarity matrix
   - Evaluate recommendation quality
   - Benchmark inference speed
   - Save model artifacts

### Model Performance

**Embedding Generation:**
- Model: sentence-transformers/all-MiniLM-L6-v2
- Dimension: 384
- Time: ~20ms per product
- Quality: Good semantic clustering

**Recommendation System:**
- Query embedding: 15-25ms
- Vector search: <5ms
- Total latency: <50ms
- Throughput: ~20-30 recommendations/sec

**Quality Metrics:**
- Diversity score: 0.4-0.6
- Top result similarity: 0.7-0.9
- Rank 10 similarity: 0.3-0.5

---

## Deployment

### Local Deployment

Follow instructions in [Installation](#installation) section.

### Cloud Deployment (Render + Vercel)

#### Backend on Render

1. Push code to GitHub
2. Create Render Web Service
3. Configure:
   - Root Directory: `backend`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
4. Add environment variables (Pinecone API key, etc.)
5. Deploy

#### Frontend on Vercel

1. Import GitHub repository to Vercel
2. Configure:
   - Framework: Vite
   - Root Directory: `frontend`
   - Build Command: `npm run build`
   - Output Directory: `dist`
3. Add environment variable: `VITE_API_URL` (Render backend URL)
4. Deploy

**Detailed deployment guide:** See `docs/DEPLOYMENT_GUIDE.md`

---

## Performance Metrics

### Backend Performance
- API Response Time: 40-60ms (avg)
- Embeddings Generation: 6-10s (full dataset)
- Vector Search: <5ms (312 products)
- Memory Usage: ~500MB

### Frontend Performance
- Initial Load: 1-2 seconds
- Chat Response: <100ms (after API)
- Analytics Dashboard: <500ms load

### Scalability
- Current: 312 products
- Tested: Up to 10,000 products
- Recommended: 50,000+ products (requires optimization)

---






