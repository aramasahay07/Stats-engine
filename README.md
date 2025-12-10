# AI Data Lab Backend

Advanced data transformation and analysis engine with 60+ intelligent transforms.

## 🚀 Features

- **File Upload**: CSV and Excel support
- **Smart Type Detection**: Automatic column type inference
- **60+ Transformations**: Date, numeric, text, and categorical transforms
- **Statistical Analysis**: Correlation, t-tests, ANOVA, regression
- **REST API**: FastAPI-powered endpoints

## 📋 API Endpoints

### Health Check
```bash
GET /health
```

### Upload Dataset
```bash
POST /upload
Content-Type: multipart/form-data

Returns: {
  "session_id": "uuid",
  "n_rows": 1000,
  "n_cols": 10,
  "schema": [...],
  "descriptives": [...]
}
```

### Run Analysis
```bash
GET /analysis/{session_id}

Returns: {
  "correlation": {...},
  "tests": [...],
  "regression": {...}
}
```

## 🛠️ Local Development

### Prerequisites
- Python 3.11+
- pip

### Setup
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/ai-datalab-backend.git
cd ai-datalab-backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn main:app --reload
```

Server will be available at: `http://localhost:8000`

API documentation at: `http://localhost:8000/docs`

## 📦 Tech Stack

- **FastAPI**: Modern, fast web framework
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **SciPy**: Statistical functions
- **Statsmodels**: Advanced statistics
- **Scikit-learn**: Machine learning utilities

## 🌐 Deployment

### Deploy to Render

1. Push code to GitHub
2. Connect repository to Render
3. Render will auto-detect `render.yaml`
4. Deploy!

### Environment Variables

No environment variables required for basic setup.

## 📊 Transform Categories

### Date/Time Transforms
- Extract month, year, quarter, weekday
- Calculate fiscal periods
- Age calculations
- Time-based features

### Numeric Transforms
- Scaling (standard, min-max, robust)
- Binning and discretization
- Mathematical operations
- Statistical transformations

### Text Transforms
- Case conversion
- String extraction
- Length and character counts
- Email/URL parsing

### Categorical Transforms
- Encoding (one-hot, label, frequency)
- Grouping and consolidation
- Top-N selection

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first.

## 📄 License

MIT License

## 👥 Authors

Your Name - [GitHub Profile](https://github.com/YOUR_USERNAME)

## 🐛 Known Issues

- Session data is stored in memory (resets on server restart)
- For production, implement persistent storage

## 🗺️ Roadmap

- [ ] Add database persistence
- [ ] Implement user authentication
- [ ] Add more ML algorithms
- [ ] Create data visualization endpoints
- [ ] Add data export functionality
