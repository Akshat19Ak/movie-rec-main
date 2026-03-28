# Movie Recommendation System

An end-to-end movie recommendation system using NLP (TF-IDF), FastAPI for the backend API, and Streamlit for the frontend interface.

## Project Structure

- `main.py`: FastAPI backend server.
- `app.py`: Streamlit frontend application.
- `movies.ipynb`: Jupyter notebook for data exploration and model training.
- `movies_metadata.csv`: Dataset containing movie information.
- `*.pkl`: Pre-computed TF-IDF matrices and data structures for fast recommendations.

## Prerequisites

- Python 3.8 or higher.
- A TMDB API Key (Optional but recommended for fetching posters and details).

## Setup and Installation

Follow these steps to get the project running locally:

### 1. Clone the Repository
```bash
git clone <repository-url>
cd movie-rec-main
```

### 2. Create a Virtual Environment (Recommended)
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configuration
Create a `.env` file in the root directory and add your TMDB API Key:
```env
TMDB_API_KEY=your_api_key_here
```
*Note: If you don't have a key, you can get one from [themoviedb.org](https://www.themoviedb.org/settings/api).*

### 5. Prepare Data (Optional)
The repository comes with pre-computed pickle files (`df.pkl`, `tfidf.pkl`, etc.). If you want to re-train the model or explore the data:
- Open `movies.ipynb` in VS Code or Jupyter.
- Run all cells to process `movies_metadata.csv` and generate the pickle files.

## Running the Application

You need to run both the backend (FastAPI) and the frontend (Streamlit).

### Step 1: Start the Backend API
In your terminal (with venv activated):
```bash
uvicorn main:app --reload
```
The API will be available at `http://127.0.0.1:8000`. You can view the interactive documentation at `http://127.0.0.1:8000/docs`.

### Step 2: Start the Frontend UI
Open a **new** terminal, activate the virtual environment, and run:
```bash
streamlit run app.py
```
The Streamlit interface will open in your default browser at `http://localhost:8501`.

## How it Works
1. **TF-IDF Vectorization**: Analyzes movie overviews and metadata to find similar movies.
2. **FastAPI**: Serves as the recommendation engine.
3. **Streamlit**: Provides a modern UI for users to search for movies and see recommendations.
4. **TMDB Integration**: Fetches real-time posters and detailed movie information.
