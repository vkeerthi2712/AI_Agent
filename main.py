from __future__ import annotations
import io
import os
import uuid
import json
import re
from enum import Enum
from typing import Any, Dict, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
import seaborn as sns
from functools import lru_cache

# ---------------------------
# Load API Key
# ---------------------------
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError("OPENROUTER_API_KEY not found in .env file")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

# ---------------------------
# FastAPI Setup
# ---------------------------
app = FastAPI(title="LLM Chart & Dataset QA Service", version="2.5.4")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])
DATASETS: Dict[str, pd.DataFrame] = {}

# ---------------------------
# Models
# ---------------------------
class ChartType(str, Enum):
    line = "line"
    bar = "bar"
    scatter = "scatter"
    hist = "hist"
    box = "box"
    heatmap = "heatmap"
    violin = "violin"
    swarm = "swarm"
    pie = "pie"
    volcano = "volcano"
    tga = "tga"
    dtg = "dtg"
    cluster_heatmap = "cluster_heatmap"
    exp_curve = "exp_curve"
    log_curve = "log_curve"
    smooth_curve = "smooth_curve"
    logistic_regression = "logistic_regression"
    kaplan_meier = "kaplan_meier"

class OutputFormat(str, Enum):
    png = "png"
    svg = "svg"

class ChartSpec(BaseModel):
    chart_type: ChartType
    x: Optional[str] = None
    y: Optional[str] = None
    group: Optional[str] = None
    bins: Optional[int] = None
    title: Optional[str] = None
    x_label: Optional[str] = None
    y_label: Optional[str] = None
    color: Optional[str] = None
    size: Optional[str] = None
    fdr_col: Optional[str] = None

class RenderRequest(BaseModel):
    dataset_id: str
    instruction: str
    format: OutputFormat = OutputFormat.png

class AskRequest(BaseModel):
    dataset_id: str
    question: str

# ---------------------------
# Helpers
# ---------------------------
@lru_cache(maxsize=1000)
def normalize_column(col: Optional[str], df_columns: tuple) -> Optional[str]:
    """Match user-given column names against normalized df.columns with caching."""
    if not col:
        return None
    target = col.strip().lower()
    for c in df_columns:
        if c == target:
            return c
        def clean(s: str) -> str:
            return re.sub(r"[^a-z0-9]", "", s)
        if clean(c) == clean(target):
            return c
    return None

def pick_default_numeric(df: pd.DataFrame, n=1):
    """Select default numeric columns."""
    nums = df.select_dtypes(include=[np.number]).columns.tolist()
    return nums[:n] if len(nums) >= n else []

# ---------------------------
# LLM Router for Plotting
# ---------------------------
def llm_router(instruction: str, df: pd.DataFrame) -> Dict[str, Any]:
    system_prompt = (
        "You are an expert assistant that converts plotting instructions into a JSON object containing a ChartSpec and executable Python code for rendering the chart. "
        "Return ONLY valid JSON with two fields: 'spec' (matching the ChartSpec model) and 'code' (a string of Python code to render the chart). "
        "Supported chart types: [line, bar, scatter, hist, box, violin, swarm, pie, heatmap, volcano, tga, dtg, cluster_heatmap, exp_curve, log_curve, smooth_curve, logistic_regression, kaplan_meier]. "
        "Only use column names that exist in the dataset: " + ", ".join(df.columns.astype(str)) + ". "
        "The code must: "
        "- NOT include any import statements (use pre-provided modules: plt, sns, np, pd, re, curve_fit, UnivariateSpline, gradient, HTTPException). "
        "- Normalize df columns to lowercase stripped using: df.columns = [c.strip().lower() for c in df.columns]. "
        "- Use normalize_column and pick_default_numeric helpers if needed. "
        "- Handle fallback defaults for x, y, etc., using pick_default_numeric. "
        "- Create fig, ax = plt.subplots(figsize=(6, 4)) for most charts, except cluster_heatmap. "
        "- Implement the full chart logic, including any special handling like curve fitting or gradients. "
        "- Set titles, labels, legends as per spec, ensuring axis labels are set for heatmaps (e.g., ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')). "
        "- Save to buf: fig.tight_layout(); fig.savefig(buf, format=fmt.value, dpi=300); plt.close(fig). "
        "- For cluster_heatmap, use sns.clustermap, set row_cluster=True, col_cluster=True, and save the figure correctly. "
        "- For logistic_regression, ensure y values are cast to np.float64 for curve fitting and use np.min and np.max for range calculations. "
        "- For smooth_curve, sort x and y by x to ensure x is strictly increasing, remove duplicates, and use UnivariateSpline with a reasonable smoothing factor (e.g., s=len(unique_x)/2). "
        "- Raise HTTPException for errors (e.g., missing columns, invalid data, or insufficient unique x values for smooth_curve). "
        "Example output for smooth_curve: {\"spec\": {\"chart_type\": \"smooth_curve\", \"x\": \"signal\", \"y\": \"score\", \"group\": null, \"bins\": null, \"title\": \"Smooth Curve: Score vs Signal\", \"x_label\": \"Signal\", \"y_label\": \"Score\", \"color\": null, \"size\": null, \"fdr_col\": null}, "
        "\"code\": \"df.columns = [c.strip().lower() for c in df.columns]\\nif 'signal' not in df.columns or 'score' not in df.columns:\\n    raise HTTPException(400, detail='Missing required columns: signal or score')\\nxy = pd.DataFrame({'x': df['signal'], 'y': df['score']}).dropna().sort_values('x').drop_duplicates('x')\\nx = xy['x'].values\\ny = xy['y'].values\\nif len(x) < 2:\\n    raise HTTPException(400, detail='Insufficient unique x values for smooth curve')\\nspl = UnivariateSpline(x, y, s=len(x)/2)\\nx_smooth = np.linspace(np.min(x), np.max(x), 200)\\ny_smooth = spl(x_smooth)\\nfig, ax = plt.subplots(figsize=(6, 4))\\nax.scatter(df['signal'], df['score'], color=spec.color if spec.color else 'steelblue', alpha=0.7, label='Data')\\nax.plot(x_smooth, y_smooth, color=spec.color if spec.color else 'crimson', linewidth=2, label='Smooth curve')\\nax.set_title(spec.title if spec.title else 'Smooth Curve: Score vs Signal')\\nax.set_xlabel(spec.x_label if spec.x_label else 'Signal')\\nax.set_ylabel(spec.y_label if spec.y_label else 'Score')\\nax.legend()\\nfig.tight_layout()\\nfig.savefig(buf, format=fmt.value, dpi=300)\\nplt.close(fig)\"}"
    )
    user_prompt = (
        f"Instruction: {instruction}\n"
        "Return a valid JSON with 'spec' (ChartSpec) and 'code' (full Python code string for rendering, without imports)."
    )
    content = ""  # Initialize content to avoid UnboundLocalError
    try:
        resp = client.chat.completions.create(
            model="deepseek/deepseek-r1:free",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
        )
        content = resp.choices[0].message.content
        print(f"Raw API response: {content}")
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:-3].strip()
        elif content.startswith("```"):
            content = content[3:-3].strip()
        match = re.match(r'\{(?:[^{}]|\{[^{}]*\})*\}', content)
        if not match:
            raise ValueError("No valid JSON object found in response")
        content = match.group(0)
        if not content:
            raise ValueError("Empty response from DeepSeek R1 after cleaning")
        result = json.loads(content)
        if 'spec' not in result or 'code' not in result:
            raise ValueError("Invalid response: missing 'spec' or 'code'")
        return result
    except json.JSONDecodeError as e:
        print(f"JSONDecodeError: {e}, Raw content: {content}")
        raise HTTPException(500, detail=f"Failed to parse DeepSeek R1 response as JSON: {e}")
    except Exception as e:
        print(f"API Error: {e}, Raw content: {content}")
        raise HTTPException(500, detail=f"DeepSeek R1 API failed: {e}")

# ---------------------------
# LLM Router for Dataset Questions
# ---------------------------
def llm_dataset_qa(question: str, df: pd.DataFrame) -> str:
    question_lower = question.lower().strip()
    
    # Handle specific column mean request (e.g., "show mean of age")
    mean_match = re.match(r"show\s+mean\s+of\s+([a-zA-Z0-9_]+)", question_lower)
    if mean_match:
        col = mean_match.group(1)
        normalized_col = normalize_column(col, tuple(df.columns))
        if normalized_col and normalized_col in df.columns:
            if pd.api.types.is_numeric_dtype(df[normalized_col]):
                mean_value = df[normalized_col].mean()
                return f"Mean of {normalized_col}: {mean_value:.2f}"
            else:
                return f"Column '{normalized_col}' is not numeric."
        else:
            return f"Column '{col}' not found in dataset."

    # Handle request for means in table format (e.g., "show mean in table")
    if "show mean in table" in question_lower:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if not numeric_cols.empty:
            means = df[numeric_cols].mean()
            table = "Mean values for numeric columns:\n"
            table += "| Column | Mean |\n"
            table += "|--------|------|\n"
            for col, mean in means.items():
                table += f"| {col} | {mean:.2f} |\n"
            return table
        else:
            return "No numeric columns found in dataset."

    # Handle EDA analysis request (e.g., "generate eda analysis of dataset")
    if "generate eda analysis" in question_lower:
        eda = []
        eda.append(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
        eda.append("\nColumn Data Types:")
        for col, dtype in df.dtypes.items():
            eda.append(f"- {col}: {dtype}")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if not numeric_cols.empty:
            eda.append("\nSummary Statistics for Numeric Columns:")
            stats = df[numeric_cols].describe().T
            stats_table = "| Column | Count | Mean | Std | Min | 25% | 50% | 75% | Max |\n"
            stats_table += "|--------|-------|------|-----|-----|-----|-----|-----|-----|\n"
            for col, row in stats.iterrows():
                stats_table += f"| {col} | {row['count']:.0f} | {row['mean']:.2f} | {row['std']:.2f} | {row['min']:.2f} | {row['25%']:.2f} | {row['50%']:.2f} | {row['75%']:.2f} | {row['max']:.2f} |\n"
            eda.append(stats_table)
        
        eda.append("\nMissing Values:")
        missing = df.isnull().sum()
        if missing.sum() == 0:
            eda.append("No missing values.")
        else:
            for col, count in missing.items():
                if count > 0:
                    eda.append(f"- {col}: {count} missing values")
        
        return "\n".join(eda)

    # Existing QA logic for other questions
    system_prompt = (
        "You are an expert assistant that answers questions about a dataset based on its structure and content. "
        "The dataset has columns: " + ", ".join(df.columns.astype(str)) + ". "
        "Provide a concise, accurate answer to the user's question. For questions requiring computation (e.g., summary statistics, counts, or correlations), perform the analysis using pandas and return the result as a string. "
        "For questions about column names, data types, or row counts, provide direct answers based on the DataFrame's metadata. "
        "Do not generate plots or code; return only the answer as a string. "
        "Example questions and answers: "
        "- Q: 'What are the columns?' A: 'Columns: col1, col2, col3' "
        "- Q: 'How many rows?' A: 'Number of rows: 100' "
        "- Q: 'Show summary statistics' A: 'Summary statistics:\ncol1: mean=10.5, std=2.3, min=5, max=15\ncol2: mean=20.1, std=4.2, min=10, max=30' "
    )
    user_prompt = (
        f"Question: {question}\n"
        f"Dataset info: {df.shape[0]} rows, columns: {', '.join(df.columns.astype(str))} with dtypes: {df.dtypes.to_dict()}\n"
        "Answer the question concisely as a string, performing any necessary computations on the dataset."
    )
    content = ""  # Initialize content to avoid UnboundLocalError
    try:
        resp = client.chat.completions.create(
            model="deepseek/deepseek-r1:free",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
        )
        content = resp.choices[0].message.content
        return content.strip()
    except Exception as e:
        print(f"QA API Error: {e}, Raw content: {content}")
        raise HTTPException(500, detail=f"Failed to process dataset question: {e}")

# ---------------------------
# Chart Rendering
# ---------------------------
def render_chart(df: pd.DataFrame, result: Dict[str, Any], fmt: OutputFormat) -> bytes:
    spec_dict = result['spec']
    code = result['code']
    spec = ChartSpec(**spec_dict)
    buf = io.BytesIO()
    from scipy.optimize import curve_fit
    from scipy.interpolate import UnivariateSpline
    safe_globals = {
        '__builtins__': {'len': len, 'min': min, 'max': max},
        'io': io,
        'np': np,
        'pd': pd,
        'plt': matplotlib.pyplot,
        'sns': sns,
        're': re,
        'curve_fit': curve_fit,
        'UnivariateSpline': UnivariateSpline,
        'gradient': np.gradient,
        'df': df,
        'spec': spec,
        'buf': buf,
        'fmt': fmt,
        'normalize_column': normalize_column,
        'pick_default_numeric': pick_default_numeric,
        'ChartType': ChartType,
        'HTTPException': HTTPException,
    }
    try:
        exec(code, safe_globals)
        buf.seek(0)
        return buf.read()
    except Exception as e:
        print(f"Execution error: {e}")
        raise HTTPException(500, detail=f"Failed to execute AI-generated chart code: {str(e)}")
    finally:
        plt.close('all')

# ---------------------------
# Endpoints
# ---------------------------
@app.post("/upload")
async def upload(file: UploadFile = File(...)) -> Dict[str, Any]:
    try:
        content = await file.read()
        if file.filename.lower().endswith(".csv"):
            df = pd.read_csv(io.BytesIO(content))
        elif file.filename.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(content))
        else:
            raise HTTPException(400, detail="Only CSV/XLSX supported.")
        df.columns = [c.strip().lower() for c in df.columns]
        dataset_id = str(uuid.uuid4())
        DATASETS[dataset_id] = df
        return {"dataset_id": dataset_id, "columns": df.columns.tolist()}
    except Exception as e:
        raise HTTPException(400, detail=f"Failed to read file: {e}")

@app.post("/plot")
def plot(req: RenderRequest):
    df = DATASETS.get(req.dataset_id)
    if df is None:
        raise HTTPException(404, detail="dataset_id not found")
    result = llm_router(req.instruction, df)
    img = render_chart(df, result, req.format)
    media_type = "image/png" if req.format == OutputFormat.png else "image/svg+xml"
    return StreamingResponse(io.BytesIO(img), media_type=media_type)

@app.post("/ask")
def ask(req: AskRequest):
    df = DATASETS.get(req.dataset_id)
    if df is None:
        raise HTTPException(404, detail="dataset_id not found")
    answer = llm_dataset_qa(req.question, df)
    return {"answer": answer}