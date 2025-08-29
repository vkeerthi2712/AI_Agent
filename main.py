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
import umap
from sklearn.metrics import roc_curve, auc
from sklearn.decomposition import PCA

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
    umap = "umap"
    roc = "roc"
    pca = "pca"

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

def pick_default_numeric(df: pd.DataFrame, n=1, exclude=None):
    """Select default numeric columns, excluding specified columns if provided."""
    nums = df.select_dtypes(include=[np.number]).columns.tolist()
    if exclude is not None:
        exclude = [col.strip().lower() for col in (exclude if isinstance(exclude, list) else [exclude])]
        nums = [col for col in nums if col not in exclude]
    return nums[:n] if len(nums) >= n else []

# ---------------------------
# LLM Router for Plotting
# ---------------------------
def llm_router(instruction: str, df: pd.DataFrame) -> Dict[str, Any]:
    system_prompt = (
        "You are an expert assistant that converts plotting instructions into a JSON object containing a ChartSpec and executable Python code for rendering the chart. "
        "Return ONLY valid JSON with two fields: 'spec' (matching the ChartSpec model) and 'code' (a string of Python code to render the chart). "
        "Supported chart types: [line, bar, scatter, hist, box, violin, swarm, pie, heatmap, volcano, tga, dtg, cluster_heatmap, exp_curve, log_curve, smooth_curve, logistic_regression, kaplan_meier, umap, roc, pca]. "
        "Only use column names that exist in the dataset: " + ", ".join(df.columns.astype(str)) + ". "
        "The code must: "
        "- NOT include any import statements (use pre-provided modules: plt, sns, np, pd, re, curve_fit, UnivariateSpline, gradient, HTTPException, umap, roc_curve, auc, PCA). "
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
        "- For umap, use umap.UMAP to reduce numeric columns to 2D, create a scatter plot, and color by 'group' if specified. Drop non-numeric columns and handle missing values with dropna(). "
        "- For roc, use roc_curve and auc from sklearn.metrics, expect 'y' as binary labels and 'x' as predicted probabilities, and plot the ROC curve with AUC in the legend. "
        "- For pca, use PCA from sklearn.decomposition to reduce numeric columns to 2D, create a scatter plot, and color by 'group' if specified. Drop non-numeric columns and handle missing values with dropna(). "
        "- Raise HTTPException for errors (e.g., missing columns, invalid data, insufficient unique x values for smooth_curve, or non-binary labels for roc). "
        "Example output for umap: {\"spec\": {\"chart_type\": \"umap\", \"x\": null, \"y\": null, \"group\": \"category\", \"bins\": null, \"title\": \"UMAP Projection\", \"x_label\": \"UMAP1\", \"y_label\": \"UMAP2\", \"color\": null, \"size\": null, \"fdr_col\": null}, "
        "\"code\": \"df.columns = [c.strip().lower() for c in df.columns]\\nnum_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]\\nif len(num_cols) < 2:\\n    raise HTTPException(400, detail='Insufficient numeric columns for UMAP')\\ndata = df[num_cols].dropna()\\nif data.empty:\\n    raise HTTPException(400, detail='No valid data after dropping missing values')\\nreducer = umap.UMAP(n_components=2, random_state=42)\\nembedding = reducer.fit_transform(data)\\nfig, ax = plt.subplots(figsize=(6, 4))\\nif spec.group and spec.group in df.columns:\\n    groups = df[spec.group].loc[data.index]\\n    for g in groups.unique():\\n        idx = groups == g\\n        ax.scatter(embedding[idx, 0], embedding[idx, 1], label=g, alpha=0.7)\\n    ax.legend()\\nelse:\\n    ax.scatter(embedding[:, 0], embedding[:, 1], color=spec.color if spec.color else 'steelblue', alpha=0.7)\\nax.set_title(spec.title if spec.title else 'UMAP Projection')\\nax.set_xlabel(spec.x_label if spec.x_label else 'UMAP1')\\nax.set_ylabel(spec.y_label if spec.y_label else 'UMAP2')\\nfig.tight_layout()\\nfig.savefig(buf, format=fmt.value, dpi=300)\\nplt.close(fig)\"}"
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

    # Helper function to format table
    def format_table(headers: list, rows: list) -> str:
        table = "| " + " | ".join(headers) + " |\n"
        table += "| " + " | ".join(["---"] * len(headers)) + " |\n"
        for row in rows:
            table += "| " + " | ".join([f"{x:.2f}" if isinstance(x, (int, float)) else str(x) for x in row]) + " |\n"
        return table

    # Handle specific column statistic requests (e.g., "show mean of age", "show median of age", "show mode of age")
    stat_match = re.match(r"show\s+(mean|median|mode)\s+of\s+([a-zA-Z0-9_]+)", question_lower)
    if stat_match:
        stat_type, col = stat_match.groups()
        normalized_col = normalize_column(col, tuple(df.columns))
        if normalized_col and normalized_col in df.columns:
            if pd.api.types.is_numeric_dtype(df[normalized_col]):
                if stat_type == "mean":
                    value = df[normalized_col].mean()
                elif stat_type == "median":
                    value = df[normalized_col].median()
                elif stat_type == "mode":
                    mode_series = df[normalized_col].mode()
                    value = mode_series[0] if not mode_series.empty else "No mode"
                return format_table(
                    headers=["Column", stat_type.capitalize()],
                    rows=[[normalized_col, value]]
                )
            else:
                return f"Column '{normalized_col}' is not numeric."
        else:
            return f"Column '{col}' not found in dataset."

    # Handle request for statistics in table format (e.g., "show mean in table", "show median in table", "show mode in table")
    stat_table_match = re.match(r"show\s+(mean|median|mode)\s+in\s+table", question_lower)
    if stat_table_match:
        stat_type = stat_table_match.group(1)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if not numeric_cols.empty:
            if stat_type == "mean":
                values = df[numeric_cols].mean()
            elif stat_type == "median":
                values = df[numeric_cols].median()
            elif stat_type == "mode":
                values = df[numeric_cols].mode().iloc[0] if not df[numeric_cols].mode().empty else pd.Series(index=numeric_cols)
            rows = [[col, values.get(col, "No mode" if stat_type == "mode" else np.nan)] for col in numeric_cols]
            return format_table(
                headers=["Column", stat_type.capitalize()],
                rows=rows
            )
        else:
            return "No numeric columns found in dataset."

    # Handle EDA analysis request (e.g., "generate eda analysis of dataset")
    if "generate eda analysis" in question_lower:
        eda = []

        # Dataset Shape
        eda.append("**Dataset Shape**:")
        rows = [["Rows", df.shape[0]], ["Columns", df.shape[1]]]
        eda.append(format_table(["Property", "Value"], rows))
        
        # Column Data Types
        eda.append("\n**Column Data Types**:")
        rows = [[col, str(dtype)] for col, dtype in df.dtypes.items()]
        eda.append(format_table(["Column", "Data Type"], rows))
        
        # Summary Statistics for Numeric Columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if not numeric_cols.empty:
            eda.append("\n**Summary Statistics for Numeric Columns**:")
            stats = df[numeric_cols].describe().T
            headers = ["Column", "Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"]
            rows = [
                [col, row['count'], row['mean'], row['std'], row['min'], row['25%'], row['50%'], row['75%'], row['max']]
                for col, row in stats.iterrows()
            ]
            eda.append(format_table(headers, rows))
        
            # Median Values
            eda.append("\n**Median Values**:")
            medians = df[numeric_cols].median()
            rows = [[col, medians[col]] for col in numeric_cols]
            eda.append(format_table(["Column", "Median"], rows))
            
            # Mode Values
            eda.append("\n**Mode Values**:")
            modes = df[numeric_cols].mode().iloc[0] if not df[numeric_cols].mode().empty else pd.Series(index=numeric_cols)
            rows = [[col, modes.get(col, "No mode")] for col in numeric_cols]
            eda.append(format_table(["Column", "Mode"], rows))
        
        # Missing Values
        eda.append("\n**Missing Values**:")
        missing = df.isnull().sum()
        rows = [[col, count] for col, count in missing.items()]
        if missing.sum() == 0:
            rows = [["All Columns", "No missing values"]]
        eda.append(format_table(["Column", "Missing Count"], rows))
        
        return "\n".join(eda)

    # Existing QA logic for other questions
    system_prompt = (
        "You are an expert assistant that answers questions about a dataset based on its structure and content. "
        "The dataset has columns: " + ", ".join(df.columns.astype(str)) + ". "
        "Provide a concise, accurate answer to the user's question. For questions requiring computation (e.g., summary statistics, counts, or correlations), perform the analysis using pandas and return the result as a markdown table if appropriate, or as a string otherwise. "
        "For statistical questions (e.g., mean, median, mode, std, min, max), return results in a markdown table with columns [Column, Statistic]. "
        "For questions about column names, data types, or row counts, provide direct answers as strings. "
        "Do not generate plots or code; return only the answer as a string. "
        "Example table for stats: | Column | Statistic |\n|--------|-----------|\n| col1   | 10.5      |\n"
        "Example non-table answers: "
        "- Q: 'What are the columns?' A: 'Columns: col1, col2, col3' "
        "- Q: 'How many rows?' A: 'Number of rows: 100' "
    )
    user_prompt = (
        f"Question: {question}\n"
        f"Dataset info: {df.shape[0]} rows, columns: {', '.join(df.columns.astype(str))} with dtypes: {df.dtypes.to_dict()}\n"
        "Answer the question concisely as a string, using a markdown table for statistical computations where appropriate."
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
        'umap': umap,
        'roc_curve': roc_curve,
        'auc': auc,
        'PCA': PCA,
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