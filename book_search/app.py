import os
import re
import json
import time
import threading
from collections import deque
from dataclasses import dataclass
from dotenv import load_dotenv
from typing import Dict, Optional, List, Tuple, Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# --- 추가된 부분 ---
from starlette.staticfiles import StaticFiles
from starlette.responses import FileResponse
# --------------------

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

import google.generativeai as genai
from google.generativeai import types

# =========================
# Rate limiter (sliding window)
# =========================

load_dotenv()

class SlidingWindowRateLimiter:
    def __init__(self, limit: int, window_seconds: float):
        self.limit = int(limit)
        self.window = float(window_seconds)
        self._ts = deque()  # timestamps (monotonic)
        self._lock = threading.Lock()

    def try_acquire(self) -> Tuple[bool, int, float]:
        """
        Returns (allowed, remaining, retry_after_seconds).
        remaining is after consuming this request when allowed.
        """
        now = time.monotonic()
        with self._lock:
            # prune old
            while self._ts and (now - self._ts[0]) > self.window:
                self._ts.popleft()

            if len(self._ts) < self.limit:
                self._ts.append(now)
                remaining = self.limit - len(self._ts)
                return True, remaining, 0.0

            # not allowed
            retry_after = self.window - (now - self._ts[0]) if self._ts else self.window
            return False, 0, max(0.0, retry_after)

class GlobalRateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: FastAPI, limiter: SlidingWindowRateLimiter, paths: Optional[List[str]] = None):
        super().__init__(app)
        self.limiter = limiter
        self.paths = paths or []  # if empty -> apply to all paths

    async def dispatch(self, request, call_next):
        path_apply = (not self.paths) or any(request.url.path.startswith(p) for p in self.paths)
        headers_to_add = None

        if path_apply:
            allowed, remaining, retry_after = self.limiter.try_acquire()
            headers_to_add = {
                "X-RateLimit-Limit": str(self.limiter.limit),
                "X-RateLimit-Remaining": str(max(0, remaining)),
                "X-RateLimit-Window": f"{int(self.limiter.window)}s",
            }
            if not allowed:
                headers_to_add["Retry-After"] = str(int(retry_after))
                return JSONResponse(
                    status_code=429,
                    headers=headers_to_add,
                    content={
                        "detail": "Too Many Requests. Rate limit exceeded.",
                        "rate_limit": self.limiter.limit,
                        "window_seconds": int(self.limiter.window),
                        "retry_after_seconds": int(retry_after),
                    },
                )

        response = await call_next(request)
        if headers_to_add:
            for k, v in headers_to_add.items():
                response.headers[k] = v
        return response

# =========================
# FastAPI app
# =========================
app = FastAPI(title="Book Search API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limit: default 100 req/min; override via env RATE_LIMIT_PER_MINUTE
rate_limit_per_min = int(os.getenv("RATE_LIMIT_PER_MINUTE", "100"))
rate_limiter = SlidingWindowRateLimiter(limit=rate_limit_per_min, window_seconds=60.0)
# /search 엔드포인트에만 적용
app.add_middleware(GlobalRateLimitMiddleware, limiter=rate_limiter, paths=["/search"])

# =========================
# Helpers and caches
# =========================
def configure_genai(api_key: Optional[str] = None):
    """Configures the Gemini API key."""
    key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY/GOOGLE_API_KEY not set and no api_key provided.")
    genai.configure(api_key=key)

def l2_normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms

def extract_title_from_text(txt: str) -> str:
    if not isinstance(txt, str):
        return ""
    head = txt[:2000]
    m = re.search(r"(?:^|\n)\s*(?:제목|Title|title)\s*:\s*(.*)", head)
    if m:
        t = m.group(1).strip()
        t = t.split("\n")[0]
        t = t.split("줄거리")[0]
        t = t.split("Synopsis")[0]
        return t.strip(" :\t-_/|")
    line = head.splitlines()[0] if "\n" in head else head
    return line[:200].strip()

def normalize_title_key(s: str) -> str:
    s = " ".join(str(s).split()).strip()
    return s.casefold()

def parse_sep(sep: Optional[str]) -> Tuple[Optional[str], Dict[str, Any]]:
    if sep is None or sep == "":
        return None, {"engine": "python"}  # auto-detect
    if sep in ("\\t", r"\t", "TAB", "tab"):
        return "\t", {}
    return sep, {}

index_lock = threading.Lock()
meta_lock = threading.Lock()
src_lock = threading.Lock()

@dataclass
class IndexItem:
    emb_normed: np.ndarray
    ids: np.ndarray
    dim: int
    model: str
    mtime: float
    path: str

@dataclass
class MetaItem:
    df: pd.DataFrame
    id_to_title: Dict[str, str]
    text_col: Optional[str]
    mtime: float
    path: str

@dataclass
class SourceItem:
    df: pd.DataFrame
    id_to_rowidx: Dict[str, int]
    id_col_used: Optional[str]
    mtime: float
    key: Tuple[str, Optional[str], Optional[str]]

index_cache: Dict[str, IndexItem] = {}
meta_cache: Dict[str, MetaItem] = {}
src_cache: Dict[Tuple[str, Optional[str], Optional[str]], SourceItem] = {}

def load_index(npz_path: str) -> IndexItem:
    if not os.path.exists(npz_path):
        raise HTTPException(status_code=400, detail=f"NPZ not found: {npz_path}")
    mtime = os.path.getmtime(npz_path)
    with index_lock:
        cached = index_cache.get(npz_path)
        if cached and cached.mtime == mtime:
            return cached
        try:
            data = np.load(npz_path, allow_pickle=False)
            emb = data["embeddings"].astype(np.float32)
            ids = data["ids"].astype(str)
            model = str(data.get("model", "gemini-embedding-001"))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load NPZ: {e}")
        emb_normed = l2_normalize_rows(emb)
        item = IndexItem(emb_normed=emb_normed, ids=ids, dim=emb.shape[1], model=model, mtime=mtime, path=npz_path)
        index_cache[npz_path] = item
        return item

def load_meta(meta_path: Optional[str]) -> Optional[MetaItem]:
    if not meta_path:
        return None
    if not os.path.exists(meta_path):
        raise HTTPException(status_code=400, detail=f"Meta CSV not found: {meta_path}")
    mtime = os.path.getmtime(meta_path)
    with meta_lock:
        cached = meta_cache.get(meta_path)
        if cached and cached.mtime == mtime:
            return cached
        try:
            df = pd.read_csv(meta_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to read meta CSV: {e}")
        if "id" not in df.columns:
            raise HTTPException(status_code=400, detail="Meta CSV must contain 'id' column.")
        candidates = [c for c in df.columns if c not in {"id", "title"}]
        text_col = candidates[0] if candidates else None
        id_to_title: Dict[str, str] = {}
        for _, row in df.iterrows():
            rid = str(row["id"])
            title_val = ""
            if "title" in df.columns and not pd.isna(row.get("title")):
                title_val = str(row.get("title", ""))
            elif text_col:
                title_val = extract_title_from_text(str(row.get(text_col, "")))
            id_to_title[rid] = title_val
        item = MetaItem(df=df, id_to_title=id_to_title, text_col=text_col, mtime=mtime, path=meta_path)
        meta_cache[meta_path] = item
        return item

def load_source(source_csv: str, source_id_col: Optional[str], source_sep: Optional[str], source_encoding: Optional[str]) -> SourceItem:
    if not os.path.exists(source_csv):
        raise HTTPException(status_code=400, detail=f"Source CSV not found: {source_csv}")
    key = (source_csv, source_sep or "", source_encoding or "")
    mtime = os.path.getmtime(source_csv)
    with src_lock:
        cached = src_cache.get(key)
        if cached and cached.mtime == mtime:
            return cached
        sep_val, extra = parse_sep(source_sep)
        read_kwargs: Dict[str, Any] = {}
        if sep_val is not None:
            read_kwargs["sep"] = sep_val
        else:
            read_kwargs.update(extra)
        if source_encoding:
            read_kwargs["encoding"] = source_encoding
        try:
            df = pd.read_csv(source_csv, dtype={"No": str, "id": str}, **read_kwargs)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to read source CSV: {e}")

        id_col_used = None
        if source_id_col and source_id_col in df.columns:
            id_col_used = source_id_col
            key_series = df[id_col_used].astype(str)
        elif "No" in df.columns:
            id_col_used = "No"
            key_series = df[id_col_used].astype(str)
        else:
            key_series = df.index.astype(str)

        id_to_rowidx: Dict[str, int] = {}
        for i, k in enumerate(key_series):
            k = str(k)
            if k not in id_to_rowidx:
                id_to_rowidx[k] = i

        item = SourceItem(df=df, id_to_rowidx=id_to_rowidx, id_col_used=id_col_used, mtime=mtime, key=key)
        src_cache[key] = item
        return item

def embed_query(query: str, model: str, out_dim: Optional[int]) -> np.ndarray:
    """Embeds the query using the configured genai module."""
    try:
        resp = genai.embed_content(
            model=model,
            content=query,
            task_type="retrieval_query",
            output_dimensionality=out_dim if out_dim is not None else None,
        )
        vec = np.array(resp['embedding'], dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec
        return vec / norm
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini embed error: {e}")

def truncate_strings_in_obj(obj: Dict[str, Any], max_chars: Optional[int]) -> Dict[str, Any]:
    if not max_chars or max_chars <= 0:
        return obj
    out = {}
    for k, v in obj.items():
        if isinstance(v, str) and len(v) > max_chars:
            out[k] = v[:max_chars] + "..."
        else:
            out[k] = v
    return out

# =========================
# API schema
# =========================
class SearchRequest(BaseModel):
    npz: str = Field(..., description="Path to NPZ index file.")
    meta: Optional[str] = Field(None, description="Path to metadata CSV (id + title + text). Enables title dedup.")
    query: str = Field(..., description="Search query.")
    top_k: int = Field(5, ge=1, le=1000, description="Number of unique results to return.")
    index_only: bool = Field(False, description="Return only IDs (and optional score/rank).")
    model: str = Field("gemini-embedding-001", description="Embedding model for the query.")
    output_dim: Optional[int] = Field(None, description="Ignored if index exists; query dim follows index dim.")
    api_key: Optional[str] = Field(None, description="Override API key per request.")
    no_dedup_title: bool = Field(False, description="Disable deduplication by title.")
    max_text_chars: Optional[int] = Field(None, description="Truncate long string fields in JSON to this many chars.")
    include_score: bool = Field(False, description="Include similarity score per result.")
    include_rank: bool = Field(True, description="Include rank per result.")
    source_csv: Optional[str] = Field(None, description="Original CSV path for JSON rows.")
    source_id_col: Optional[str] = Field(None, description="ID column in source CSV matching stored ids (e.g., 'No').")
    source_sep: Optional[str] = Field(None, description="Delimiter for source CSV, e.g., ',', '\\t'.")
    source_encoding: Optional[str] = Field(None, description="Encoding for source CSV, e.g., 'utf-8', 'cp949'.")
    json_cols: Optional[List[str]] = Field(None, description="Columns to include in JSON. Default: all source columns.")
    randomize: bool = Field(False, description="점수에 작은 Gumbel 노이즈를 더해 순위를 무작위화.")
    tau: Optional[float] = Field(0.01, description="무작위화 세기(temperature). 0이면 비활성. 보통 0.01~0.1 권장.")
    seed: Optional[int] = Field(None, description="재현 가능한 무작위화를 위한 시드.")

class SearchResponse(BaseModel):
    query: str
    top_k: int
    unique_returned: int
    dedup_by_title: bool
    randomize: bool
    tau: float
    seed: Optional[int]
    index_path: str
    meta_path: Optional[str]
    source_csv: Optional[str]
    results: List[Dict[str, Any]]

# =========================
# Endpoints
# =========================
@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    # Load index
    idx = load_index(req.npz)

    # Configure API key and embed query
    configure_genai(req.api_key)
    q = embed_query(req.query, req.model, out_dim=idx.dim)


    # Similarity (cosine because both normalized)
    scores = idx.emb_normed @ q
    scores_for_sort = scores.copy()
    if req.randomize and req.tau > 0:
        rng = np.random.default_rng()
        noise = rng.gumbel(loc=0.0, scale=req.tau, size=scores.shape)
        scores_for_sort = scores + noise
    idx_sorted = np.argsort(-scores_for_sort)

    # Meta (for title-dedup)
    meta_item = load_meta(req.meta) if req.meta else None
    dedup_by_title = (meta_item is not None) and (not req.no_dedup_title)

    # Select top-k with title dedup
    selected: List[int] = []
    seen_keys = set()
    for i in idx_sorted:
        rid = str(idx.ids[i])
        key = rid
        if dedup_by_title and meta_item and rid in meta_item.id_to_title:
            title_val = meta_item.id_to_title.get(rid, "")
            key = normalize_title_key(title_val) if title_val else rid
        if key in seen_keys:
            continue
        seen_keys.add(key)
        selected.append(i)
        if len(selected) >= req.top_k:
            break

    # Build JSON results
    results: List[Dict[str, Any]] = []

    # Source CSV rows
    src_item: Optional[SourceItem] = None
    if req.source_csv:
        src_item = load_source(req.source_csv, req.source_id_col, req.source_sep, req.source_encoding)

    # index_only -> return id (+score/rank)
    if req.index_only:
        for rank, i in enumerate(selected, 1):
            rid = str(idx.ids[i])
            obj: Dict[str, Any] = {"id": rid}
            if req.include_score:
                obj["_score"] = float(scores[i])
            if req.include_rank:
                obj["_rank"] = rank
            results.append(obj)
        # index_only인 경우 여기서 함수가 종료됩니다.
        return SearchResponse(query=req.query, top_k=req.top_k, unique_returned=len(results), dedup_by_title=dedup_by_title, randomize=req.randomize, tau=req.tau, seed=req.seed, index_path=req.npz, meta_path=req.meta, source_csv=req.source_csv, results=results)

    # Otherwise include source CSV row if available, else fallback to meta
    for rank, i in enumerate(selected, 1):
        rid = str(idx.ids[i])
        obj: Dict[str, Any] = {} # 상세 정보가 있을 때만 id를 포함하도록 초기화

        # 상세 정보 소스(CSV)에서 ID를 기반으로 데이터 찾기
        if src_item and rid in src_item.id_to_rowidx:
            obj["id"] = rid # ID가 유효할 때만 추가
            row = src_item.df.iloc[src_item.id_to_rowidx[rid]]
            
            # NaN 값을 None이나 빈 문자열로 변환
            row_dict = row.where(pd.notna(row), None).to_dict()
            
            if req.json_cols:
                include_cols = [c for c in req.json_cols if c in src_item.df.columns]
                obj.update({c: row_dict[c] for c in include_cols})
            else:
                obj.update(row_dict)
        # 만약 상세 정보 소스에 ID가 없다면, 이 책은 결과에 포함시키지 않습니다.
        # 이것이 ID 불일치 시에도 프로그램이 안정적으로 동작하게 합니다.
        
        # 유효한 책 정보가 obj에 담겼을 때만 score, rank 추가 및 최종 결과에 포함
        if "id" in obj:
            if req.include_score:
                obj["_score"] = float(scores[i])
            if req.include_rank:
                obj["_rank"] = rank

            obj = truncate_strings_in_obj(obj, req.max_text_chars)
            results.append(obj)

    return SearchResponse(
        query=req.query,
        top_k=req.top_k,
        unique_returned=len(results),
        dedup_by_title=dedup_by_title,
        randomize=req.randomize,
        tau=req.tau,
        seed=req.seed,
        index_path=req.npz,
        meta_path=req.meta,
        source_csv=req.source_csv,
        results=results,
    )

# --- 추가된 부분 ---
# 루트 경로 ("/") 요청 시 index.html 파일을 반환
# 이 코드는 다른 모든 경로 설정보다 *앞에* 위치해도 괜찮지만,
# 명확성을 위해 API 엔드포인트들 다음에 두는 것이 일반적입니다.
@app.get("/", include_in_schema=False)
async def read_index():
    return FileResponse('static/index.html')

# "static" 폴더를 정적 파일 디렉토리로 마운트
# 이 코드는 반드시 다른 모든 @app.get, @app.post 보다 뒤에 와야 합니다.
# 그래야 /search 같은 API 경로가 정적 파일로 처리되는 것을 막을 수 있습니다.
app.mount("/", StaticFiles(directory="static"), name="static")
# --------------------