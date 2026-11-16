#!/usr/bin/env python3
"""
19_hybrid_retrieval.py  —  Hybrid graph-RAG retrieval system (Stage 4-A, Task R1).

Three retrieval modes behind one config flag (rag_architecture.md §3):

  focal_only    — Returns only the focal firm's own profile (lower bound baseline).
  hybrid_text   — BM25 (exact: tickers, product codes) + dense (BGE-base-en-v1.5)
                  fused by Reciprocal Rank Fusion (rank-based, score-scale agnostic).
                  No KG traversal.
  hybrid_graph  — hybrid_text, then k-hop KG expansion (1–2 hops, optional relation-type
                  filter) adds neighbour firm profiles to the candidate set before final
                  RRF re-rank. This is the system described in rag_architecture.md.

Provides `Retriever` class for import by 20_eval_hybrid.py. When run directly, builds and
persists the Chroma dense index (one-time ~30s for 149 firm profiles).

USAGE
  # Build index (first time or to rebuild)
  python 19_hybrid_retrieval.py --build

  # Run retrieval smoke-test on 5 events
  python 19_hybrid_retrieval.py --smoke-test

IMPORTS
  from 19_hybrid_retrieval import Retriever
  r = Retriever()
  results = r.retrieve(event_text, primary_ticker, event_date, mode="hybrid_graph", k=20)
  # returns: list of {"ticker": str, "rank": int, "source": str, "hop": int}
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
import warnings
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    print("[FATAL] pip install rank-bm25")
    sys.exit(2)

RETRIEVAL_DIR = Path("data/retrieval")
KG_PATH       = Path("kg/networkx_export.gpickle")
CHROMA_DIR    = RETRIEVAL_DIR / "chroma_db"
MODEL_NAME    = "BAAI/bge-base-en-v1.5"
RRF_K         = 60   # standard RRF constant


# ---------------------------------------------------------------------------
# Document loading
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


# ---------------------------------------------------------------------------
# RRF fusion
# ---------------------------------------------------------------------------

def rrf_fuse(ranked_lists: list[list[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    """
    Reciprocal Rank Fusion. Each list is a ranked list of ticker strings.
    Returns merged list of (ticker, rrf_score) sorted descending.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, ticker in enumerate(ranked, 1):
            scores[ticker] = scores.get(ticker, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])


# ---------------------------------------------------------------------------
# Time-aware KG neighbours
# ---------------------------------------------------------------------------

def kg_neighbours_asof(
    g: nx.MultiDiGraph,
    focal: str,
    event_date: pd.Timestamp,
    max_hops: int = 2,
    rel_filter: Optional[set[str]] = None,
    universe: Optional[set[str]] = None,
) -> dict[str, dict]:
    """
    BFS from focal over edges with valid_from <= event_date.
    Returns {ticker: {"hop": int, "rel_types": list[str]}} for universe firms only.
    rel_filter: if set, only traverse edges of those relation types (but still collect
                hop distance regardless — filter only affects which edges are traversed).
    """
    if focal not in g:
        return {}

    # Build time-filtered edge subgraph
    sub = nx.Graph()
    sub.add_nodes_from(g.nodes())
    for u, v, key, d in g.edges(keys=True, data=True):
        vf = str(d.get("valid_from", "") or "0001-01-01")[:10]
        try:
            ok = pd.to_datetime(vf) <= event_date
        except Exception:
            ok = False
        if ok:
            if rel_filter is None or key in rel_filter:
                sub.add_edge(u, v)

    lengths = nx.single_source_shortest_path_length(sub, focal, cutoff=max_hops)

    result = {}
    for node, hops in lengths.items():
        if node == focal or hops == 0:
            continue
        if universe is not None and node not in universe:
            continue
        # Collect direct relation types (hop-1 only for efficiency)
        rel_types: list[str] = []
        if hops == 1:
            for u, v, key, d in g.edges(keys=True, data=True):
                if {u, v} == {focal, node}:
                    vf = str(d.get("valid_from", "") or "0001-01-01")[:10]
                    try:
                        if pd.to_datetime(vf) <= event_date:
                            rel_types.append(key)
                    except Exception:
                        pass
        result[node] = {"hop": hops, "rel_types": sorted(set(rel_types))}
    return result


# ---------------------------------------------------------------------------
# Retriever class
# ---------------------------------------------------------------------------

class Retriever:
    """
    Encapsulates BM25 + dense indices + KG. Call .retrieve() for any mode.
    """

    def __init__(
        self,
        retrieval_dir: Path = RETRIEVAL_DIR,
        kg_path: Path = KG_PATH,
        chroma_dir: Path = CHROMA_DIR,
        model_name: str = MODEL_NAME,
    ) -> None:
        self.retrieval_dir = retrieval_dir
        self.chroma_dir    = chroma_dir
        self.model_name    = model_name

        # Load firm profiles
        self.firm_profiles = load_jsonl(retrieval_dir / "firm_profiles.jsonl")
        self.tickers       = [d["ticker"] for d in self.firm_profiles]
        self.ticker_set    = set(self.tickers)
        self.ticker_to_profile = {d["ticker"]: d for d in self.firm_profiles}

        # BM25
        corpus = [self._tokenize(d["text"]) for d in self.firm_profiles]
        self.bm25 = BM25Okapi(corpus)

        # Dense (Chroma)
        self._embed_model = None
        self._chroma_col  = None
        self._init_dense()

        # KG
        self.g = None
        if kg_path.exists():
            with open(kg_path, "rb") as f:
                self.g = pickle.load(f)
        else:
            print(f"[warn] KG not found at {kg_path} — hybrid_graph mode unavailable")

    # ---- tokenisation -------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\b[a-zA-Z0-9\.]+\b", text.lower())

    # ---- dense index --------------------------------------------------------

    def _init_dense(self) -> None:
        """Load or build the Chroma persistent collection."""
        try:
            import chromadb
            from sentence_transformers import SentenceTransformer
        except ImportError:
            print("[warn] sentence-transformers / chromadb not available — dense retrieval disabled")
            return

        client = chromadb.PersistentClient(path=str(self.chroma_dir))
        col_name = "firm_profiles_bge"

        existing = [c.name for c in client.list_collections()]
        if col_name in existing:
            self._chroma_col = client.get_collection(col_name)
            # Verify count matches
            if self._chroma_col.count() == len(self.firm_profiles):
                self._embed_model = SentenceTransformer(self.model_name)
                return
            else:
                client.delete_collection(col_name)

        print(f"[info] building Chroma index with {self.model_name} ...")
        self._embed_model = SentenceTransformer(self.model_name)
        texts = [d["text"] for d in self.firm_profiles]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            embeddings = self._embed_model.encode(
                texts, batch_size=32, show_progress_bar=True,
                normalize_embeddings=True,
            ).tolist()
        self._chroma_col = client.create_collection(
            name=col_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._chroma_col.add(
            documents=texts,
            embeddings=embeddings,
            ids=[d["doc_id"] for d in self.firm_profiles],
            metadatas=[{"ticker": d["ticker"]} for d in self.firm_profiles],
        )
        print(f"[ok] Chroma index built: {self._chroma_col.count()} docs")

    # ---- individual retrieval methods ---------------------------------------

    def _bm25_rank(self, query: str, exclude: set[str], n: int) -> list[str]:
        scores = self.bm25.get_scores(self._tokenize(query))
        ranked = sorted(zip(self.tickers, scores), key=lambda x: -x[1])
        return [t for t, _ in ranked if t not in exclude][:n]

    def _dense_rank(self, query: str, exclude: set[str], n: int) -> list[str]:
        if self._chroma_col is None or self._embed_model is None:
            return []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            qvec = self._embed_model.encode(
                [query], normalize_embeddings=True
            ).tolist()
        results = self._chroma_col.query(
            query_embeddings=qvec,
            n_results=min(n + len(exclude) + 5, self._chroma_col.count()),
        )
        return [m["ticker"] for m in results["metadatas"][0]
                if m["ticker"] not in exclude][:n]

    # ---- public retrieve ----------------------------------------------------

    def retrieve(
        self,
        event_text: str,
        primary_ticker: str,
        event_date,
        mode: str = "hybrid_graph",
        k: int = 20,
        max_hops: int = 2,
        rel_filter: Optional[set[str]] = None,
    ) -> list[dict]:
        """
        Returns a ranked list of dicts:
          {"ticker": str, "rank": int, "source": str ("bm25"|"dense"|"graph"|"focal"),
           "hop": int (-1 = no direct path / not from graph)}
        Length <= k.
        """
        exclude = {primary_ticker}
        event_date = pd.Timestamp(event_date)

        if mode == "focal_only":
            if primary_ticker in self.ticker_set:
                return [{"ticker": primary_ticker, "rank": 1, "source": "focal", "hop": 0}]
            return []

        # hybrid_text or hybrid_graph: BM25 + dense → RRF
        n_candidates = k * 3   # over-fetch before RRF trim
        bm25_ranked  = self._bm25_rank(event_text, exclude, n_candidates)
        dense_ranked = self._dense_rank(event_text, exclude, n_candidates)

        # If dense unavailable, fall back to BM25 only
        if dense_ranked:
            fused = rrf_fuse([bm25_ranked, dense_ranked])
        else:
            fused = [(t, 1.0 / (RRF_K + i + 1)) for i, (t, _) in enumerate(
                [(t, 0) for t in bm25_ranked])]

        # Build source map
        source_map: dict[str, str] = {}
        for t in bm25_ranked[:k]:
            source_map.setdefault(t, "bm25")
        for t in dense_ranked[:k]:
            source_map.setdefault(t, "dense")
        for t in bm25_ranked[:k]:
            if t in (source_map.get(tt) for tt in dense_ranked[:k]):
                source_map[t] = "bm25+dense"
        # Simpler: mark fused sources
        bm25_set  = set(bm25_ranked[:k])
        dense_set = set(dense_ranked[:k])
        source_map = {t: ("bm25+dense" if t in bm25_set and t in dense_set
                          else "dense" if t in dense_set else "bm25")
                      for t in set(bm25_ranked + dense_ranked)}

        if mode == "hybrid_graph" and self.g is not None:
            # KG expansion: build a KG-proximity ranked list and add it as a third RRF signal.
            # Rank by hop distance ascending (hop-1 best), then by BM25 score as tiebreak.
            kg_nbrs = kg_neighbours_asof(
                self.g, primary_ticker, event_date,
                max_hops=max_hops, rel_filter=rel_filter,
                universe=self.ticker_set,
            )
            hop_map: dict[str, int] = {t: d["hop"] for t, d in kg_nbrs.items()}

            # Sort KG neighbours: hop-1 first, hop-2 next, rest unchanged
            kg_ranked = sorted(kg_nbrs.keys(), key=lambda t: kg_nbrs[t]["hop"])

            # Re-fuse all three signals: BM25, dense, KG-proximity
            if dense_ranked:
                fused = rrf_fuse([bm25_ranked, dense_ranked, kg_ranked])
            else:
                fused = rrf_fuse([bm25_ranked, kg_ranked])

            # Update source map for KG-sourced tickers
            for t in kg_nbrs:
                if t not in source_map or source_map[t] == "unknown":
                    source_map[t] = f"graph_hop{kg_nbrs[t]['hop']}"
                else:
                    source_map[t] = source_map[t] + f"+graph_hop{kg_nbrs[t]['hop']}"
        else:
            hop_map = {}

        # Trim to k and assign ranks
        output = []
        for rank, (ticker, _) in enumerate(fused[:k], 1):
            if ticker in exclude:
                continue
            output.append({
                "ticker": ticker,
                "rank": rank,
                "source": source_map.get(ticker, "unknown"),
                "hop": hop_map.get(ticker, -1),
            })
            if len(output) >= k:
                break
        return output


# ---------------------------------------------------------------------------
# CLI: build index / smoke test
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build",      action="store_true", help="Force rebuild Chroma index")
    ap.add_argument("--smoke-test", action="store_true", help="Run retrieval on 5 events")
    args = ap.parse_args()

    if args.build:
        if CHROMA_DIR.exists():
            import shutil
            shutil.rmtree(CHROMA_DIR)
            print("[info] deleted old Chroma index")

    print("[info] loading Retriever ...")
    r = Retriever()

    if args.smoke_test or not args.build:
        events = load_jsonl(RETRIEVAL_DIR / "event_contexts.jsonl")[:5]
        for ev in events:
            results = r.retrieve(
                ev["text"], ev["primary_ticker"],
                ev.get("filing_date", "2024-01-01"),
                mode="hybrid_graph", k=10,
            )
            print(f"\n[{ev['primary_ticker']}] {ev['text'][:80].strip()!r}")
            for res in results[:5]:
                print(f"  rank={res['rank']}  {res['ticker']:<10}  "
                      f"hop={res['hop']}  source={res['source']}")
    else:
        print("[ok] index built — import Retriever from 19_hybrid_retrieval")
    return 0


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sys.exit(main())
