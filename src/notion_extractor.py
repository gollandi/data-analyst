"""
notion_extractor.py
─────────────────────────────────────────────────────────────────────
Pulls data from Notion databases → pandas DataFrames.
Handles all Notion property types, pagination, and auto-saves a
timestamped Parquet snapshot for reproducibility.

Usage:
    from src.notion_extractor import NotionExtractor
    extractor = NotionExtractor()
    df = extractor.get_database("compass_circumcision")
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from dotenv import load_dotenv
from notion_client import Client

load_dotenv()
logger = logging.getLogger(__name__)


# ─── CONFIG ──────────────────────────────────────────────────────────────────

def _load_config() -> dict:
    cfg_path = Path(__file__).parents[1] / "config" / "settings.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)

CFG = _load_config()


# ─── PROPERTY PARSERS ────────────────────────────────────────────────────────

def _parse_property(prop: dict) -> Any:
    """
    Normalise every Notion property type to a Python scalar.
    Returns None where a value is genuinely missing.
    """
    t = prop.get("type")

    if t == "title":
        items = prop.get("title", [])
        return "".join(i["plain_text"] for i in items) if items else None

    elif t == "rich_text":
        items = prop.get("rich_text", [])
        return "".join(i["plain_text"] for i in items) if items else None

    elif t == "number":
        return prop.get("number")

    elif t == "select":
        sel = prop.get("select")
        return sel["name"] if sel else None

    elif t == "multi_select":
        return [s["name"] for s in prop.get("multi_select", [])] or None

    elif t == "date":
        d = prop.get("date")
        if not d:
            return None
        start = d.get("start")
        return pd.to_datetime(start) if start else None

    elif t == "checkbox":
        return bool(prop.get("checkbox"))

    elif t == "url":
        return prop.get("url")

    elif t == "email":
        return prop.get("email")

    elif t == "phone_number":
        return prop.get("phone_number")

    elif t == "formula":
        formula = prop.get("formula", {})
        sub_type = formula.get("type")
        return formula.get(sub_type)

    elif t == "rollup":
        rollup = prop.get("rollup", {})
        sub_type = rollup.get("type")
        if sub_type == "array":
            return [_parse_property(i) for i in rollup.get("array", [])]
        return rollup.get(sub_type)

    elif t == "people":
        people = prop.get("people", [])
        return [p.get("name") for p in people] if people else None

    elif t == "files":
        files = prop.get("files", [])
        return [f.get("name") for f in files] if files else None

    elif t == "relation":
        rels = prop.get("relation", [])
        return [r["id"] for r in rels] if rels else None

    elif t == "status":
        status = prop.get("status")
        return status["name"] if status else None

    else:
        logger.debug(f"Unknown property type: {t}")
        return None


# ─── EXTRACTOR ────────────────────────────────────────────────────────────────

class NotionExtractor:
    """
    Pull entire Notion databases into tidy pandas DataFrames.

    Parameters
    ----------
    api_key : str, optional
        Notion integration token. Defaults to NOTION_API_KEY env var.
    snapshot : bool, optional
        If True (default from config), saves a Parquet snapshot on each pull.
    """

    def __init__(
        self,
        api_key: str | None = None,
        snapshot: bool | None = None,
    ):
        key = api_key or os.getenv("NOTION_API_KEY")
        if not key:
            raise EnvironmentError(
                "NOTION_API_KEY not found. Set it in .env or pass api_key=."
            )
        self.client = Client(auth=key)
        self._cfg = CFG
        self._snapshot = (
            snapshot
            if snapshot is not None
            else CFG["project"].get("snapshot_on_extract", True)
        )
        self._data_dir = (
            Path(__file__).parents[1] / CFG["project"]["data_dir"] / "raw"
        )
        self._data_dir.mkdir(parents=True, exist_ok=True)

    # ── public API ─────────────────────────────────────────────────────────

    def get_database(
        self,
        db_alias: str,
        filters: dict | None = None,
        sorts: list | None = None,
        rename_map: dict[str, str] | None = None,
    ) -> pd.DataFrame:
        """
        Fetch all pages from a Notion database.

        Parameters
        ----------
        db_alias : str
            Key from config/settings.yaml → notion.databases
        filters : dict, optional
            Notion filter object (https://developers.notion.com/reference/post-database-query-filter)
        sorts : list, optional
            Notion sort array
        rename_map : dict, optional
            {'Notion Column Name': 'python_variable_name'}

        Returns
        -------
        pd.DataFrame
        """
        db_id = self._resolve_id(db_alias)
        pages = self._fetch_all_pages(db_id, filters=filters, sorts=sorts)
        df = self._pages_to_dataframe(pages)

        if rename_map:
            df = df.rename(columns=rename_map)

        if self._snapshot:
            self._save_snapshot(df, db_alias)

        logger.info(f"[{db_alias}] extracted {len(df)} rows, {len(df.columns)} cols")
        return df

    def get_database_by_id(
        self,
        database_id: str,
        label: str = "unnamed",
        **kwargs,
    ) -> pd.DataFrame:
        """Same as get_database but accepts a raw UUID instead of alias."""
        pages = self._fetch_all_pages(database_id, **kwargs)
        df = self._pages_to_dataframe(pages)
        if self._snapshot:
            self._save_snapshot(df, label)
        return df

    def list_database_schema(self, db_alias: str) -> pd.DataFrame:
        """
        Returns a DataFrame describing all properties in the database.
        Useful for mapping column names before analysis.
        """
        db_id = self._resolve_id(db_alias)
        db_meta = self.client.databases.retrieve(database_id=db_id)
        props = db_meta["properties"]
        rows = [{"column": k, "type": v["type"]} for k, v in props.items()]
        return pd.DataFrame(rows).sort_values("column")

    # ── internals ──────────────────────────────────────────────────────────

    def _resolve_id(self, db_alias: str) -> str:
        db_map = self._cfg["notion"]["databases"]
        db_id = db_map.get(db_alias)
        if not db_id:
            raise KeyError(
                f"Database alias '{db_alias}' not found in config/settings.yaml. "
                f"Available: {list(db_map.keys())}"
            )
        return db_id

    def _fetch_all_pages(
        self,
        database_id: str,
        filters: dict | None = None,
        sorts: list | None = None,
    ) -> list[dict]:
        """Handles Notion API pagination automatically."""
        pages = []
        cursor = None
        query_kwargs: dict[str, Any] = {"database_id": database_id, "page_size": 100}
        if filters:
            query_kwargs["filter"] = filters
        if sorts:
            query_kwargs["sorts"] = sorts

        while True:
            if cursor:
                query_kwargs["start_cursor"] = cursor
            response = self.client.databases.query(**query_kwargs)
            pages.extend(response["results"])
            if not response.get("has_more"):
                break
            cursor = response.get("next_cursor")

        return pages

    def _pages_to_dataframe(self, pages: list[dict]) -> pd.DataFrame:
        rows = []
        for page in pages:
            row: dict[str, Any] = {
                "_notion_id": page["id"],
                "_created_time": pd.to_datetime(page.get("created_time")),
                "_last_edited": pd.to_datetime(page.get("last_edited_time")),
            }
            for col_name, prop_value in page.get("properties", {}).items():
                row[col_name] = _parse_property(prop_value)
            rows.append(row)
        return pd.DataFrame(rows)

    def _save_snapshot(self, df: pd.DataFrame, label: str) -> None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self._data_dir / f"{label}_{ts}.parquet"
        df.to_parquet(path, index=False)
        logger.info(f"Snapshot saved → {path}")
