from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class CachePaths:
    base_dir: Path

    def dataset_dir(self, user_id: str, dataset_id: str) -> Path:
        return self.base_dir / "datasets" / user_id / dataset_id

    def raw_dir(self, user_id: str, dataset_id: str) -> Path:
        return self.dataset_dir(user_id, dataset_id) / "raw"

    def parquet_dir(self, user_id: str, dataset_id: str) -> Path:
        return self.dataset_dir(user_id, dataset_id) / "parquet"

    def raw_path(self, user_id: str, dataset_id: str, filename: str) -> Path:
        return self.raw_dir(user_id, dataset_id) / filename

    def parquet_path(self, user_id: str, dataset_id: str) -> Path:
        return self.parquet_dir(user_id, dataset_id) / "data.parquet"
