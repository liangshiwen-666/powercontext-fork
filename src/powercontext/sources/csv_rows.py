# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Capture row-level CSV records as stable, replayable Sources."""

from __future__ import annotations

import csv
import io
import json
from urllib.parse import quote

from pydantic import BaseModel

from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources import (
    TEXT_EVIDENCE_PROJECTION_KEY,
    AdapterSourceDefinition,
    Source,
    SourceMaterialization,
    TextEvidence,
)

CSV_ROW_SOURCE_NAME = "csv-row"
CSV_ROW_DEFINITION_VERSION = "1"

__all__ = [
    "CSV_ROW_DEFINITION_VERSION",
    "CSV_ROW_SOURCE_DEFINITION",
    "CSV_ROW_SOURCE_NAME",
    "CSVDocumentInput",
    "CSVRowInput",
    "CSVRowSource",
    "CSVRowTextEvidenceProjection",
    "csv_row_source_id",
    "parse_csv_document",
]


class CSVDocumentInput(BaseModel):
    """One complete offline CSV document to split into captured row Sources."""

    document_id: str
    primary_key: str
    content: bytes


class CSVRowInput(BaseModel):
    """One parsed CSV data row with exact, order-preserving field values."""

    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]


class CSVRowSource(Source):
    """A captured Source holding one complete original CSV record."""

    source_id: str
    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]


def _validate_document_id(document_id: object) -> str:
    """Require a non-empty document identity without surrounding whitespace."""

    if not isinstance(document_id, str) or not document_id.strip():
        raise ValueError("document_id must be a non-empty string")  # noqa: TRY003
    if document_id != document_id.strip():
        raise ValueError("document_id must not contain leading or trailing whitespace")  # noqa: TRY003
    return document_id


def _validate_primary_key_value(primary_key_value: object) -> str:
    """Require a non-blank primary key value; surrounding spaces stay significant."""

    if not isinstance(primary_key_value, str) or not primary_key_value.strip():
        raise ValueError("primary_key_value must be a non-empty string")  # noqa: TRY003
    return primary_key_value


def _validate_header(header: tuple[str, ...]) -> None:
    """Require a non-empty, non-blank, exactly unique header record."""

    if not header:
        raise ValueError("CSV header must contain at least one field")  # noqa: TRY003
    seen: set[str] = set()
    for column in header:
        if not column.strip():
            raise ValueError("CSV header fields must not be blank")  # noqa: TRY003
        if column in seen:
            raise ValueError(f"duplicate CSV header field: {column!r}")  # noqa: TRY003
        seen.add(column)


def csv_row_source_id(document_id: str, primary_key_value: str) -> str:
    """Return the stable, delimiter-safe identity of one CSV data row."""

    document_id = _validate_document_id(document_id)
    primary_key_value = _validate_primary_key_value(primary_key_value)
    identity = f"csv:{quote(document_id, safe='')}:{quote(primary_key_value, safe='')}"
    if len(identity) > MAX_SOURCE_ID_LENGTH:
        raise ValueError(f"CSV row identity must not exceed {MAX_SOURCE_ID_LENGTH} characters")  # noqa: TRY003
    return identity


def parse_csv_document(value: CSVDocumentInput) -> tuple[CSVRowInput, ...]:
    """Parse and validate one complete CSV document atomically into data rows."""

    document_id = _validate_document_id(value.document_id)
    primary_key = value.primary_key
    if not isinstance(primary_key, str) or not primary_key.strip():
        raise ValueError("primary_key must be a non-empty string")  # noqa: TRY003

    try:
        text = value.content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("CSV document must be valid UTF-8") from error
    try:
        records = list(csv.reader(io.StringIO(text, newline=""), strict=True))
    except csv.Error as error:
        raise ValueError(f"malformed CSV document: {error}") from error
    if not records:
        raise ValueError("CSV document must not be empty")  # noqa: TRY003

    header = tuple(records[0])
    _validate_header(header)
    try:
        primary_index = header.index(primary_key)
    except ValueError:
        raise ValueError(f"primary_key {primary_key!r} does not match any CSV header field") from None

    width = len(header)
    rows: list[CSVRowInput] = []
    seen_keys: set[str] = set()
    for record in records[1:]:
        if not record:
            # A completely empty physical record carries no data row.
            continue
        if len(record) != width:
            raise ValueError(f"CSV data row width {len(record)} does not match header width {width}")  # noqa: TRY003
        primary_key_value = record[primary_index]
        if not primary_key_value.strip():
            raise ValueError("CSV primary key value must not be blank")  # noqa: TRY003
        if primary_key_value in seen_keys:
            raise ValueError(f"duplicate CSV primary key value: {primary_key_value!r}")  # noqa: TRY003
        seen_keys.add(primary_key_value)
        rows.append(
            CSVRowInput(
                document_id=document_id,
                primary_key=primary_key,
                primary_key_value=primary_key_value,
                columns=header,
                values=tuple(record),
            )
        )
    return tuple(rows)


class CSVRowAdapter:
    """Resolve and read one captured CSV data row."""

    input_class = CSVRowInput
    name = CSV_ROW_SOURCE_NAME
    source_class = CSVRowSource

    async def resolve(self, value: CSVRowInput, /) -> CSVRowSource:
        identity = csv_row_source_id(value.document_id, value.primary_key_value)
        return CSVRowSource(
            name=identity,
            source_id=identity,
            materialization=SourceMaterialization.CAPTURED,
            document_id=value.document_id,
            primary_key=value.primary_key,
            primary_key_value=value.primary_key_value,
            columns=value.columns,
            values=value.values,
        )

    async def read(self, source: CSVRowSource, /) -> CSVRowInput:
        return CSVRowInput(
            document_id=source.document_id,
            primary_key=source.primary_key,
            primary_key_value=source.primary_key_value,
            columns=source.columns,
            values=source.values,
        )


CSV_ROW_ADAPTER = CSVRowAdapter()


class CSVRowTextEvidenceProjection:
    """Expose one CSV row as the ordered row object consumed as text evidence."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = CSVRowSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: CSVRowSource, /) -> TextEvidence:
        row = dict(zip(source.columns, source.values, strict=True))
        return TextEvidence(
            source_type=CSV_ROW_SOURCE_NAME,
            source_id=source.source_id,
            content=json.dumps(row, ensure_ascii=False, separators=(",", ":")),
            metadata={
                "document_id": source.document_id,
                "primary_key": source.primary_key,
                "primary_key_value": source.primary_key_value,
                "columns": list(source.columns),
                "row": row,
            },
        )


CSV_ROW_SOURCE_DEFINITION: AdapterSourceDefinition[CSVRowInput, CSVRowSource, CSVRowInput] = AdapterSourceDefinition(
    CSV_ROW_ADAPTER,
    version=CSV_ROW_DEFINITION_VERSION,
    projections=(CSVRowTextEvidenceProjection(),),
)
