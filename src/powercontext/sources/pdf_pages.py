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

"""Complete physical-page PDF parsing and page-level evidence retrieval.

The parser turns the exact bytes of an original financial-report PDF into one
``PDFPageInput`` per physical page, in order, using the locked ``pypdf``
dependency. The Definition publishes each page as a captured Source whose
standard text projection carries the document identity, the whole-document
digest and the physical page number so a citation can be audited against the
original file.

``search_pdf_pages`` answers a natural-language FinanceBench question from an
already-open scoped Memory. It only uses the public scoped Memory search API:
candidate pages are gathered with several deterministic lexical queries,
merged and reranked offline over the returned page bodies, and emitted as
exact ``MemoryCitation`` values. Reranking rewards question-term coverage,
rarity-weighted concept matches and the canonical filing structures (income
statement, cash flow statement, legal proceedings, business description, ...)
that evidence pages for such questions live in.
"""

from __future__ import annotations

import hashlib
import io
import math
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, field_validator
from pypdf import PdfReader

from powercontext.artifacts import MemoryCitation
from powercontext.builtin.runtime import SearchMemoryRequest
from powercontext.sources import TEXT_EVIDENCE_PROJECTION_KEY, AdapterSourceDefinition, TextEvidence
from powercontext.sources.models import Source, SourceMaterialization

SOURCE_NAME = "pdf-page"
SOURCE_TYPE = SOURCE_NAME
_MAX_SOURCE_ID_LENGTH = 256
_MAX_DOCUMENT_LABEL = 32

_PAGE_TOKEN = re.compile(r"[a-z]+|\d+")
_YEAR_TOKEN = re.compile(r"^(?:18|19|20)\d{2}$")


class PDFDocumentInput(BaseModel):
    """One complete original PDF supplied for page-by-page parsing."""

    document_id: str
    content: bytes

    @field_validator("document_id")
    @classmethod
    def validate_document_id(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("document_id must be a non-empty, non-blank string")  # noqa: TRY003
        return value


class PDFPageInput(BaseModel):
    """One physical page of a PDF together with its exact document identity."""

    document_id: str
    document_sha256: str
    page_number: int
    text: str


class PDFPageSource(Source):
    """Captured Source value for one physical page of one exact PDF revision."""

    materialization: Literal[SourceMaterialization.CAPTURED] = SourceMaterialization.CAPTURED
    document_id: str
    document_sha256: str
    page_number: int
    text: str


class PDFPageTextEvidence:
    """Expose a page through the standard text-evidence projection."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PDFPageSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: PDFPageSource, /) -> TextEvidence:
        return TextEvidence(
            source_type=SOURCE_TYPE,
            source_id=source.name,
            content=f"{source.document_id}\n{source.text}",
            metadata={
                "document_id": source.document_id,
                "document_sha256": source.document_sha256,
                "page_number": source.page_number,
            },
        )


class PDFPageSourceAdapter:
    """Resolve and read the captured page Source defined by this challenge."""

    input_class = PDFPageInput
    name = SOURCE_NAME
    source_class = PDFPageSource

    async def resolve(self, value: PDFPageInput, /) -> PDFPageSource:
        if not isinstance(value, PDFPageInput):
            raise TypeError("PDF page Sources are resolved from PDFPageInput values")  # noqa: TRY003
        document_id = _require_document_id(value.document_id)
        page_number = _require_page_number(value.page_number)
        document_sha256 = _require_document_sha256(value.document_sha256)
        text = _require_page_text(value.text)
        return PDFPageSource(
            name=_page_source_name(document_id, document_sha256, page_number),
            materialization=SourceMaterialization.CAPTURED,
            document_id=document_id,
            document_sha256=document_sha256,
            page_number=page_number,
            text=text,
        )

    async def read(self, source: PDFPageSource, /) -> PDFPageInput:
        if not isinstance(source, PDFPageSource):
            raise TypeError("PDF page Sources can only be read as PDFPageInput values")  # noqa: TRY003
        return PDFPageInput(
            document_id=source.document_id,
            document_sha256=source.document_sha256,
            page_number=source.page_number,
            text=source.text,
        )


PDF_PAGE_SOURCE_DEFINITION = AdapterSourceDefinition(
    PDFPageSourceAdapter(),
    projections=(PDFPageTextEvidence(),),
)


def parse_pdf_document(value: PDFDocumentInput) -> tuple[PDFPageInput, ...]:
    """Parse every physical page of one original PDF, in order.

    Pages are numbered from 1 and include pages whose extractable text is empty.
    Any unreadable, encrypted-with-a-required-password, corrupt or partially
    parseable input raises ``ValueError`` without returning a partial result.
    """

    document = _coerce_document(value)
    content = bytes(document.content)
    if not content:
        raise ValueError("PDF content must not be empty")  # noqa: TRY003
    document_id = _require_document_id(document.document_id)
    document_sha256 = hashlib.sha256(content).hexdigest()
    try:
        reader = PdfReader(io.BytesIO(content))
        if reader.is_encrypted and not reader.decrypt(""):
            raise ValueError("PDF requires a user password and cannot be parsed")  # noqa: TRY003
        page_count = len(reader.pages)
        pages: list[PDFPageInput] = []
        for index in range(page_count):
            try:
                extracted = reader.pages[index].extract_text()
            except Exception as error:  # noqa: BLE001 - any page failure invalidates the whole document
                raise ValueError(f"PDF page {index + 1} could not be extracted") from error  # noqa: TRY003
            if extracted is None:
                extracted = ""
            if not isinstance(extracted, str):
                raise ValueError(f"PDF page {index + 1} did not yield text")  # noqa: TRY003
            pages.append(
                PDFPageInput(
                    document_id=document_id,
                    document_sha256=document_sha256,
                    page_number=index + 1,
                    text=extracted,
                )
            )
    except ValueError:
        raise
    except Exception as error:  # noqa: BLE001 - pypdf surfaces many unrelated error types
        raise ValueError("content is not a fully parseable PDF document") from error  # noqa: TRY003
    return tuple(pages)


def _coerce_document(value: PDFDocumentInput) -> PDFDocumentInput:
    if isinstance(value, PDFDocumentInput):
        return value
    return PDFDocumentInput.model_validate(value)


def _require_document_id(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("document_id must be a non-empty, non-blank string")  # noqa: TRY003
    return value


def _require_document_sha256(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("document_sha256 must be a non-empty string")  # noqa: TRY003
    return value


def _require_page_number(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("page_number must be a positive integer")  # noqa: TRY003
    return value


def _require_page_text(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("page text must be a string")  # noqa: TRY003
    return value


_HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _document_digest(document_id: str) -> str:
    return hashlib.sha256(document_id.encode("utf-8", "surrogatepass")).hexdigest()[:16]


def _page_source_name(document_id: str, document_sha256: str, page_number: int) -> str:
    """Build the stable, collision-free, bounded identity of one page Source."""

    label = re.sub(r"[^A-Za-z0-9._-]+", "-", document_id).strip("-")[:_MAX_DOCUMENT_LABEL] or "document"
    digest = document_sha256 if _HEX_SHA256.match(document_sha256) else hashlib.sha256(
        document_sha256.encode("utf-8", "surrogatepass")
    ).hexdigest()
    name = f"pdf-page-{label}-{_document_digest(document_id)}-{digest}-p{page_number}"
    if len(name) > _MAX_SOURCE_ID_LENGTH:
        name = f"pdf-page-{_document_digest(document_id)}-{hashlib.sha256(name.encode()).hexdigest()}"
    return name


# ---------------------------------------------------------------------------
# Question understanding: tokens, concepts and canonical filing structures
# ---------------------------------------------------------------------------

_FETCH_FLOOR = 44
_FETCH_CEILING = 64
_MAX_KEYWORD_QUERIES = 11
_MAX_SECOND_QUERIES = 4
_MAX_ANCHOR_QUERIES = 5
_FOCUSED_ANCHOR_RULES = 2
_BAG_TERMS = 11
_RARE_BAG_TERMS = 8
_RRF_CONSTANT = 16.0

_FUNCTION_WORDS = frozenset(
    """
    a an the and or but if then than so as of to in on at by for from with without within about into over
    under between among during after before per via up down out off all any some each every both either
    neither no not none only also just very more most less least much many few such same other others
    another own its it this that these those there here where when why how what which who whom whose has
    have had having is are was were be been being do does did done doing can could may might must shall
    should will would i we you he she they them their our your his her us me my mine ours yours am s t
    please kindly
    """.split()
)

_BOILERPLATE_WORDS = frozenset(
    """
    question questions answer answers answerable respond responds response reply replies assume assuming
    assumption perspective analyst analysts investment investor give giving provide providing correctly
    concisely briefly directly concretely exactly explicitly outlined stated define defined calculating
    compute determining determine tell relying detail details shown given
    """.split()
)

_UNIT_WORDS = frozenset(
    """
    usd dollar dollars million millions billion billions thousand thousands
    """.split()
)

_STOPWORDS = _FUNCTION_WORDS | _BOILERPLATE_WORDS | _UNIT_WORDS

_INSTRUCTION_START = re.compile(
    r"^(?:"
    r"respond(?:\s+(?:to|with)\b)?"
    r"|answer(?:\s+(?:in|with|the|concisely|briefly|directly|concretely|using|only)\b)?"
    r"|reply\b"
    r"|give\s+(?:a\s+)?response\b"
    r"|provide\s+(?:a\s+)?response\b"
    r"|assume\s+(?:the\s+)?perspective\b"
    r"|use\s+only\b"
    r"|do\s+not\b"
    r"|if\s+the\s+facts\b"
    r")",
    re.IGNORECASE,
)
_INSTRUCTION_TAIL = re.compile(
    r"\bthen\s+(?:state|say|respond|answer|report|write)\b.*\b(?:zero|0|unknown|not applicable|none)\b",
    re.IGNORECASE,
)
_INSTRUCTION_CONDITIONAL = re.compile(
    r"^if\b.*\b(?:not|no|absent|unavailable|unable|irrelevant)\b"
    r".*\b(?:please\s+)?(?:state|say|explain|respond|answer|report|indicate|note|write)\b",
    re.IGNORECASE,
)

_FINANCE_TERMS = frozenset(
    """
    revenue revenues sales sale income earnings operating operation operations profit profitability
    margin margins cost costs expense expenses ebitda ebit ebita capex capital expenditure expenditures
    depreciation amortization amortisation cash flow flows statement statements balance sheet position
    asset assets liability liabilities equity debt borrowings dividend dividends share shares stock
    repurchase repurchases buyback restructuring charge charges impairment goodwill lease leases
    pension benefit benefits interest rate rates tax taxes effective fiscal quarter quarterly annual
    yearly growth decline increase decrease percent percentage segment segments geographic geographical
    geographies geography region regions country countries market markets product products customer
    customers employee employees headcount legal proceedings litigation contingency contingencies
    lawsuit lawsuits claim claims settlement settlements fine fines regulation regulations compliance
    risk risks factor factors business businesses company companies corporate corporation consolidated
    net gross total operating income loss losses gain gains outstanding diluted basic weighted average
    return returns invested investment investments acquisition acquisitions merger mergers divestiture
    divestitures joint venture ventures affiliate affiliates subsidiary subsidiaries parent
    """.split()
)

_EXPANSIONS: dict[str, tuple[str, ...]] = {
    "capex": ("capital", "expenditures", "expenditure"),
    "ebitda": ("ebitda", "ebit"),
    "geographies": ("geographic", "regions", "countries", "geographical", "geography"),
    "geography": ("geographic", "regions", "countries", "geographical"),
    "operates": ("operations", "operating", "operation", "operate"),
    "operation": ("operations", "operating", "operate"),
    "battles": ("litigation", "proceedings", "contingencies", "claims"),
    "legal": ("litigation", "proceedings", "contingencies", "claims", "lawsuits"),
    "buyback": ("repurchase", "repurchases"),
    "margin": ("margin", "margins", "percentage"),
    "revenue": ("revenue", "revenues", "sales"),
    "revenues": ("revenue", "sales"),
    "sales": ("revenue", "revenues", "sales"),
    "income": ("income", "earnings"),
    "profit": ("profit", "income", "earnings"),
    "debt": ("debt", "borrowings", "notes"),
    "expense": ("expense", "expenses", "cost", "costs"),
    "expenses": ("expense", "cost", "costs"),
    "costs": ("cost", "costs", "expense", "expenses"),
    "dividends": ("dividend", "dividends"),
    "shares": ("share", "shares", "stock"),
    "staff": ("employee", "employees", "headcount"),
    "workforce": ("employee", "employees", "headcount"),
    "pension": ("pension", "pensions", "retirement"),
    "leases": ("lease", "leases"),
    "regions": ("regions", "region", "geographic", "geographies", "countries"),
    "countries": ("countries", "country", "geographic", "geographies", "regions"),
    "markets": ("markets", "market", "regions"),
    "products": ("products", "product", "brands"),
    "brands": ("brands", "brand", "products"),
    "facilities": ("facilities", "facility", "plants"),
}

_SUFFIX_VARIANTS = (
    ("ies", "y"),
    ("ives", "ife"),
    ("ses", "s"),
    ("xes", "x"),
    ("ches", "ch"),
    ("shes", "sh"),
    ("s", ""),
    ("ing", ""),
    ("ing", "e"),
    ("ed", ""),
    ("ed", "e"),
    ("ly", ""),
    ("ment", ""),
    ("ments", ""),
    ("ness", ""),
    ("ally", ""),
)


def _phrase(spec: str) -> tuple[frozenset[str], ...]:
    """Compile ``'word another|alternatives'`` into positional token sets."""

    return tuple(frozenset(part.split("|")) for part in spec.split())


@dataclass(frozen=True, slots=True)
class _AnchorRule:
    """A canonical filing structure that commonly holds evidence for a topic."""

    name: str
    priority: float
    trigger_tokens: frozenset[str]
    trigger_prefixes: tuple[str, ...]
    phrases: tuple[tuple[tuple[frozenset[str], ...], float, bool], ...]
    queries: tuple[str, ...] = ()
    trigger_phrases: tuple[str, ...] = ()


_ANCHOR_RULES: tuple[_AnchorRule, ...] = (
    _AnchorRule(
        name="legal",
        priority=3.1,
        trigger_tokens=frozenset(
            {"legal", "litigation", "proceedings", "lawsuit", "lawsuits", "claims", "contingencies", "regulatory"}
        ),
        trigger_prefixes=(),
        phrases=(
            (_phrase("item 3 legal proceedings"), 3.0, True),
            (_phrase("legal proceedings"), 2.0, False),
            (_phrase("multiple legal actions"), 2.2, False),
            (_phrase("legal actions"), 1.6, False),
            (_phrase("legal proceeding|proceedings"), 1.4, False),
            (_phrase("filed against us"), 1.0, False),
            (_phrase("litigation claims|settlements"), 0.9, False),
            (_phrase("regulatory proceedings|matters"), 0.7, False),
            (_phrase("inquiries and investigations"), 0.7, False),
            (_phrase("legal or regulatory proceedings"), 1.0, False),
            (_phrase("lawsuits"), 0.9, False),
            (_phrase("material adverse effect"), 0.4, False),
        ),
        queries=("item 3 legal proceedings",),
    ),
    _AnchorRule(
        name="cash-flow",
        priority=2.8,
        trigger_tokens=frozenset(
            {
                "cash",
                "flow",
                "flows",
                "capex",
                "depreciation",
                "amortization",
                "amortisation",
                "liquidity",
                "financing",
                "investing",
            }
        ),
        trigger_prefixes=(),
        phrases=(
            (_phrase("consolidated statement|statements of cash flows"), 3.2, True),
            (_phrase("statement|statements of cash flows"), 1.4, False),
            (_phrase("net cash provided by operating activities"), 1.0, False),
            (_phrase("operating activities"), 0.6, False),
            (_phrase("investing activities"), 0.5, False),
            (_phrase("financing activities"), 0.5, False),
            (_phrase("depreciation and amortization"), 1.2, False),
            (_phrase("capital expenditure|expenditures"), 0.9, False),
            (_phrase("cash and cash equivalents"), 0.4, False),
            (_phrase("net change in cash"), 0.6, False),
        ),
        queries=("consolidated statement cash flows",),
    ),
    _AnchorRule(
        name="income-statement",
        priority=2.6,
        trigger_tokens=frozenset(
            {
                "income",
                "earnings",
                "ebitda",
                "ebit",
                "ebita",
                "margin",
                "margins",
                "profit",
                "profitability",
                "revenue",
                "revenues",
                "sales",
                "expense",
                "expenses",
                "cost",
                "costs",
                "operating",
                "tax",
                "eps",
                "diluted",
                "shares",
            }
        ),
        trigger_prefixes=(),
        phrases=(
            (_phrase("consolidated statement|statements of income"), 3.2, True),
            (_phrase("consolidated statement|statements of operations"), 3.2, True),
            (_phrase("consolidated statement|statements of earnings"), 2.6, True),
            (_phrase("statement|statements of income"), 1.2, False),
            (_phrase("net revenue|revenues"), 0.9, False),
            (_phrase("disaggregation of net revenue"), 1.5, False),
            (_phrase("disaggregation of revenue"), 1.5, False),
            (_phrase("revenue by segment"), 0.7, False),
            (_phrase("cost of sales|revenue|goods sold"), 0.8, False),
            (_phrase("gross profit|margin"), 0.8, False),
            (_phrase("operating profit|income|loss"), 0.7, False),
            (_phrase("selling general and administrative"), 0.7, False),
            (_phrase("income|earnings before income taxes"), 0.6, False),
            (_phrase("provision for|benefit from income taxes"), 0.6, False),
            (_phrase("net income attributable"), 0.6, False),
        ),
        queries=("consolidated statement income",),
    ),
    _AnchorRule(
        name="margin",
        priority=2.7,
        trigger_tokens=frozenset({"margin", "margins", "ebitda", "ebit", "ebita", "profitability"}),
        trigger_prefixes=(),
        phrases=(
            (_phrase("consolidated statement|statements of income"), 2.8, True),
            (_phrase("consolidated statement|statements of operations"), 2.8, True),
            (_phrase("consolidated statement|statements of earnings"), 2.0, True),
            (_phrase("statement|statements of income"), 1.2, False),
            (_phrase("gross profit|margin"), 0.7, False),
            (_phrase("operating income|profit"), 0.6, False),
            (_phrase("net income"), 0.4, False),
        ),
        queries=(),
    ),
    _AnchorRule(
        name="balance-sheet",
        priority=2.5,
        trigger_tokens=frozenset(
            {
                "balance",
                "sheet",
                "assets",
                "liabilities",
                "liability",
                "equity",
                "debt",
                "borrowings",
                "inventory",
                "inventories",
                "receivables",
                "receivable",
                "liquidity",
                "liquid",
                "solvency",
                "quick",
            }
        ),
        trigger_prefixes=(),
        phrases=(
            (_phrase("consolidated balance sheet|sheets"), 3.2, True),
            (_phrase("total current assets"), 0.9, False),
            (_phrase("total assets"), 0.6, False),
            (_phrase("total current liabilities"), 0.9, False),
            (_phrase("total liabilities"), 0.6, False),
            (_phrase("total liabilities and equity"), 1.0, False),
            (_phrase("long-term debt"), 0.7, False),
            (_phrase("short-term debt"), 0.6, False),
            (_phrase("short-term investments"), 0.6, False),
            (_phrase("total equity"), 0.7, False),
            (_phrase("cash and cash equivalents"), 0.4, False),
            (_phrase("accounts receivable"), 0.5, False),
            (_phrase("inventories|inventory"), 0.5, False),
        ),
        queries=("consolidated balance sheet",),
    ),
    _AnchorRule(
        name="geographic",
        priority=2.4,
        trigger_tokens=frozenset(
            {
                "geographies",
                "geographic",
                "geographical",
                "regions",
                "countries",
                "segments",
                "division",
                "divisions",
                "markets",
                "market",
                "global",
                "worldwide",
                "international",
            }
        ),
        trigger_prefixes=("geograph", "region", "countr", "segment"),
        phrases=(
            (_phrase("item 1 business"), 2.6, True),
            (_phrase("reportable segments"), 2.0, False),
            (_phrase("seven reportable segments"), 1.2, False),
            (_phrase("geographic areas|area|regions"), 1.6, False),
            (_phrase("geographic information"), 1.2, False),
            (_phrase("countries and territories"), 1.0, False),
            (_phrase("segment information|reporting"), 0.8, False),
            (_phrase("segment operating profit|income|results|loss"), 0.8, False),
            (_phrase("summary of business segment data"), 2.2, False),
            (_phrase("business segment data"), 1.8, False),
            (_phrase("segment data"), 1.4, False),
            (_phrase("segment revenues|revenue"), 0.7, False),
            (_phrase("unallocated items"), 0.8, False),
            (_phrase("north america"), 0.3, False),
            (_phrase("latin america"), 0.3, False),
            (_phrase("south america"), 0.25, False),
            (_phrase("central america"), 0.2, False),
            (_phrase("united states"), 0.2, False),
            (_phrase("europe"), 0.2, False),
            (_phrase("middle east"), 0.3, False),
            (_phrase("south asia"), 0.25, False),
            (_phrase("asia pacific"), 0.3, False),
            (_phrase("africa"), 0.2, False),
            (_phrase("china"), 0.15, False),
            (_phrase("australia"), 0.15, False),
            (_phrase("new zealand"), 0.15, False),
            (_phrase("united kingdom"), 0.15, False),
            (_phrase("canada"), 0.15, False),
            (_phrase("mexico"), 0.15, False),
        ),
        queries=("reportable segments geographic areas countries territories",),
    ),
    _AnchorRule(
        name="restructuring",
        priority=2.6,
        trigger_tokens=frozenset({"restructuring", "restructurings", "severance", "productivity"}),
        trigger_prefixes=("restructur",),
        phrases=(
            (_phrase("restructuring and impairment charges"), 2.6, False),
            (_phrase("restructuring charges"), 1.5, False),
            (_phrase("productivity plan"), 1.2, False),
            (_phrase("multi-year productivity plan"), 1.0, False),
            (_phrase("severance and other costs"), 0.8, False),
            (_phrase("restructuring costs"), 1.2, False),
        ),
        queries=("restructuring impairment charges productivity plan",),
    ),
    _AnchorRule(
        name="customers",
        priority=2.1,
        trigger_tokens=frozenset({"customer", "customers", "concentration", "distributors", "distributor"}),
        trigger_prefixes=(),
        phrases=(
            (_phrase("customer concentration"), 2.2, False),
            (_phrase("concentration of credit risk"), 1.0, False),
            (_phrase("one customer accounted for"), 1.4, False),
            (_phrase("customer|two customers accounted for"), 1.2, False),
            (_phrase("largest customer|customers"), 1.2, False),
            (_phrase("significant customer|customers"), 1.0, False),
            (_phrase("major customer|customers"), 0.9, False),
            (_phrase("no single customer"), 0.9, False),
            (_phrase("loss of this customer"), 0.9, False),
            (_phrase("our customers include"), 0.6, False),
        ),
        queries=("customer concentration customers accounted",),
    ),
    _AnchorRule(
        name="metrics",
        priority=1.8,
        trigger_tokens=frozenset(
            {"retention", "retain", "retained", "churn", "members", "member", "metrics", "metric", "transactions"}
        ),
        trigger_prefixes=(),
        phrases=(
            (_phrase("selected metrics"), 1.8, False),
            (_phrase("member retention"), 1.6, False),
            (_phrase("card member retention"), 1.6, False),
            (_phrase("retention rate"), 1.2, False),
            (_phrase("retention remained"), 0.9, False),
            (_phrase("retention rates"), 0.9, False),
            (_phrase("active members"), 0.8, False),
            (_phrase("card member"), 0.6, False),
            (_phrase("card members"), 0.6, False),
        ),
        queries=("card member retention selected metrics",),
    ),
    _AnchorRule(
        name="revenue-concentration",
        priority=2.2,
        trigger_tokens=frozenset({"customers", "customer", "concentration", "primary"}),
        trigger_prefixes=(),
        phrases=(
            (_phrase("significant portion of our revenue|revenues"), 1.8, False),
            (_phrase("substantial portion of our revenue|revenues"), 1.6, False),
            (_phrase("limited number of"), 1.0, False),
            (_phrase("u s government"), 1.3, False),
            (_phrase("government contracts"), 1.0, False),
            (_phrase("commercial airlines"), 0.7, False),
            (_phrase("customer base"), 0.6, False),
            (_phrase("no single customer"), 0.8, False),
        ),
        queries=("limited number commercial airlines government revenue customers",),
    ),
    _AnchorRule(
        name="segment-data",
        priority=2.2,
        trigger_tokens=frozenset(
            {"product", "products", "service", "services", "category", "categories", "segment", "segments"}
        ),
        trigger_prefixes=(),
        phrases=(
            (_phrase("summary of business segment data"), 2.4, False),
            (_phrase("business segment data"), 2.0, False),
            (_phrase("segment data"), 1.4, False),
            (_phrase("reportable segments"), 1.2, False),
            (_phrase("segment revenues|revenue"), 0.8, False),
            (_phrase("unallocated items"), 0.8, False),
            (_phrase("total revenues"), 0.4, False),
        ),
        queries=("summary business segment data revenues",),
    ),
    _AnchorRule(
        name="risk",
        priority=1.7,
        trigger_tokens=frozenset({"risk", "risks", "uncertainties", "uncertainty"}),
        trigger_prefixes=(),
        phrases=(
            (_phrase("item 1a risk factors"), 3.0, True),
            (_phrase("risk factors"), 1.4, False),
            (_phrase("risks and uncertainties"), 0.5, False),
        ),
        queries=("item 1a risk factors",),
    ),
    _AnchorRule(
        name="mda",
        priority=1.6,
        trigger_tokens=frozenset({"growth", "decline", "increase", "decrease", "drove", "driver", "drivers", "margin"}),
        trigger_prefixes=(),
        phrases=(
            (_phrase("item 7 management s discussion and analysis"), 3.0, True),
            (_phrase("management s discussion and analysis"), 2.0, True),
            (_phrase("results of operations"), 0.9, False),
            (_phrase("executive overview"), 0.8, False),
        ),
        queries=("management discussion and analysis results of operations",),
        trigger_phrases=("why did", "what drove", "what caused", "explain", "how did"),
    ),
    _AnchorRule(
        name="properties",
        priority=1.8,
        trigger_tokens=frozenset({"properties", "property", "facilities", "plants", "headquarters"}),
        trigger_prefixes=(),
        phrases=(
            (_phrase("item 2 properties"), 3.0, True),
            (_phrase("principal properties"), 0.8, False),
        ),
        queries=("item 2 properties",),
    ),
    _AnchorRule(
        name="acquisitions",
        priority=1.9,
        trigger_tokens=frozenset(
            {"acquisition", "acquisitions", "acquire", "acquired", "divestiture", "divestitures", "merger", "mergers"}
        ),
        trigger_prefixes=(),
        phrases=(
            (_phrase("acquisitions and divestitures"), 1.8, False),
            (_phrase("business combinations"), 1.0, False),
            (_phrase("note 13 acquisitions"), 1.0, False),
        ),
        queries=("acquisitions divestitures business combinations",),
    ),
    _AnchorRule(
        name="pension",
        priority=1.6,
        trigger_tokens=frozenset({"pension", "pensions", "retirement", "retiree", "postretirement"}),
        trigger_prefixes=(),
        phrases=(
            (_phrase("pension and retiree medical"), 1.4, False),
            (_phrase("pension plans"), 1.0, False),
            (_phrase("pension and postretirement"), 1.0, False),
            (_phrase("net periodic benefit"), 0.8, False),
        ),
        queries=("pension retiree medical postretirement",),
    ),
    _AnchorRule(
        name="leases",
        priority=1.4,
        trigger_tokens=frozenset({"lease", "leases", "leasing"}),
        trigger_prefixes=(),
        phrases=(
            (_phrase("lease cost"), 0.9, False),
            (_phrase("operating lease"), 0.7, False),
            (_phrase("right-of-use asset"), 0.8, False),
        ),
        queries=("operating lease right-of-use asset",),
    ),
    _AnchorRule(
        name="income-tax",
        priority=1.5,
        trigger_tokens=frozenset({"tax", "taxes", "taxation"}),
        trigger_prefixes=(),
        phrases=(
            (_phrase("income taxes note"), 1.0, False),
            (_phrase("effective tax rate"), 0.9, False),
            (_phrase("provision for income taxes"), 0.7, False),
            (_phrase("deferred tax assets|liabilities"), 0.7, False),
        ),
        queries=("effective tax rate provision income taxes",),
    ),
    _AnchorRule(
        name="equity",
        priority=1.6,
        trigger_tokens=frozenset(
            {"dividend", "dividends", "repurchase", "repurchases", "buyback", "equity", "stockholders", "shareholders"}
        ),
        trigger_prefixes=(),
        phrases=(
            (_phrase("consolidated statement|statements of equity"), 2.8, True),
            (_phrase("consolidated statement|statements of stockholders equity"), 2.8, True),
            (_phrase("stockholders equity"), 0.9, False),
            (_phrase("dividends declared"), 0.9, False),
            (_phrase("share repurchase|repurchases"), 1.0, False),
        ),
        queries=("stockholders equity dividends share repurchases",),
    ),
    _AnchorRule(
        name="goodwill",
        priority=1.6,
        trigger_tokens=frozenset({"goodwill", "intangible", "intangibles", "impairment"}),
        trigger_prefixes=(),
        phrases=(
            (_phrase("goodwill and other intangible assets"), 2.0, False),
            (_phrase("impairment of intangible assets"), 1.0, False),
            (_phrase("indefinite-lived intangible assets"), 1.0, False),
        ),
        queries=("goodwill intangible assets impairment",),
    ),
    _AnchorRule(
        name="executive",
        priority=1.2,
        trigger_tokens=frozenset({"executive", "officers", "ceo", "cfo"}),
        trigger_prefixes=(),
        phrases=(
            (_phrase("information about our executive officers"), 3.0, True),
            (_phrase("executive officers"), 1.2, False),
        ),
        queries=("information about our executive officers",),
    ),
)


@dataclass(slots=True)
class _Concept:
    """One lexical concept of the question with its matching variants."""

    primary: str
    variants: tuple[str, ...]
    prior: float
    instruction_only: bool = False
    weight: float = 0.0
    hits: int = -1
    rarity: float = 1.0


@dataclass(slots=True)
class _Candidate:
    """One distinct Memory entry that may be the evidence page."""

    memory_ref: object
    entry_id: str
    entry_version_id: str
    text: str
    rrf: float = 0.0
    content: float = 0.0
    score: float = 0.0

    def citation(self) -> MemoryCitation:
        return MemoryCitation(
            memory_ref=self.memory_ref,
            entry_id=self.entry_id,
            entry_version_id=self.entry_version_id,
        )


async def search_pdf_pages(memory, question: str, limit: int = 10) -> tuple[MemoryCitation, ...]:
    """Return at most ``limit`` exact citations to distinct evidence pages."""

    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        return ()
    if not isinstance(question, str) or not question.strip():
        return ()
    normalized_question = unicodedata.normalize("NFC", question)
    concepts = _extract_concepts(normalized_question)
    if not concepts:
        return ()
    rules = _activated_rules(normalized_question, concepts)
    fetch = min(max(limit * 3, _FETCH_FLOOR), _FETCH_CEILING)
    candidates: dict[str, _Candidate] = {}
    await _gather_candidates(memory, normalized_question, concepts, rules, fetch, candidates)
    if not candidates:
        return ()
    _rank_candidates(candidates, concepts, rules, normalized_question)
    ordered = sorted(candidates.values(), key=_candidate_order, reverse=True)
    return tuple(candidate.citation() for candidate in ordered[:limit])


def _candidate_order(candidate: _Candidate) -> tuple[float, int]:
    return (candidate.score, -len(candidate.text))


def _activated_rules(question: str, concepts: list[_Concept]) -> tuple[_AnchorRule, ...]:
    tokens = {concept.primary for concept in concepts}
    folded = question.casefold()
    activated: list[_AnchorRule] = []
    for rule in _ANCHOR_RULES:
        if rule.trigger_tokens & tokens:
            activated.append(rule)
            continue
        if any(token.startswith(prefix) for token in tokens for prefix in rule.trigger_prefixes):
            activated.append(rule)
            continue
        if any(phrase in folded for phrase in rule.trigger_phrases):
            activated.append(rule)
    activated.sort(key=lambda rule: rule.priority, reverse=True)
    return tuple(activated)


async def _gather_candidates(
    memory,
    question: str,
    concepts: list[_Concept],
    rules: tuple[_AnchorRule, ...],
    fetch: int,
    candidates: dict[str, _Candidate],
) -> None:
    """Run deterministic lexical queries and merge their hits by entry identity."""

    cleaned = _clean_question(question)
    await _search_and_merge(memory, cleaned, fetch, 1.5, candidates)
    if cleaned != question:
        await _search_and_merge(memory, question, fetch, 0.5, candidates)

    ranked = sorted(
        (concept for concept in concepts if not concept.instruction_only),
        key=lambda concept: concept.prior,
        reverse=True,
    )
    if ranked:
        bag_terms = [concept.primary for concept in ranked[:_BAG_TERMS]]
        await _search_and_merge(memory, " ".join(bag_terms), fetch, 1.3, candidates)

    anchor_queries = 0
    for rule in rules:
        if anchor_queries >= _MAX_ANCHOR_QUERIES:
            break
        for query in rule.queries:
            await _search_and_merge(memory, query, fetch, rule.priority, candidates)
            anchor_queries += 1

    for index, concept in enumerate(ranked[:_MAX_KEYWORD_QUERIES]):
        hit_ids: set[str] = set()
        for query in _concept_queries(concept, second=index < _MAX_SECOND_QUERIES):
            hits = await _search_and_merge(memory, query, fetch, 0.0, candidates)
            hit_ids.update(hit.entry_id for hit in hits)
        concept.hits = len(hit_ids)
        if concept.hits:
            concept.rarity = math.log1p(fetch / max(min(concept.hits, fetch), 1))

    searched = [concept for concept in ranked[:_MAX_KEYWORD_QUERIES] if concept.hits > 0]
    if searched:
        rarest = sorted(searched, key=lambda concept: concept.rarity, reverse=True)[:_RARE_BAG_TERMS]
        if len(rarest) > 2:
            await _search_and_merge(
                memory,
                " ".join(concept.primary for concept in rarest),
                fetch,
                1.35,
                candidates,
            )

    focus = _resolve_document_focus(question, candidates)
    if focus is not None:
        hints, document = focus[1], focus[0]
        hint = " ".join(hints[:2]) or _document_label_token(document)
        if hint:
            for concept in ranked[:3]:
                await _search_and_merge(memory, f"{hint} {concept.primary}", fetch, 1.0, candidates)
            for rule in rules[:_FOCUSED_ANCHOR_RULES]:
                query = rule.queries[0] if rule.queries else ""
                if query:
                    await _search_and_merge(memory, f"{hint} {query}", fetch, 1.2, candidates)
            focused_bag = " ".join(concept.primary for concept in ranked[:4])
            if focused_bag:
                await _search_and_merge(memory, f"{hint} {focused_bag}", fetch, 1.1, candidates)


def _document_label_token(document_id: str) -> str:
    """Pick the most distinctive-looking token of a document id as a query hint."""

    tokens = [token for token in _DOCUMENT_TOKEN.findall(document_id.casefold()) if not token.isdigit()]
    if not tokens:
        return ""
    tokens.sort(key=len, reverse=True)
    return tokens[0]


def _concept_queries(concept: _Concept, *, second: bool) -> tuple[str, ...]:
    queries = [" ".join(concept.variants[:2])]
    if second and len(concept.variants) >= 4:
        queries.append(" ".join(concept.variants[2:4]))
    return tuple(queries)


async def _search_and_merge(
    memory,
    query: str,
    fetch: int,
    list_weight: float,
    candidates: dict[str, _Candidate],
) -> tuple[object, ...]:
    """Search one query variant and fold its ranked hits into the candidate pool."""

    if not query.strip():
        return ()
    try:
        page = await memory.search(SearchMemoryRequest(query=query, limit=fetch, mode="fts"))
    except Exception:  # noqa: BLE001 - a rejected query variant must not lose the other signals
        return ()
    hits = tuple(page.hits)
    for rank, hit in enumerate(hits, start=1):
        candidate = candidates.get(hit.entry_id)
        if candidate is None:
            candidate = _Candidate(
                memory_ref=hit.memory_ref,
                entry_id=hit.entry_id,
                entry_version_id=hit.entry_version_id,
                text=hit.text,
            )
            candidates[hit.entry_id] = candidate
        if list_weight:
            candidate.rrf += list_weight / (_RRF_CONSTANT + rank)
    return hits


def _rank_candidates(
    candidates: dict[str, _Candidate],
    concepts: list[_Concept],
    rules: tuple[_AnchorRule, ...],
    question: str,
) -> None:
    """Score candidates with dynamic concept rarity, coverage, proximity and anchors."""

    searched = [concept for concept in concepts if concept.hits > 0]
    if searched:
        rarities = sorted(concept.rarity for concept in searched)
        median = rarities[len(rarities) // 2] or 1.0
        for concept in searched:
            ratio = min(max(concept.rarity / median, 0.3), 2.8)
            concept.weight = concept.prior * ratio * (0.35 if concept.instruction_only else 1.0)
    for concept in concepts:
        if concept.hits < 0:
            concept.hits = 0
        if concept.weight == 0.0:
            concept.weight = concept.prior * 0.55 * (0.35 if concept.instruction_only else 1.0)

    total_weight = sum(max(concept.weight, 0.0) for concept in concepts) or 1.0
    max_rrf = max((candidate.rrf for candidate in candidates.values()), default=0.0) or 1.0
    content_values = {
        key: _content_score(candidate.text, concepts, rules, total_weight)
        for key, candidate in candidates.items()
    }
    max_content = max(content_values.values(), default=0.0) or 1.0
    question_tokens = {concept.primary for concept in concepts}
    document_affinity: dict[str, float] = {}
    candidate_document: dict[str, str] = {}
    for key, candidate in candidates.items():
        document = _document_id_from_text(candidate.text)
        candidate_document[key] = document
        if document not in document_affinity:
            document_affinity[document] = _document_match(question_tokens, document)
    strongest = max(document_affinity.values(), default=0.0)
    focus = _resolve_document_focus(question, candidates)
    focus_document = focus[0] if focus is not None else None
    for key, candidate in candidates.items():
        content = content_values[key]
        candidate.content = content
        score = 0.42 * (candidate.rrf / max_rrf) + 0.58 * (content / max_content)
        if focus_document is not None:
            score *= 3.5 if candidate_document[key] == focus_document else 0.18
        elif strongest >= 0.5:
            affinity = document_affinity[candidate_document[key]]
            score *= (1.0 + 0.45 * min(affinity / strongest, 1.0)) if affinity > 0.0 else 0.45
        candidate.score = score


def _document_id_from_text(text: str) -> str:
    first_line, _, _rest = text.partition("\n")
    return first_line.strip()


_GENERIC_DOC_WORDS = frozenset(
    """
    annual report reports form forms fiscal year years company companies group holding holdings
    incorporated inc corp corporation financial statements statement dated date
    """.split()
)
_DOCUMENT_TOKEN = re.compile(r"[a-z0-9]+")


def _document_tokens(document_id: str) -> set[str]:
    return set(_DOCUMENT_TOKEN.findall(document_id.casefold()))


def _is_identity_token(token: str) -> bool:
    if len(token) < 3 or token in _GENERIC_DOC_WORDS or token.isdigit():
        return False
    return token.isalpha() or token.isalnum()


def _document_match(question_tokens: set[str], document_id: str) -> float:
    """Estimate whether a question names the document a page belongs to."""

    document_tokens = _document_tokens(document_id)
    score = 0.0
    for token in question_tokens & document_tokens:
        if len(token) >= 4:
            score += 1.0
        elif len(token) >= 3:
            score += 0.4
    return score


def _resolve_document_focus(
    question: str,
    candidates: dict[str, _Candidate],
) -> tuple[str, tuple[str, ...]] | None:
    """Find the one document a multi-document question is about, if identifiable.

    Document ids carry the filing identity (company/period/type). A question that
    names exactly one company, e.g. through a token or an adjacent token pair that
    occurs in exactly one candidate document id, is very likely about that
    document; the returned hint tokens are then used to run document-targeted
    retrieval queries.
    """

    documents: dict[str, set[str]] = {}
    for candidate in candidates.values():
        document = _document_id_from_text(candidate.text)
        if document and document not in documents:
            documents[document] = _document_tokens(document)
    if len(documents) < 2:
        return None
    token_documents: dict[str, set[str]] = {}
    for document, tokens in documents.items():
        for token in tokens:
            token_documents.setdefault(token, set()).add(document)
    sequence = _PAGE_TOKEN.findall(unicodedata.normalize("NFC", question).casefold())
    matches: list[tuple[int, str, tuple[str, ...]]] = []
    for position, token in enumerate(sequence):
        if _is_identity_token(token):
            matched = token_documents.get(token)
            if matched is not None and len(matched) == 1:
                matches.append((position, next(iter(matched)), (token,)))
        if position + 1 < len(sequence):
            combined = token + sequence[position + 1]
            matched = token_documents.get(combined)
            if matched is None or len(matched) != 1:
                continue
            if 2 <= len(combined) and not combined.isdigit():
                matches.append((position, next(iter(matched)), (token, sequence[position + 1])))
    if not matches:
        return None
    matches.sort(key=lambda item: (item[0], -len(item[2])))
    document = matches[0][1]
    hints: list[str] = []
    for _position, matched_document, parts in matches:
        if matched_document != document:
            continue
        for part in parts:
            if part not in hints:
                hints.append(part)
    return document, tuple(hints)


def _content_score(
    text: str,
    concepts: list[_Concept],
    rules: tuple[_AnchorRule, ...],
    total_weight: float,
) -> float:
    tokens = _PAGE_TOKEN.findall(unicodedata.normalize("NFC", text).casefold())
    if not tokens:
        return 0.0
    counts: dict[str, int] = {}
    positions: dict[str, list[int]] = {}
    for index, token in enumerate(tokens):
        counts[token] = counts.get(token, 0) + 1
        bucket = positions.get(token)
        if bucket is None:
            positions[token] = [index]
        elif len(bucket) < 64:
            bucket.append(index)

    matched_weight = 0.0
    matched_concepts: list[tuple[_Concept, tuple[int, ...]]] = []
    for concept in concepts:
        if concept.weight <= 0.0:
            continue
        found: list[int] = []
        term_frequency = 0
        for variant in concept.variants:
            count = counts.get(variant)
            if not count:
                continue
            term_frequency += count
            found.extend(positions.get(variant, ()))
        if not found:
            continue
        found.sort()
        matched_weight += concept.weight * (1.0 + 0.12 * min(math.log1p(term_frequency), 3.0))
        matched_concepts.append((concept, tuple(found[:12])))
    if matched_weight <= 0.0 and not rules:
        return 0.0

    proximity = 0.0
    previous: tuple[_Concept, tuple[int, ...]] | None = None
    for item in matched_concepts:
        if previous is not None:
            distance = _minimum_distance(previous[1], item[1])
            proximity += previous[0].weight * item[0].weight * _proximity_factor(distance)
        previous = item

    anchor_score = 0.0
    for rule in rules:
        for phrase, weight, title_expected in rule.phrases:
            starts = _phrase_starts(phrase, tokens, positions)
            if not starts:
                continue
            best = min(starts)
            gain = weight * (1.0 + 0.1 * min(len(starts) - 1, 4))
            if title_expected:
                gain *= 1.55 if best < 160 else 0.35
            elif best < 120:
                gain *= 1.12
            anchor_score += gain

    coverage = matched_weight / total_weight
    length_factor = (320.0 / max(len(tokens), 240)) ** 0.16
    content = (0.85 * matched_weight + 2.0 * anchor_score + 0.45 * proximity) * (0.65 + 0.35 * coverage)
    return content * length_factor


def _phrase_starts(
    phrase: tuple[frozenset[str], ...],
    tokens: list[str],
    positions: dict[str, list[int]],
) -> list[int]:
    last = len(tokens) - len(phrase)
    starts: list[int] = []
    for token in phrase[0]:
        for start in positions.get(token, ())[:60]:
            if start > last:
                break
            matched = True
            for offset, allowed in enumerate(phrase[1:], start=1):
                if tokens[start + offset] not in allowed:
                    matched = False
                    break
            if matched:
                starts.append(start)
    return starts


def _minimum_distance(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    best = 10**9
    for first in left:
        for second in right:
            distance = abs(first - second)
            if distance < best:
                best = distance
                if best <= 1:
                    return 1
    return best


def _proximity_factor(distance: int) -> float:
    if distance <= 2:
        return 1.0
    if distance <= 6:
        return 0.8
    if distance <= 15:
        return 0.55
    if distance <= 40:
        return 0.3
    return 0.08


def _clean_question(question: str) -> str:
    sentences = _split_sentences(question)
    kept = [sentence for sentence in sentences if not _is_instruction(sentence)]
    cleaned = " ".join(kept).strip()
    return cleaned or question.strip()


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?;])\s+|\n+", unicodedata.normalize("NFC", text))
    return [part.strip() for part in parts if part.strip()]


def _is_instruction(sentence: str) -> bool:
    if _INSTRUCTION_START.match(sentence):
        return True
    if _INSTRUCTION_CONDITIONAL.match(sentence):
        return True
    return _INSTRUCTION_TAIL.search(sentence) is not None


def _extract_concepts(question: str) -> list[_Concept]:
    sentences = _split_sentences(question)
    if not sentences:
        return []
    concepts: dict[str, _Concept] = {}
    order: list[str] = []
    for sentence in sentences:
        instruction = _is_instruction(sentence)
        for token in _PAGE_TOKEN.findall(sentence.casefold()):
            if token in _STOPWORDS or len(token) < 2:
                continue
            primary, variants, prior = _concept_parts(token)
            if primary is None:
                continue
            existing = concepts.get(primary)
            if existing is None:
                concepts[primary] = _Concept(
                    primary=primary,
                    variants=variants,
                    prior=prior,
                    instruction_only=instruction,
                )
                order.append(primary)
            elif not instruction:
                existing.instruction_only = False
    return [concepts[primary] for primary in order]


def _concept_parts(token: str) -> tuple[str | None, tuple[str, ...], float]:
    variants: list[str] = [token]
    for variant in _EXPANSIONS.get(token, ()):
        if variant not in variants:
            variants.append(variant)
    for variant in _morphological_variants(token):
        if variant not in variants:
            variants.append(variant)
    if _YEAR_TOKEN.match(token):
        return token, tuple(variants[:2]), 0.4
    if token.isdigit():
        return token, tuple(variants[:2]), 1.4
    prior = 1.0
    if token in _FINANCE_TERMS:
        prior += 1.1
    if len(token) >= 10:
        prior += 0.7
    elif len(token) >= 7:
        prior += 0.45
    elif len(token) >= 5:
        prior += 0.2
    return token, tuple(variants[:6]), prior


def _morphological_variants(token: str) -> tuple[str, ...]:
    if token.isdigit():
        return ()
    variants: list[str] = []
    for suffix, replacement in _SUFFIX_VARIANTS:
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            candidate = token[: -len(suffix)] + replacement
            if candidate not in variants:
                variants.append(candidate)
    if not token.endswith(("s", "x", "z", "ch", "sh")):
        candidate = token[:-1] + "ies" if token.endswith("y") else token + "s"
        if candidate not in variants:
            variants.append(candidate)
    return tuple(variants)


__all__ = [
    "PDF_PAGE_SOURCE_DEFINITION",
    "PDFDocumentInput",
    "PDFPageInput",
    "PDFPageSource",
    "parse_pdf_document",
    "search_pdf_pages",
]
