"""This module finds PDF links on the landing pages of publishers and repositories.

It also knows the PDF URL patterns of some publishers.
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import quote, urljoin, urlsplit

from bs4 import BeautifulSoup, Tag

# Meta tags with the PDF URL: Highwire Press tags (most publishers, OJS and DSpace),
# bepress Digital Commons and EPrints.
PDF_META = (
    "citation_pdf_url",
    "bepress_citation_pdf_url",
    "eprints.document_url",
    "wkhealth_pdf_url",
    "citation_fulltext_pdf_url",
)
_PDF_TEXT = re.compile(r"\b(pdf|download (the )?(full )?(text|article|paper)|full[- ]text)\b", re.I)
_DEPRIORITISE = re.compile(
    r"suppl|supplement|appendix|figure|fig\d|table\d|esm|mmc\d|cover|toc|erratum", re.I
)


def _attr(tag: Tag, name: str) -> str:
    """Return the attribute as one string. bs4 gives a list for attributes with more than one value."""
    value = tag.get(name)
    if isinstance(value, list):
        return " ".join(value)
    return value or ""


def pdf_links_from_html(html: str | bytes, base_url: str, limit: int = 12) -> list[str]:
    """Return the PDF URLs of an HTML page. The best URLs come first: the meta tags, then the links."""
    soup = BeautifulSoup(html, "html.parser")
    meta: list[str] = []
    links: list[str] = []
    weak: list[str] = []
    for m in soup.find_all("meta"):
        name = (_attr(m, "name") or _attr(m, "property")).strip().lower()
        if name in PDF_META and _attr(m, "content"):
            meta.append(urljoin(base_url, _attr(m, "content").strip()))
    for link in soup.find_all("link", href=True):
        if "pdf" in _attr(link, "type").lower():
            links.append(urljoin(base_url, _attr(link, "href")))
    for a in soup.find_all("a", href=True):
        href = _attr(a, "href").strip()
        low = href.lower()
        if low.startswith(("javascript:", "mailto:", "#", "tel:")):
            continue
        text = " ".join(a.get_text(" ", strip=True).split()).lower()[:80]
        label = " ".join([text, _attr(a, "title").lower(), _attr(a, "aria-label").lower()])
        path = urlsplit(low).path
        looks_pdf = (
            path.endswith(".pdf")
            or "/pdf" in path
            or "pdf=render" in low
            or "/ndownloader/files/" in low  # Figshare, for example the thesis repository of a university
            or "/bitstream/" in low  # DSpace
            or ("download" in low and "pdf" in label)
            or bool(_PDF_TEXT.search(label))
        )
        if looks_pdf:
            url = urljoin(base_url, href)
            (weak if _DEPRIORITISE.search(low + " " + label) else links).append(url)
    out: list[str] = []
    seen: set[str] = set()
    for url in meta + links + weak:
        if url not in seen and urlsplit(url).scheme in ("http", "https"):
            seen.add(url)
            out.append(url)
    return out[:limit]


_PLOS = {"pone": "plosone", "pbio": "plosbiology", "pcbi": "ploscompbiol", "pgen": "plosgenetics",
         "ppat": "plospathogens", "pntd": "plosntds", "pmed": "plosmedicine"}  # fmt: skip


def publisher_pdf_urls(doi: str, crossref_item: dict[str, Any] | None = None) -> list[str]:
    """Return the direct PDF URLs that publishers use for open-access articles."""
    d = doi.lower()
    prefix, _, suffix = d.partition("/")
    urls: list[str] = []
    # Old Wiley DOIs (SICI) contain "#", "<" and ";". A raw "#" starts the URL fragment and cuts the URL.
    q = quote(doi, safe="/")
    if prefix in ("10.1007", "10.1186"):  # Springer and BMC
        urls.append(f"https://link.springer.com/content/pdf/{q}.pdf")
    elif prefix in ("10.1002", "10.1111"):  # Wiley
        urls.append(f"https://onlinelibrary.wiley.com/doi/pdfdirect/{q}?download=true")
    elif prefix == "10.1080":  # Taylor & Francis
        urls.append(f"https://www.tandfonline.com/doi/pdf/{q}?download=true")
    elif prefix == "10.1515":  # De Gruyter, for example the journal Mammalia
        urls.append(f"https://www.degruyter.com/document/doi/{q}/pdf")
    elif prefix == "10.3389":  # Frontiers
        urls.append(f"https://www.frontiersin.org/articles/{q}/pdf")
    elif prefix == "10.1098":  # Royal Society
        urls.append(f"https://royalsocietypublishing.org/doi/pdf/{q}")
    elif prefix == "10.1038":  # Nature journals
        urls.append(f"https://www.nature.com/articles/{quote(suffix)}.pdf")
    elif prefix == "10.7717" and (m := re.match(r"peerj\.(\d+)", suffix)):
        urls.append(f"https://peerj.com/articles/{m.group(1)}.pdf")
    elif prefix == "10.1371" and (m := re.match(r"journal\.(\w{4})\.", suffix)) and m.group(1) in _PLOS:
        urls.append(f"https://journals.plos.org/{_PLOS[m.group(1)]}/article/file?id={q}&type=printable")
    elif prefix == "10.1016":  # Elsevier: the URL contains the PII. It operates only for open-access PDFs.
        for alt in (crossref_item or {}).get("alternative-id") or []:
            # For example S1319562X22007082. A PII has 17 characters: S, the 8-character ISSN,
            # a 2-digit year, a 5-digit item number and 1 check character.
            pii = re.sub(r"[^0-9A-Za-z]", "", alt)
            if re.fullmatch(r"S\d{7}[\dX]\d{7}[\dX]", pii, re.I):
                urls.append(
                    f"https://www.sciencedirect.com/science/article/pii/{pii}/pdfft?isDTMRedir=true&download=true"
                )
                break
    return urls
