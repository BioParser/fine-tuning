"""Output contract shared by dataset building, training, inference and the FastAPI service.

Everything that must be byte-identical between training and production lives here:
column order, which columns the model generates, the system prompt and the label
normalisation. Stdlib only, so the API service can import or vendor it.

If you change anything that alters model input or output (columns, prompt text,
normalisation maps), bump ``PROMPT_VERSION``, rebuild the dataset and retrain.
Adapters record the version they were trained with (see ``train.py``).
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

PROMPT_VERSION = "ets-tsv-v1"

# Column order of the client's ETS TSV exports (identical header in all delivered files).
ETS_COLUMNS: tuple[str, ...] = (
    "references",
    "verbatimScientificName",
    "taxonRank",
    "verbatimTraitName",
    "verbatimTraitUnit",
    "individualCount",
    "measurementValue_min",
    "measurementValue_max",
    "dispersion",
    "statisticalMethod",
    "verbatimTraitValue",
    "sex",
    "lifeStage",
    "measurementMethod",
    "measurementRemarks",
    "measurementAccuracy",
    "measurementDeterminedBy",
    "verbatimLocality",
    "author",
    "associatedReferences",
)

# Columns the model never generates. Client: author is always "BioParser".
# (The delivered labels contain the curator's ORCID here, which must not be learned.)
CONSTANT_COLUMNS: Mapping[str, str] = MappingProxyType({"author": "BioParser"})

# Columns that hold one value per article. The model writes them on the first data
# row only; an empty cell on a later row means "same as the row above" and
# post-processing forward-fills it. `references` is the Google Scholar
# "Cite -> Harvard" citation (client: it is NOT copied from the PDF); repeating it on
# every row costs ~60 tokens/row, about half of all output tokens.
DOCUMENT_COLUMNS: tuple[str, ...] = ("references",)

# What the model emits: header line + one line per record, in this order.
MODEL_COLUMNS: tuple[str, ...] = tuple(c for c in ETS_COLUMNS if c not in CONSTANT_COLUMNS)

# Separator for list values. Client cells with embedded line breaks (e.g. several
# associatedReferences) are joined with it; Darwin Core recommends " | " for lists.
LIST_SEPARATOR = " | "

# Label normalisation for controlled vocabularies, keyed by casefolded value.
# Measured on the delivered data: "Species" vs "species" (2,926 rows), "Male" vs
# "male" vs "Males", typo "Famale", "both"/"sexes combined" for mixed-sex samples.
# Unknown values pass through unchanged. Remove an entry to keep values verbatim.
CANONICAL_VALUES: Mapping[str, Mapping[str, str]] = MappingProxyType(
    {
        "taxonRank": MappingProxyType(
            {
                "superfamily": "Superfamily",
                "family": "Family",
                "subfamily": "Subfamily",
                "tribe": "Tribe",
                "subtribe": "Subtribe",
                "genus": "Genus",
                "species": "Species",
                "subspecies": "Subspecies",
            }
        ),
        "sex": MappingProxyType(
            {
                "male": "Male",
                "males": "Male",
                "female": "Female",
                "females": "Female",
                "famale": "Female",
                "mixed": "Mixed",
                "both": "Mixed",
                "sexes combined": "Mixed",
                "unknown": "Unknown",
            }
        ),
        "lifeStage": MappingProxyType(
            {"adult": "Adult", "adults": "Adult", "juvenile": "Juvenile", "subadult": "Subadult"}
        ),
        "statisticalMethod": MappingProxyType(
            {
                "sd": "SD",
                "standard deviation": "SD",
                "se": "SE",
                "sem": "SE",
                "standard error": "SE",
                "error of the mean": "SE",
                "standard error of the mean": "SE",
            }
        ),
    }
)

# Field definitions, from the client's specification (translated from Finnish) and
# checked against the delivered labels (e.g. statisticalMethod holds SD/SE, i.e. the
# type of `dispersion`, not "mean"). Kept short: the prompt is part of every example.
FIELD_DEFINITIONS: Mapping[str, str] = MappingProxyType(
    {
        "references": (
            'Harvard citation of this article in Google Scholar "Cite > Harvard" style, e.g. '
            '"Abi-Said, M.R. and El Haddad, E., 2023. The first documented record of Acomys '
            "dimidiatus (Cretzschmar, 1826), Eastern Spiny Mouse (Rodentia, Muridae) from "
            'Lebanon. Check List, 19(6), pp.1061-1064." Write it on the first data row only.'
        ),
        "verbatimScientificName": (
            "full scientific name of the measured taxon. Expand abbreviated genus names "
            '(e.g. "H. sapiens" -> "Homo sapiens").'
        ),
        "taxonRank": (
            "Genus (one word: Canis), Species (two words: Canis lupus) or Subspecies "
            "(three words: Canis lupus familiaris); for higher taxa by suffix: -oidea "
            "Superfamily, -idae Family, -inae Subfamily, -ini Tribe."
        ),
        "verbatimTraitName": (
            "trait name as written. If a table abbreviation is defined elsewhere in the "
            'article, write "Full name (ABBR)", e.g. "Head and body length (HB)".'
        ),
        "verbatimTraitUnit": "unit as written, e.g. mm, g, kHz.",
        "individualCount": "number of individuals the value is based on (n).",
        "measurementValue_min": "minimum of the reported range.",
        "measurementValue_max": "maximum of the reported range.",
        "dispersion": "the variation value, e.g. the SD or SE number.",
        "statisticalMethod": "type of the dispersion value: SD or SE.",
        "verbatimTraitValue": "the reported value (usually the mean), as written.",
        "sex": "Male, Female, Mixed or Unknown.",
        "lifeStage": "e.g. Adult, Juvenile, Subadult.",
        "measurementMethod": "how the trait was measured, as written.",
        "measurementRemarks": "other notes on the measurement or the specimens.",
        "measurementAccuracy": "stated precision of the measurement, e.g. 0.1 mm.",
        "measurementDeterminedBy": "who took the measurement, if stated.",
        "verbatimLocality": "sampling locality as written.",
        "associatedReferences": (
            'the source the article cites for this value (e.g. "Darwin et al., 1830"); '
            '"Original study" when the article reports its own measurement.'
        ),
    }
)

assert set(FIELD_DEFINITIONS) == set(MODEL_COLUMNS), "every generated column needs a definition"

MODEL_HEADER = "\t".join(MODEL_COLUMNS)


def _build_system_prompt() -> str:
    definitions = "\n".join(f"- {col}: {FIELD_DEFINITIONS[col]}" for col in MODEL_COLUMNS)
    return (
        "You are BioParser. Extract every trait measurement reported in the scientific "
        "article (Markdown converted from a PDF) into a TSV table.\n"
        "\n"
        "Output rules:\n"
        f"- The first line is exactly this header ({len(MODEL_COLUMNS)} tab-separated columns):\n"
        f"{MODEL_HEADER}\n"
        "- Then one line per record: one taxon, one trait, one reported value group "
        "(e.g. per sex, life stage or locality as reported).\n"
        "- Separate cells with one TAB. Never put a TAB or line break inside a cell. "
        "Leave a cell empty when the article does not state it.\n"
        "- Output only the TSV: no Markdown, no code fences, no comments.\n"
        "\n"
        "Columns:\n"
        f"{definitions}"
    )


SYSTEM_PROMPT: str = _build_system_prompt()


def user_message(markdown: str) -> str:
    """User turn. `markdown` must already be cleaned with markdown.clean_markdown."""
    return f"<article>\n{markdown.strip()}\n</article>"


def build_messages(markdown: str, target_tsv: str | None = None) -> list[dict[str, str]]:
    """Chat messages for one article; add the assistant turn for training examples.

    Applies markdown.clean_markdown, so training, offline inference and the FastAPI
    service cannot diverge on input preprocessing.
    """
    from .markdown import clean_markdown  # local import keeps schema import-light

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message(clean_markdown(markdown))},
    ]
    if target_tsv is not None:
        messages.append({"role": "assistant", "content": target_tsv})
    return messages
