from __future__ import annotations
import json
from dataclasses import dataclass, field
from uuid import uuid4
from enum import Enum
from typing import Optional, Set, List, Any, Dict, Tuple, Union
import os
import re
from openai import OpenAI
from collections import defaultdict

# from Segmentation import *
from KnowGraph import *
from transformers import PreTrainedModel
from FlagEmbedding import BGEM3FlagModel


EMBED_MODEL = BGEM3FlagModel("BAAI/bge-m3")
HAS_EMBEDDINGS = True

MODEL_ID = "gpt-oss:120b-cloud"
BASE_URL = "http://localhost:11434/v1"
LLM = OpenAI(
    base_url=BASE_URL,
    api_key="eb8c507c1e284d8f9bf5867f42b6cd8b.saBEZsoabpJuz3J6pIwAW1uk",
)


def llm_generate(
    prompt: str,
    model: str = MODEL_ID,
    llm=LLM,
    topic="You write a set of hypothetical questions for retrieval relevant facts.",
) -> str:
    resp = llm.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": topic},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


# Optional imports with graceful fallback
try:
    from deep_translator import GoogleTranslator

    HAS_TRANSLATOR = True
except ImportError:
    HAS_TRANSLATOR = False
    GoogleTranslator = None
    print("Warning: deep_translator not available")

try:
    import numpy as np
    from numpy import array

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None
    print("Warning: numpy not available")

try:
    from sklearn.metrics.pairwise import cosine_similarity

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    cosine_similarity = None
    print("Warning: scikit-learn not available")

# try:
#     from FlagEmbedding import BGEM3FlagModel

#     EMBED_MODEL = BGEM3FlagModel("BAAI/bge-m3")
#     HAS_EMBEDDINGS = True
# except ImportError:
#     EMBED_MODEL = None
#     HAS_EMBEDDINGS = False
#     print("Warning: FlagEmbedding not available for reference type detection")


class ReferenceType(str, Enum):
    """Types of references between policy documents"""

    EXTENDS = "extends"
    REPLACES = "replaces"
    AMENDS = "amends"
    REFERENCES = "references"
    SUPERSEDES = "supersedes"


class PolicyRegistry:
    """Global registry to manage PolicyDocument by ID as primary key"""

    _instance = None
    _policies: Dict[str, PolicyDocument] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def register(self, policy_doc: PolicyDocument) -> None:
        """Register a PolicyDocument by its ID"""
        self._policies[policy_doc.id] = policy_doc

    def get(self, policy_id: str) -> Optional[PolicyDocument]:
        """Get a PolicyDocument by ID"""
        return self._policies.get(policy_id)

    def get_or_create(self, policy_id: str, subject: str = "") -> PolicyDocument:
        """Get or create a PolicyDocument by ID"""
        if policy_id not in self._policies:
            self._policies[policy_id] = PolicyDocument(
                id=policy_id, subject=subject, decisions=()
            )
        return self._policies[policy_id]


@dataclass()
class PolicyDocument:
    id: str
    subject: str = ""
    full_content: str = ""
    decisions: tuple = field(default_factory=tuple)  # List of LawUnit
    sef_fact: str

    def __post_init__(self):
        """Automatically register this policy in the registry"""
        registry = PolicyRegistry()
        registry.register(self)

    def add_decision(self, law_unit: LawUnit) -> PolicyDocument:
        """Return a new PolicyDocument with an added decision (immutable)"""
        new_decisions = tuple(list(self.decisions) + [law_unit])
        return PolicyDocument(id=self.id, subject=self.subject, decisions=new_decisions)


@dataclass
class LawUnit:
    """Represents a unit of law with content and references to other policy documents"""

    type_unit: str = ""
    content: str = ""
    id: str = field(default_factory=lambda: str(uuid4()))
    content_embed: Any = None
    hypothetical_embed: tuple = field(default_factory=tuple)
    # List of (policy_id, ReferenceType) pairs - stored as IDs, not objects
    references: tuple = field(default_factory=tuple)
    cross_references: tuple = field(default_factory=tuple)

    def __init__(
        self,
        type_unit="",
        content="",
        embed=EMBED_MODEL,
        llm=LLM,
        prompt_path=r"prompts/prompt_hyp.txt",
    ):
        self.type_unit = type_unit
        self.content = content
        self.id = str(uuid4())
        self.content_embed = None
        self.hypothetical_embed = ()
        self.references = ()
        self.cross_references()
        adjust_law(self)

        # Skip embedding initialization if prompt file doesn't exist
        if not os.path.exists(prompt_path):
            return

        try:
            with open(prompt_path, "r", encoding="utf-8") as file:
                prefix = file.read()
        except:
            print("ERROR DISMISSING")
            return

        full_prompt = prefix + self.content
        hyp_json = llm_generate(full_prompt, model=MODEL_ID)
        print(hyp_json)
        try:
            # 1. Parse the JSON string into a Python Dictionary
            data = json.loads(hyp_json)

            # 2. Extract the list
            questions_list = data.get("investigative_questions", [])
        except json.JSONDecodeError:
            raise KeyError("Error: LLM did not return valid JSON")

        if EMBED_MODEL:
            self.content_embed = embed.encode(self.content)["dense_vecs"]
            self.hypothetical_embed = embed.encode(questions_list)["dense_vecs"]

    def get_max_embed(self, input_vec):
        """Get the maximum embedding similarity score"""
        max_ = self.content_embed @ input_vec
        for embed in self.hypothetical_embed:
            sim = input_vec @ embed
            if sim > max_:
                max_ = sim
        return max_

    def get_cross_references(self) -> tuple:
        """
        Find cross-references in the content that match the pattern of this unit's type and id.

        The method analyzes the id to determine the pattern (number or letter) and then
        searches for similar patterns in the content.

        Examples:
            - If type_unit="Điều" and id="Điều 1", finds patterns like "Điều 2", "Điều 3", etc.
            - If type_unit="Điểm" and id="Điểm a", finds patterns like "Điểm b", "Điểm c", etc.
            - If type_unit="Article" and id="Article 5", finds "Article 1", "Article 10", etc.

        Returns:
            tuple: A tuple of cross-reference strings found in the content (excluding self-reference)
        """
        if not self.content or not self.type_unit or not self.id:
            return ()

        import re
        import unicodedata

        def is_vietnamese_letter(char: str) -> bool:
            """Check if a character is a letter (including Vietnamese)"""
            if not char:
                return False
            # Check if it's a letter using Unicode category
            return unicodedata.category(char).startswith("L")

        def is_single_letter(s: str) -> bool:
            """Check if string is a single letter (including Vietnamese letters like a, ă, â, etc.)"""
            if len(s) != 1:
                return False
            return is_vietnamese_letter(s)

        type_esc = re.escape(self.type_unit)

        # Analyze the id to determine the pattern (number or letter)
        # Extract what comes after the type_unit in the id
        # Use Unicode flag for proper Vietnamese matching
        id_pattern = re.compile(rf"{type_esc}\s+(\S+)", re.IGNORECASE | re.UNICODE)
        id_match = id_pattern.search(self.id)

        if not id_match:
            return ()

        id_value = id_match.group(1).strip()
        # Remove trailing punctuation like ".", ":", ")"
        id_value = re.sub(r"[.:)\-,;]+$", "", id_value)

        # Vietnamese letter character class (lowercase and uppercase)
        viet_letters = r"a-zA-ZàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđĐÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴ"

        # Determine if the id uses a number or letter pattern
        if id_value.isdigit():
            # Pattern: type_unit followed by number (e.g., "Điều 1", "Article 10")
            search_pattern = re.compile(
                rf"({type_esc}\s+\d+)", re.IGNORECASE | re.UNICODE
            )
        elif is_single_letter(id_value):
            # Pattern: type_unit followed by single letter (e.g., "Điểm a", "điểm đ")
            search_pattern = re.compile(
                rf"({type_esc}\s+[{viet_letters}])(?![{viet_letters}])",
                re.IGNORECASE | re.UNICODE,
            )
        else:
            # Mixed pattern (e.g., "2a", "10b") - search for similar
            # Check if starts with digits
            if id_value and id_value[0].isdigit():
                search_pattern = re.compile(
                    rf"({type_esc}\s+\d+[{viet_letters}]*)", re.IGNORECASE | re.UNICODE
                )
            else:
                # Default to searching for the type_unit followed by any word
                search_pattern = re.compile(
                    rf"({type_esc}\s+\S+)", re.IGNORECASE | re.UNICODE
                )

        # Find all matches in content
        matches = search_pattern.findall(self.content)

        # Normalize and deduplicate, excluding self-reference
        cross_refs = set()
        # Normalize self id for comparison (remove punctuation, lowercase)
        self_id_normalized = re.sub(r"[.:)\-,;]+", "", self.id.strip()).lower()

        for match in matches:
            normalized = match.strip()
            # Normalize for comparison
            match_normalized = re.sub(r"[.:)\-,;]+", "", normalized).lower()
            # Exclude self-reference
            if match_normalized != self_id_normalized:
                cross_refs.add(normalized)

        # Update the cross_references attribute and return
        self.cross_references = tuple(sorted(cross_refs))
        return self.cross_references

    def add_reference(
        self, policy_id_or_doc: Union[str, PolicyDocument], ref_type: ReferenceType
    ) -> LawUnit:
        """Add a reference by policy ID or PolicyDocument object (immutable)"""
        # Extract ID if PolicyDocument object is passed
        if isinstance(policy_id_or_doc, PolicyDocument):
            policy_id = policy_id_or_doc.id
        else:
            policy_id = policy_id_or_doc

        self.references = tuple(list(self.get_references()) + [(policy_id, ref_type)])

    def get_references(self) -> List[Tuple[PolicyDocument, ReferenceType]]:
        """Resolve all reference IDs to actual PolicyDocument objects"""
        registry = PolicyRegistry()
        resolved_refs = []
        for policy_id, ref_type in self.references:
            policy_doc = registry.get(policy_id)
            if policy_doc:
                resolved_refs.append((policy_doc, ref_type))
        return resolved_refs

    def hop(self) -> Set[PolicyDocument]:
        registry = PolicyRegistry()
        set_policy = set()
        for policy, ref in self.references:
            if ref != ReferenceType.REPLACES:
                add = registry.get(policy)
                if add is not None:
                    set_policy.add(add)
        return set_policy


def get_documents(folder: str = None) -> List[str]:
    """
    Retrieve all files from the document folder.
    Returns a list of file contents as strings.
    Supports .txt, .json, and .md files.

    If folder is None, looks for 'documents' folder relative to workspace root.
    """
    # If no folder specified, use default relative to script location
    if folder is None:
        # Get the workspace root (parent of utils folder)
        utils_dir = os.path.dirname(os.path.abspath(__file__))
        workspace_root = os.path.dirname(utils_dir)
        folder = os.path.join(workspace_root, "documents")

    documents = []

    # Check if folder exists
    if not os.path.exists(folder):
        print(f"Warning: Folder '{folder}' does not exist")
        print(f"Expected at: {os.path.abspath(folder)}")
        return documents

    # Get all files in the folder
    try:
        files_found = os.listdir(folder)
        print(f"Files in folder: {files_found}")

        for filename in files_found:
            filepath = os.path.join(folder, filename)

            # Skip directories
            if os.path.isdir(filepath):
                continue

            # Only process supported file types
            if filename.endswith((".txt", ".json", ".md")):
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        content = f.read()
                        documents.append(content)
                        print(f"Loaded: {filename}")
                except Exception as e:
                    print(f"Error reading {filename}: {e}")
    except Exception as e:
        print(f"Error reading folder: {e}")

    return documents


def translate_to_english(text):
    """
    Translates the given text to English using GoogleTranslator from deep-translator.
    Handles empty or invalid inputs gracefully. Returns original text if translator unavailable.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Input must be a non-empty string.")

    if not HAS_TRANSLATOR:
        print("Warning: Translation unavailable, returning original text")
        return text

    try:
        translated = GoogleTranslator(source="auto", target="en").translate(text)
        return translated
    except Exception as e:
        print(f"Translation failed: {e}, returning original text")
        return text


def extract_document_ids(text: str) -> List[str]:
    r"""
    Extract all document IDs from text using the pattern: xx/xx or xx/xx/xx
    Pattern: [A-Za-z0-9\-À-ỿ]+/[A-Za-z0-9\-À-ỿ]+(?:/[A-Za-z0-9\-À-ỿ]+)?
    Supports Vietnamese characters (Đ, ă, â, ê, ô, etc.)
    Examples: 2411/QĐ-NHNN, 48/2024/TT-NHNN

    Args:
        text: The text to search for document IDs

    Returns:
        List of unique document IDs found
    """
    import re

    pattern = r"[A-Za-z0-9\-À-ỿ]+/[A-Za-z0-9\-À-ỿ]+(?:/[A-Za-z0-9\-À-ỿ]+)?"
    matches = re.findall(pattern, text)
    return list(set(matches))  # Return unique IDs


def extract_ids_with_context(text: str) -> List[str]:
    r"""
    Extract document IDs with "Number:" or "Só:" prefix.
    Pattern: (Number|Số):\s*([A-Za-z0-9\-À-ỿ]+/[A-Za-z0-9\-À-ỿ]+(?:/[A-Za-z0-9\-À-ỿ]+)?)

    Supports Vietnamese characters (Đ, ă, â, ê, ô, etc.)
    Examples: 2411/QĐ-NHNN, 48/2024/TT-NHNN

    Args:
        text: The text to search

    Returns:
        List of IDs found after "Number:" or "Số:" prefix
    """
    import re

    pattern = (
        r"(?:Number|Số):\s*([A-Za-z0-9\-À-ỿ]+/[A-Za-z0-9\-À-ỿ]+(?:/[A-Za-z0-9\-À-ỿ]+)?)"
    )
    matches = re.findall(pattern, text)
    return matches


def extract_context_around_id(text: str, doc_id: str, word_count: int = 5) -> str:
    """
    Extract context around a document ID.
    Gets word_count words before and after the ID.

    Args:
        text: The full text
        doc_id: The document ID to find
        word_count: Number of words to extract around the ID

    Returns:
        Context string surrounding the ID
    """
    # Escape special regex characters in doc_id
    escaped_id = re.escape(doc_id)

    # Find position of the ID
    match = re.search(escaped_id, text)
    if not match:
        return ""

    start_pos = match.start()
    end_pos = match.end()

    # Extract words before the ID
    text_before = text[:start_pos]
    words_before = text_before.split()[-word_count:]  # Last N words

    # Extract words after the ID
    text_after = text[end_pos:]
    words_after = text_after.split()[:word_count]  # First N words

    # Combine context
    context = " ".join(words_before + [doc_id] + words_after)
    return context


def determine_reference_type(
    context: str, model=EMBED_MODEL
) -> Tuple[ReferenceType, float]:
    """
    Determine the type of reference by embedding the context and comparing with reference types.

    Args:
        context: Context string surrounding the reference ID
        model: Embedding model (BGEM3FlagModel)

    Returns:
        Tuple of (ReferenceType, similarity_score)
    """
    if model is None or not context.strip() or not HAS_EMBEDDINGS or not HAS_SKLEARN:
        return ReferenceType.REFERENCES, 0.0

    # Embed the context
    context_embedding = model.encode(
        [context], return_dense=True, return_sparse=False, return_colbert_vecs=False
    )
    context_vec = context_embedding["dense_vecs"][0].reshape(1, -1)

    # Embed all reference types
    ref_types_text = {
        ReferenceType.EXTENDS: "extends the scope and provisions of the referenced document",
        ReferenceType.REPLACES: "replaces and supersedes the previous version",
        ReferenceType.AMENDS: "amends and modifies the referenced document",
        ReferenceType.REFERENCES: "references the related policy or regulation",
        ReferenceType.SUPERSEDES: "supersedes and invalidates the earlier version",
    }

    # Calculate similarity with each reference type
    similarities = {}
    ref_embeddings = model.encode(
        list(ref_types_text.values()),
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    ref_vecs = ref_embeddings["dense_vecs"]

    max_similarity = -1
    best_type = ReferenceType.REFERENCES

    for idx, (ref_type, _) in enumerate(ref_types_text.items()):
        ref_vec = ref_vecs[idx].reshape(1, -1)
        similarity = cosine_similarity(context_vec, ref_vec)[0][0]
        similarities[ref_type] = similarity

        if similarity > max_similarity:
            max_similarity = similarity
            best_type = ref_type

    return best_type, max_similarity


def extract_ids_with_types(
    text: str, model=EMBED_MODEL
) -> List[Tuple[str, ReferenceType, float]]:
    """
    Extract all document IDs with their automatically determined reference types.

    Args:
        text: The text to search
        model: Embedding model

    Returns:
        List of tuples (doc_id, reference_type, confidence)
    """
    ids = extract_document_ids(text)
    results = []

    for doc_id in ids:
        context = extract_context_around_id(text, doc_id, word_count=5)
        ref_type, confidence = determine_reference_type(context, model)
        results.append((doc_id, ref_type, confidence))

    return results


def get_doc_id(doc):
    pattern = r"Number:\s*([A-Za-z0-9]+/[A-Za-z0-9]+(?:/[A-Za-z0-9]+)?)"
    translate = translate_to_english(doc)
    return extract_ids_with_context(translate)[0]


def get_or_create_doc(registry: PolicyRegistry, doc, list_law):
    id = get_doc_id(doc)
    policy = PolicyRegistry.get_or_create(id)
    for law in list_law:
        policy.add_decision(law)


def adjust_law(law_unit: LawUnit):
    """
    Adjust a LawUnit by extracting referenced document IDs and determining their reference types.

    Args:
        law_unit: The LawUnit to adjust

    Returns:
        Updated LawUnit with extracted references
    """
    doc = law_unit.content
    ref_ids = extract_document_ids(doc)

    for id in ref_ids:
        context = extract_context_around_id(doc, id)
        # determine_reference_type returns (ReferenceType, confidence)
        ref_type, confidence = determine_reference_type(context=context)
        law_unit.add_reference(policy_id_or_doc=id, ref_type=ref_type)


def extract_law_units(
    translated_policy: str, include_preamble: bool = True, strip_body: bool = False
) -> Tuple[List[LawUnit], List[PolicyDocument]]:
    """
    Extract law units from policy text and convert to PolicyDocument objects with references.

    Args:
        translated_policy: The policy text to parse
        include_preamble: Whether to include preamble as a unit
        strip_body: Whether to strip whitespace from body text

    Returns:
        Tuple of (law_units, policy_documents)
    """
    # Keep \n structure; only normalize CRLF -> LF
    text = translated_policy.replace("\r\n", "\n").replace("\r", "\n")

    header_patterns = [
        (
            "chapter",
            re.compile(
                r"^(?:CHAPTER|Chapter)\s+([IVXLCDM]+|\d+)\b[ \t]*[.:–-]?[ \t]*(.*)$",
                re.M,
            ),
        ),
        (
            "section",
            re.compile(
                r"^(?:SECTION|Section)\s+(\d+[A-Za-z]?)\b[ \t]*[.:–-]?[ \t]*(.*)$", re.M
            ),
        ),
        (
            "article",
            re.compile(
                r"^(?:ARTICLE|Article|Art\.)\s+(\d+[A-Za-z]?)\b[ \t]*[.:–-]?[ \t]*(.*)$",
                re.M,
            ),
        ),
        (
            "article",
            re.compile(
                r"^(?:ĐIỀU|Điều|DIEU|Dieu)\s+(\d+[A-Za-z]?)\b[ \t]*[.:–-]?[ \t]*(.*)$",
                re.M,
            ),
        ),
    ]

    hits = []
    for level, pat in header_patterns:
        for m in pat.finditer(text):
            hits.append(
                {
                    "level": level,
                    "number": m.group(1).strip(),
                    "title": (m.group(2) or ""),
                    "start": m.start(),
                    "header_end": m.end(),
                }
            )

    if not hits:
        raise KeyError()

    hits.sort(key=lambda x: (x["start"], -x["header_end"]))
    deduped = []
    for h in hits:
        if not deduped or h["start"] != deduped[-1]["start"]:
            deduped.append(h)

    units: List[LawUnit] = []

    if include_preamble and deduped[0]["start"] > 0:
        pre = text[: deduped[0]["start"]]
        if strip_body:
            pre = pre.strip()
        if pre:
            units.append(LawUnit("preamble", None, "", pre, (0, deduped[0]["start"])))

    for i, h in enumerate(deduped):
        body_start = h["header_end"]
        body_end = deduped[i + 1]["start"] if i + 1 < len(deduped) else len(text)
        body = text[body_start:body_end]  # <-- keeps trailing \n\n for last article
        if strip_body:
            body = body.strip()

        units.append(
            LawUnit(
                h["level"],
                h["number"],
                h["title"].strip(),
                body,
                (h["start"], body_end),
            )
        )

    # Convert LawUnits to PolicyDocuments with extracted references
    policy_docs: List[PolicyDocument] = []
    for unit in units:
        # Extract referenced document IDs and their types from the unit content
        ids_with_types = extract_ids_with_types(unit.content, EMBED_MODEL)

        # Create references as tuples of (policy_id, ReferenceType)
        references = tuple(ids_with_types) if ids_with_types else ()

        # Create LawUnit with references
        law_unit_with_refs = LawUnit(
            id=unit.id,
            type_unit=unit.type_unit,
            content=unit.content,
            references=references,
        )

        # Create PolicyDocument with the law unit as a decision
        # Use type_unit as the policy ID if available
        policy_id = f"{unit.type_unit}_{unit.id}" if unit.type_unit else unit.id
        policy_doc = PolicyDocument(
            id=policy_id,
            subject=unit.type_unit or "Policy Unit",
            decisions=(law_unit_with_refs,),
        )
        policy_docs.append(policy_doc)

    return units, policy_docs


def test_2():
    law_1 = LawUnit(
        type_unit="Article 1",
        content="""
The maximum interest rates for Vietnamese Dong deposits of organizations (excluding credit institutions and branches of foreign banks) and individuals at credit institutions and branches of foreign banks, as stipulated in Circular No. 07/2014/TT-NHNN dated March 17, 2014, are as follows:

1. The maximum interest rate applicable to demand deposits and deposits with a term of less than 1 month is 0.5%/year.

2. The maximum interest rate applicable to deposits with a term from 1 month to less than 6 months is 4.75%/year; however, for People's Credit Funds and Microfinance Institutions, the maximum interest rate for deposits with a term from 1 month to less than 6 months is 5.25%/year.
""",
    )
    law_2 = LawUnit(
        type_unit="Article 2",
        content="""
1. This Decision shall take effect from June 19, 2023 and replaces Decision No. 951/QD-NHNN dated May 23, 2023 of the Governor of the State Bank of Vietnam on the maximum interest rate for Vietnamese Dong deposits of organizations and individuals at credit institutions and branches of foreign banks as stipulated in Circular No. 07/2014/TT-NHNN dated March 17, 2014.

2. For interest rates on Vietnamese Dong time deposits of organizations and individuals at credit institutions and branches of foreign banks arising before the effective date of this Decision, they shall be applied until the end of the term; if, after the agreed term, the organization or individual does not withdraw the deposit, the credit institution or branch of the foreign bank shall apply the interest rate for deposits as stipulated in this Decision.""",
    )
    law_3 = LawUnit(
        type_unit="Article 3",
        content="  The Chief of the Office, the Director of the Monetary Policy Department, and the heads of units under the State Bank of Vietnam, credit institutions, and branches of foreign banks are responsible for implementing this Decision.",
    )
    policy2 = PolicyDocument(
        id="2411/QD-NHNN",
        full_content="""
Article 1. The maximum interest rates for Vietnamese Dong deposits of organizations (excluding credit institutions and branches of foreign banks) and individuals at credit institutions and branches of foreign banks, as stipulated in Circular No. 07/2014/TT-NHNN dated March 17, 2014, are as follows:

1. The maximum interest rate applicable to demand deposits and deposits with a term of less than 1 month is 0.5%/year.

2. The maximum interest rate applicable to deposits with a term from 1 month to less than 6 months is 4.75%/year; however, for People's Credit Funds and Microfinance Institutions, the maximum interest rate for deposits with a term from 1 month to less than 6 months is 5.25%/year.

Article 2.

1. This Decision shall take effect from June 19, 2023 and replaces Decision No. 951/QD-NHNN dated May 23, 2023 of the Governor of the State Bank of Vietnam on the maximum interest rate for Vietnamese Dong deposits of organizations and individuals at credit institutions and branches of foreign banks as stipulated in Circular No. 07/2014/TT-NHNN dated March 17, 2014.

2. For interest rates on Vietnamese Dong time deposits of organizations and individuals at credit institutions and branches of foreign banks arising before the effective date of this Decision, they shall be applied until the end of the term; if, after the agreed term, the organization or individual does not withdraw the deposit, the credit institution or branch of the foreign bank shall apply the interest rate for deposits as stipulated in this Decision.

Article 3. The Chief of the Office, the Director of the Monetary Policy Department, and the heads of units under the State Bank of Vietnam, credit institutions, and branches of foreign banks are responsible for implementing this Decision.""",
        decisions=(law_1, law_2, law_3),
    )
    seg = Segmentation()
    know = Knowlegde()
    know.full_pipeline_chunks_to_knowledge(
        seg.segment_whole_to_chunks(policy2.full_content, chunk_size=500, overlap=50)
    )
    for knowlegdge in know.knowledge:
        print(knowlegdge)
    policy = PolicyDocument(id="07/2014/TT-NHNN", full_content="Hello" "")
    # adjust_law(law_unit=law_1)

    # for doc in law_1.hop():
    #     print(doc)

    # Show extracted references
    print(f"References extracted: {law_1.references}")
    list_fact_extract = [
        "This DECISION prescribes the maximum interest rate for deposits in Vietnamese Dong",
        "This DECISION takes effect from June 19, 2023",
        "This DECISION replaces Decision No. 951/QD-NHNN dated May 23, 2023",
        "This DECISION assigns implementation responsibility to the Chief of Office",
        "This DECISION assigns implementation responsibility to the Director of the Monetary Policy Department",
        "This DECISION assigns implementation responsibility to the Heads of units under the State Bank of Vietnam",
        "This DECISION assigns implementation responsibility to credit institutions",
        "This DECISION assigns implementation responsibility to foreign bank branches",
    ]

    fact_embeded = []
    for fact in list_fact_extract:
        fact_embeded.append(EMBED_MODEL.encode(fact)["dense_vecs"])

    document_dictionary = load_document_data(r"form\scanned_doc.json")

    law_key_dict = defaultdict(list)
    list_populate = []
    for key in document_dictionary.keys():
        key = key + " is " + document_dictionary[key]
        encode_key = EMBED_MODEL.encode(key)["dense_vecs"]
        for fact in fact_embeded:
            f_k = fact @ encode_key
            print(f_k)
            if f_k > 0.45:
                list_populate.append(key)
        for law in policy2.decisions:
            print(law.type_unit)
            if law.get_max_embed(encode_key) > 0.55:
                law_key_dict[law.type_unit].append(key)
                print("HIT!!")
    for law in law_key_dict.keys():
        for pop in list_populate:
            if pop not in law_key_dict[law]:
                law_key_dict[law].append(pop)
    for law in law_key_dict.keys():
        print(law, "   ", law_key_dict[law])


def test_1():
    text = "Number: 2411/QD-NHNN and another Number: 1234/ABC/XYZ"
    pattern = r"Number:\s*([A-Za-z0-9]+/[A-Za-z0-9]+(?:/[A-Za-z0-9]+)?)"

    matches = re.findall(pattern, text)
    print(matches)  # ['2411/QD-NHNN', '1234/ABC/XYZ']

    # With match objects to get full match:
    for match in re.finditer(pattern, text):
        print(f"Found: {match.group(0)}")  # Full match including "Number:"
        print(f"ID: {match.group(1)}")  # Just the ID

    for doc in get_documents():
        # Translate document
        try:
            translated = translate_to_english(doc)
            print(f"Translated: {translated[:100]}...\n")
        except Exception as e:
            print(f"Translation error: {e}\n")

        # Extract all IDs
        ids = extract_document_ids(doc)
        print(f"Found IDs: {ids}\n")

        # Extract IDs with "Number:" context
        ids_with_context = extract_ids_with_context(translated)
        print(f"IDs with Number context: {ids_with_context}\n")
        print("-" * 80)


def load_document_data(path):
    try:
        with open(path, "r", encoding="utf-8") as file:
            # Parse the JSON file directly into a Python dictionary
            data_dict = json.load(file)
            return data_dict
    except FileNotFoundError:
        print(f"Error: The file at {path} was not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: The file at {path} is not valid JSON.")
        return None


def test_3():
    # Example 1: Number pattern
    unit = LawUnit(
        type_unit="Điều",
        id="Điều 2",
        content="Theo quy định tại Điều 1 và Điều 3, mức lãi suất...",
    )
    unit.get_cross_references()
    # Returns: ("Điều 1", "Điều 3")

    # Example 2: Letter pattern
    unit = LawUnit(
        type_unit="Điểm",
        id="Điểm a",
        content="Căn cứ điểm b và điểm c của khoản này...",
    )
    unit.get_cross_references()
    # Returns: ("điểm b", "điểm c")

    # Example 3: Article with number
    unit = LawUnit(
        type_unit="Article",
        id="Article 5",
        content="As stated in Article 1 and Article 10...",
    )
    unit.get_cross_references()


# Returns: ("Article 1", "Article 10")
if __name__ == "__main__":
    # try:
    #     test_2()
    #     print("\n[OK] Test completed successfully")
    #     exit(0)
    # except Exception as e:
    #     print(f"Error in test_2: {e}")
    #     import traceback

    #     traceback.print_exc()
    #     exit(1)
    test_3()
