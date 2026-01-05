from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import PlainTextResponse
import uvicorn
from openai import OpenAI
import os
from pydantic import BaseModel, Field
import json_numpy
import json
from typing import List, Dict
from utils.PolicyGraph import *


class UserCreate(BaseModel):
    user_id: int
    username: str


class Payload(BaseModel):
    text: str


# LLM Configuration
# MODEL_ID = "gpt-oss:120b-cloud"
MODEL_ID = "gpt-oss:120b-cloud"
# MODEL_ID = "gemini-3-flash-preview:cloud"
BASE_URL = "http://localhost:11434/v1"
PROMPT_URL = "prompts/promp_fact_vi.txt"
PROMPT_JUDGE_URL = "prompts/prompt_judge_vi.txt"
API_KEY = "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"

llm = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
)
app = FastAPI()


def llm_generate(
    prompt: str,
    model: str = MODEL_ID,
    llm=llm,
    topic="You write a set of self-facts",
) -> str:
    try:
        resp = llm.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": topic},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating response: {str(e)}"


def load_prefix():
    """Load the prefix prompt from file."""
    if not os.path.exists(PROMPT_URL):
        return ""
    try:
        with open(PROMPT_URL, "r", encoding="utf-8") as file:
            return file.read()
    except Exception as e:
        print(f"Error loading prefix: {e}")
        return ""


import re
from typing import List, Tuple


def segmenting(
    text: str,
    identifier: str = "Điều",
    end_identifier: str = "\r\n\r\n\r\n\r\n",
) -> List[str]:
    """
    Segment text by identifier (e.g., 'Điều', 'Article').

    Rules:
    1. Only return segments that start with the identifier
    2. Stop processing if end_identifier is found (e.g., triple newlines)
    3. Skip back-references: if we're at Article N and encounter Article M where M <= previous max,
       skip it (it's a reference, not a new article)
    4. Articles must be in ascending order (Article 1 -> Article 2 -> Article 3, etc.)

    Args:
        text: The full document text
        identifier: The article identifier (e.g., "Điều", "Article")
        end_identifier: Stop processing when this is encountered (default: triple CRLF)

    Returns:
        List of segments, each starting with the identifier
    """
    if not text:
        return []

    # Stop at end_identifier (handle both CRLF and LF-only)
    if end_identifier:
        cut = text.find(end_identifier)
        # Also check for LF-only version if CRLF version not found
        if cut == -1 and "\r\n" in end_identifier:
            cut = text.find(end_identifier.replace("\r\n", "\n"))
        if cut != -1:
            text = text[:cut]

    id_esc = re.escape(identifier)

    # Match headers at the START of a line only
    # Pattern matches: "Điều 1.", "Điều 2a:", "Article 10 -", "Điều 3)", etc.
    # Must be at beginning of line (after optional whitespace)
    header_re = re.compile(
        rf"(?m)^\s*{id_esc}\s+(?P<num>\d+)\s*(?P<suffix>[A-Za-zÀ-ỹ]*)\s*[.:)\-]?"
    )

    matches = list(header_re.finditer(text))
    if not matches:
        return []

    def article_key(m) -> Tuple[int, str]:
        """Extract numeric key and suffix for comparison (e.g., Article 2a -> (2, 'a'))"""
        return (int(m.group("num")), (m.group("suffix") or "").lower())

    segments: List[str] = []
    last_kept_key: Tuple[int, str] = (0, "")  # Start at 0, so Article 1 is valid

    i = 0
    while i < len(matches):
        m = matches[i]
        current_key = article_key(m)

        # Skip if this article number is <= the last kept article number
        # This handles back-references like "theo Điều 1" when we're already at Điều 2
        if current_key <= last_kept_key:
            i += 1
            continue

        # Find the end of this segment (start of next valid article or end of text)
        start = m.start()

        # Look for the next article that would be valid (greater than current)
        end = len(text)
        for j in range(i + 1, len(matches)):
            next_key = article_key(matches[j])
            # Next valid article must have a higher number than current
            if next_key > current_key:
                end = matches[j].start()
                break

        # Extract and clean the segment
        chunk = text[start:end].strip()
        if chunk:
            segments.append(chunk)
            last_kept_key = current_key

        i += 1

    return segments


def extract_id_from_segment(
    segment: str, identifier: str = "Điều", rule: str = "num"
) -> str:
    """
    Extract the identifier and the next word from a text segment.

    Args:
        segment: The text segment to process
        identifier: The identifier to search for (e.g., "Điều", "Article", "Khoản")
        rule: Extraction rule
            - "num": Search for a number after the identifier (e.g., "Điều 1" -> "Điều 1")
            - "abc": Search for a single character after the identifier (e.g., "Điểm a" -> "Điểm a")

    Returns:
        A string containing the identifier and the extracted value (e.g., "Điều 1", "Khoản 2", "Điểm a")
        Returns empty string if not found.

    Examples:
        >>> extract_id_from_segment("Điều 1. Quy định về lãi suất", "Điều", "num")
        "Điều 1"
        >>> extract_id_from_segment("Điểm a) Mức lãi suất tối đa", "Điểm", "abc")
        "Điểm a"
        >>> extract_id_from_segment("Khoản 2. Các trường hợp ngoại lệ", "Khoản", "num")
        "Khoản 2"
    """
    if not segment or not identifier:
        return ""

    id_esc = re.escape(identifier)

    if rule == "num":
        # Search for identifier followed by a number
        # Matches: "Điều 1", "Article 10", "Khoản 2", etc.
        pattern = re.compile(rf"({id_esc})\s+(\d+)", re.IGNORECASE)
        match = pattern.search(segment)
        if match:
            return f"{match.group(1)} {match.group(2)}"

    elif rule == "abc":
        # Search for identifier followed by a single character (letter)
        # Matches: "Điểm a", "Point b", "Mục c", etc.
        pattern = re.compile(rf"({id_esc})\s+([A-Za-zÀ-ỹ])\b", re.IGNORECASE)
        match = pattern.search(segment)
        if match:
            return f"{match.group(1)} {match.group(2)}"

    return ""


def get_hyde():
    return


def get_fact(
    full_policy, source_entity="Nghị định này", prompt_path="prompts/promp_fact_vi.txt"
):
    """Full_policy is a concatenated segments, source_entity is the type of document.
    This function return a list of facts"""
    if not os.path.exists(prompt_path):
        return

    try:
        with open(prompt_path, "r", encoding="utf-8") as file:
            prefix = file.read()
    except:
        print("ERROR DISMISSING")
        return
    full_prompt = (
        prefix
        + "\n # Thực thể nguồn \n"
        + source_entity
        + "\n#Tài liệu xử lý\n"
        + full_policy
    )
    fact_json = llm_generate(full_prompt, model=MODEL_ID)
    print(fact_json)
    try:
        # 1. Parse the JSON string into a Python Dictionary
        data = json.loads(fact_json)

        # 2. Extract the list
        relation_list = data.get("extracted_data", [])
    except json.JSONDecodeError:
        raise KeyError("Error: LLM did not return valid JSON")
    list_fact = []
    for relation in relation_list:
        list_fact.append(
            source_entity
            + " "
            + relation["relation"]["relationship"]
            + " "
            + relation["relation"]["object"]
        )
    return list_fact


@app.post("/process_policy")
async def process_policy(
    policy_id: str,
    subject: str,
    segments: List[str],
    self_facts: List[str],
    identifier: str = "Điều",
    rule: str = "num",
):
    law_list = []
    full_policy = ""
    for segment in segments:
        full_policy + segment + "\r\n"
        law_list.append(
            LawUnit(
                type_unit=identifier,
                content=segment,
                id=extract_id_from_segment(segment=segment),
            )
        )
    policy = PolicyDocument(
        id=policy_id,
        subject=subject,
        full_content=full_policy,
        decisions=law_list,
        self_fact=self_facts,
    )
    return {
        "id": policy.id,
        # "Policy Subject": policy.subject,
        # "Policy Content": policy.full_content,
        "self_fact": policy.self_fact,
        "self_fact_embed": policy.self_fact_embed.tolist(),
        "decisions": [
            {
                "id": law.id,
                "content": law.content,
                "references": law.references,
                "hypothetical_questions": law.hypothetical_questions,
                "cross_references": law.cross_references,
                "hypothetical_embed": law.hypothetical_embed.tolist(),
            }
            for law in policy.decisions
        ],
    }


class LawModel(BaseModel):
    id: str
    content: str
    references: List = []
    cross_references: List = []
    hypothetical_embed: List[List[float]]


class PolicyModel(BaseModel):
    id: str
    self_fact: List[str]
    self_fact_embed: List[List[float]]
    decisions: List[LawModel]


def max_embed(encode, list_encode):
    cross_product = [encode @ encode2 for encode2 in list_encode]
    return max(cross_product)


from collections import defaultdict


@app.post("/judge")
async def judge(policy: PolicyModel, customer_info: Dict[str, Any]):
    # TODO: augment law unit with cross-reference unit
    # TODO : Rebounce if JSON is not valid
    self_facts = policy.self_fact
    self_fact_embed = [
        np.array(vect, dtype=np.float32) for vect in policy.self_fact_embed
    ]
    concat_fact = ""
    for fact in self_facts:
        concat_fact += fact + "\n"
    # CANIDATE RETRIVAL
    facts = ""
    universal_info = []
    canidate = defaultdict(list)
    for info in customer_info.keys():
        info = info + " is " + customer_info[info]
        encode_info = EMBED_MODEL.encode(info)["dense_vecs"]

        for fact in self_fact_embed:
            f_k = fact @ encode_info
            if f_k > 0.5:
                universal_info.append(info)
                print(f"Hit {info} Universal with {f_k}")
                break

        for law in policy.decisions:
            law_embed = [np.array(hyp_embed) for hyp_embed in law.hypothetical_embed]
            if max_embed(encode=encode_info, list_encode=law_embed) > 0.55:
                print(f"Hit {info} with {law.id} ")
                canidate[law.content].append(info)

    for info in universal_info:
        for law in canidate.keys():
            if info not in canidate[law]:
                canidate[law].append(info)

    full_fact = ""
    for fact in self_facts:
        full_fact += fact + "\n"
    # JUDGE
    if not os.path.exists(PROMPT_JUDGE_URL):
        return

    try:
        with open(PROMPT_JUDGE_URL, "r", encoding="utf-8") as file:
            prefix = file.read()
    except:
        print("ERROR DISMISSING")
        return

    final_decision = "Pass"
    judge_list = []
    reason_list = []
    for law_content in canidate.keys():
        # Get full extract info
        judge_info = ""
        for info in canidate[law]:
            judge_info += "- " + info + "\n"
        full_prompt = (
            prefix
            + "\n"
            + judge_info
            + "\n ###Văn bản"
            + law_content
            + "\n###Thông tin tóm tắt về văn bản chính sách:"
            + full_fact
        )
        judgement = llm_generate(full_prompt, model=MODEL_ID)
        print("Judging...")
        try:
            # 1. Parse the JSON string into a Python Dictionary
            judge = json.loads(judgement)
            if judge["decision"] == "NOT_COMPLY":
                final_decision = "Not Comply"
                reason_list.append(judge["reason"])
            judge_list.append(judge["decision"])
        except json.JSONDecodeError:
            raise KeyError("Error: LLM did not return valid JSON")
    return {
        "Final Decision": final_decision,
        "Decisions": judge_list,
        "Reasons": reason_list,
    }


@app.post("/self_fact")
async def self_fact(full_policy, source_entity="Nghị định này"):
    return {"self_fact": get_fact(full_policy=full_policy, source_entity=source_entity)}


@app.post("/segment")
async def process_segment(
    identifier: str = "Điều",
    end_identifier: str = "\r\n\r\n\r\n\r\n",
    file: UploadFile = File(...),
):
    if not file.filename.lower().endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt files are allowed")

    try:
        # Read file content as text
        content_bytes = await file.read()
        content_str = content_bytes.decode("utf-8")  # Decode UTF-8 text

        # Optional: limit file size (e.g., 1MB)
        if len(content_bytes) > 1_000_000:
            raise HTTPException(status_code=413, detail="File too large (max 1MB)")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File is not valid UTF-8 text")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")
    segments = segmenting(
        content_str, identifier=identifier, end_identifier=end_identifier
    )
    return {
        "Number of segment": len(segments),
        "Segments": segments,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
#     text = """
# Điều 1. Mức lãi suất tối đa đối với tiền gửi bằng đồng Việt Nam của tổ chức (trừ tổ chức tín dụng, chi nhánh ngân hàng nước ngoài) và cá nhân tại tổ chức tín dụng, chi nhánh ngân hàng nước ngoài theo quy định tại Thông tư số 48/2024/TT-NHNN ngày 30 tháng 9 năm 2024 như sau:
# Điều 1. Điều 1.
# 1. Mức lãi suất tối đa áp dụng đối với tiền gửi không kỳ hạn và có kỳ hạn dưới 1 tháng là 0,5%/năm.

# 2. Mức lãi suất tối đa áp dụng đối với tiền gửi có kỳ hạn từ 1 tháng đến dưới 6 tháng là 4,75%/năm; riêng Quỹ tín dụng nhân dân và Tổ chức tài chính vi mô áp dụng mức lãi suất tối đa đối với tiền gửi có kỳ hạn từ 1 tháng đến dưới 6 tháng là 5,25%/năm.

# """
#     print(extract_id_from_segment(text), "H")
