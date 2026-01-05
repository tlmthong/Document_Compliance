from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import requests
from pathlib import Path
import json
import sqlite3
import sqlite_vec
import struct

BASE_DIR = Path(__file__).resolve().parent
judge_url = "http://127.0.0.1:8000/judge"

app = FastAPI()

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def blob_to_f32list(b: bytes, dim: int = 1024):
    return list(struct.unpack(f"{dim}f", b))


def f32blob(vec):
    return struct.pack(f"{len(vec)}f", *vec)  # len(vec) must be 1024


def feed_json(policy_id="Test3"):
    db = sqlite3.connect("data/policy.db")
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    rows = db.execute(
        f"""
        SELECT * FROM policy 
        """
    ).fetchall()
    print(rows)
    rows = db.execute(
        f"""
        SELECT * FROM sel_fact WHERE policy_id = "{policy_id}"
        """
    ).fetchall()
    print(rows)
    self_fact = [x[2] for x in rows]

    rows = db.execute(
        f"""
        SELECT embedding FROM sel_fact_vec sv JOIN sel_fact s on sv.rowid = s.id WHERE s.policy_id = "{policy_id}" 
        """
    )
    self_fact_embed = [blob_to_f32list(x[0]) for x in rows]

    rows = db.execute(
        f"""
        SELECT id FROM law_unit where policy_id = "{policy_id}" 
        """
    )
    print(rows)
    law_id_list = [x[0] for x in rows]
    decisions = []

    for law_id in law_id_list:
        print(law_id)
        rows = db.execute(
            f"""
            SELECT content FROM law_unit where policy_id = "{policy_id}" AND id = "{law_id}" 
            """
        )
        content = [x[0] for x in rows][0]  # SHOULD ONLY HAVE ONLY ONE CONTENT

        # get reference
        rows = db.execute(
            f"""
            SELECT reference from law_references WHERE law_unit_id = "{law_id}" AND policy_id = "{policy_id}"
            """
        )
        references = [x[0] for x in rows]

        # get cross-reference
        rows = db.execute(
            f"""
            SELECT cross_reference from law_cross_references WHERE law_unit_id = "{law_id}" AND policy_id = "{policy_id}"
            """
        )
        cross_references = [x[0] for x in rows]

        # get embed
        rows = db.execute(
            f"""    
            SELECT embedding from hiq_vec hv JOIN hiq h ON hv.rowid = h.id WHERE h.policy_id = "{policy_id}" AND h.law_unit_id = "{law_id}"
            """
        )
        embed = [blob_to_f32list(x[0]) for x in rows]

        decisions.append((law_id, content, references, cross_references, embed))
    return {
        "id": policy_id,
        "self_fact": self_fact,
        "self_fact_embed": self_fact_embed,
        "decisions": [
            {
                "id": decision[0],
                "content": decision[1],
                "references": decision[2],
                "cross_references": decision[3],
                "hypothetical_embed": decision[4],
            }
            for decision in decisions
        ],
    }


def save_json(dict_json: dict):
    from datetime import datetime

    form_dir = BASE_DIR / "form"
    form_dir.mkdir(parents=True, exist_ok=True)

    # Generate unique filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = form_dir / f"scanned_doc{timestamp}.json"

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(dict_json, f, ensure_ascii=False, indent=4)

    return file_path


def judge_llm(process_policy_json, documnet_url: str = "Documents/scanned_doc.json"):
    doc_path = BASE_DIR / documnet_url
    print("DOC PATH:", doc_path)
    print("EXISTS:", doc_path.exists())

    with open(doc_path, "r", encoding="utf-8") as f:
        document = json.load(f)
    request_json = {"policy": process_policy_json, "customer_info": document}
    r = requests.post(judge_url, json=request_json, timeout=270)
    r_json = r.json()
    return r_json


from typing import Any, Dict


@app.post("/judge")
async def judge(policy_id: str, user_json: Dict[str, Any]):

    j_json = feed_json(policy_id=policy_id)
    print(j_json)
    file_path = save_json(user_json)
    return judge_llm(process_policy_json=j_json, documnet_url=file_path)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=6969)
