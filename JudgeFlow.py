import os
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import requests
from pathlib import Path
import json
from db_config import get_connection, vector_to_list

BASE_DIR = Path(__file__).resolve().parent
POLICY_API_HOST = os.environ.get("POLICY_API_HOST", "http://127.0.0.1:8000")
judge_url = f"{POLICY_API_HOST}/judge"

app = FastAPI()

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def feed_json(policy_id="Test3"):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("SELECT * FROM policy")
    rows = cur.fetchall()
    print("Policies:", rows)

    cur.execute("SELECT * FROM sel_fact WHERE policy_id = %s", (policy_id,))
    rows = cur.fetchall()
    print("Self facts:", rows)
    # Column order: id, policy_id, fact, embedding
    self_fact = [x[2] for x in rows]

    cur.execute(
        """
        SELECT embedding FROM sel_fact 
        WHERE policy_id = %s AND embedding IS NOT NULL
    """,
        (policy_id,),
    )
    rows = cur.fetchall()
    self_fact_embed = [vector_to_list(x[0]) for x in rows]

    cur.execute("SELECT id FROM law_unit WHERE policy_id = %s", (policy_id,))
    rows = cur.fetchall()
    print("Law units:", rows)
    law_id_list = [x[0] for x in rows]
    decisions = []

    for law_id in law_id_list:
        print("Processing law_id:", law_id)

        cur.execute(
            """
            SELECT content FROM law_unit 
            WHERE policy_id = %s AND id = %s
        """,
            (policy_id, law_id),
        )
        rows = cur.fetchall()
        content = rows[0][0]  # SHOULD ONLY HAVE ONLY ONE CONTENT

        # get reference
        cur.execute(
            """
            SELECT reference FROM law_references 
            WHERE law_unit_id = %s AND policy_id = %s
        """,
            (law_id, policy_id),
        )
        references = [x[0] for x in cur.fetchall()]

        # get cross-reference
        cur.execute(
            """
            SELECT cross_reference FROM law_cross_references 
            WHERE law_unit_id = %s AND policy_id = %s
        """,
            (law_id, policy_id),
        )
        cross_references = [x[0] for x in cur.fetchall()]

        # get embed
        cur.execute(
            """
            SELECT embedding FROM hiq 
            WHERE policy_id = %s AND law_unit_id = %s AND embedding IS NOT NULL
        """,
            (policy_id, law_id),
        )
        embed = [vector_to_list(x[0]) for x in cur.fetchall()]

        decisions.append((law_id, content, references, cross_references, embed))

    cur.close()
    conn.close()

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

    uvicorn.run(app, host="0.0.0.0", port=6969)
