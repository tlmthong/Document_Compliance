import os
import requests
from pathlib import Path
import json
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from db_config import get_connection, list_to_vector, vector_to_list


BASE_DIR = Path(__file__).resolve().parent
HOST = os.environ.get("POLICY_API_HOST", "http://127.0.0.1:8000")

segment_url = HOST + "/segment"
self_fact_url = HOST + "/self_fact"
process_policy_url = HOST + "/process_policy"
judge_url = HOST + "/judge"


def segment_document(file_path):
    with open(file_path, "rb") as f:
        files = {"file": ("input.txt", f, "text/plain")}
        r = requests.post(segment_url, files=files, timeout=60)
    r.raise_for_status()
    r_json = r.json()
    segments = r_json["Segments"]
    return segments


def process_self_fact(segments_list, source_entity="Nghị định này"):
    full_policy = ""
    for segment in segments_list:
        full_policy += segment + "\n"
    request_json = {"full_policy": full_policy, "source_entity": source_entity}
    r = requests.post(self_fact_url, params=request_json, timeout=180)
    r_json = r.json()
    return r_json["self_fact"]


def process_policy(segments, self_fact, policy_id: str = "Test", subject: str = "test"):
    params = {"policy_id": policy_id, "subject": subject}
    payload = {
        "segments": segments,
        "self_facts": self_fact,
    }
    r = requests.post(process_policy_url, params=params, json=payload, timeout=180)
    r_json = r.json()
    return r_json


def store_to_db(process_policy_json):
    p_json = process_policy_json
    policy_id = p_json["id"]
    self_fact = p_json["self_fact"]
    self_fact_embed = p_json["self_fact_embed"]
    laws = p_json["decisions"]

    conn = get_connection()
    cur = conn.cursor()

    try:
        # CHECK IF POLICY EXIST
        cur.execute("SELECT 1 FROM policy WHERE id = %s", (policy_id,))
        rows = cur.fetchall()
        if len(rows) > 0:
            print("skipping - policy already exists")
            return

        # Insert Policy
        cur.execute(
            """
            INSERT INTO policy (id, subject, content)
            VALUES (%s, %s, %s)
            """,
            (policy_id, "test", "test"),
        )

        # INSERT self_facts with embeddings
        for fact, embed in zip(self_fact, self_fact_embed):
            print("Fact: ", fact)
            cur.execute(
                """
                INSERT INTO sel_fact (policy_id, fact, embedding) 
                VALUES (%s, %s, %s)
                """,
                (policy_id, fact, list_to_vector(embed)),
            )

        # INSERT LAW
        for law in laws:
            law_id = law["id"]
            law_content = law["content"]
            law_reference = law["references"]
            law_cross_reference = law["cross_references"]
            law_hyp_embed = law["hypothetical_embed"]
            law_hyp_questions = law["hypothetical_questions"]

            # ADD LAW
            cur.execute(
                """
                INSERT INTO law_unit (id, policy_id, content) VALUES(%s, %s, %s)
                """,
                (law_id, policy_id, law_content),
            )

            # ADD cross_references
            for cr in law_cross_reference:
                cur.execute(
                    """
                    INSERT INTO law_cross_references (law_unit_id, policy_id, cross_reference) 
                    VALUES(%s, %s, %s)
                    """,
                    (law_id, policy_id, cr),
                )

            # ADD reference
            for r in law_reference:
                cur.execute(
                    """
                    INSERT INTO law_references (law_unit_id, policy_id, reference) 
                    VALUES(%s, %s, %s)
                    """,
                    (law_id, policy_id, r[0]),
                )

            # ADD hypothetical questions with embeddings
            for h, e in zip(law_hyp_questions, law_hyp_embed):
                cur.execute(
                    """
                    INSERT INTO hiq (law_unit_id, policy_id, hypothetical_question, embedding) 
                    VALUES (%s, %s, %s, %s)
                    """,
                    (law_id, policy_id, h, list_to_vector(e)),
                )

        conn.commit()
        return 1
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cur.close()
        conn.close()


def check_db(num):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("SELECT * FROM policy")
    rows = cur.fetchall()
    print("Policies:", rows)

    cur.execute("SELECT * FROM sel_fact")
    rows = cur.fetchall()
    print("Self facts:", rows)

    cur.execute(
        """
        SELECT fact FROM sel_fact sf 
        JOIN policy p ON sf.policy_id = p.id 
        WHERE p.id = 'TEST_HAAH'
    """
    )
    rows = cur.fetchall()
    print("Facts for TEST_HAAH:", rows)

    cur.execute(
        """
        SELECT l.id, l.content FROM law_unit l 
        JOIN policy p ON l.policy_id = p.id 
        WHERE p.id = 'TEST_HAAH'
    """
    )
    rows = cur.fetchall()
    print("Law units for TEST_HAAH:", rows)

    cur.execute("SELECT * FROM hiq")
    rows = cur.fetchall()
    print("HIQ:", rows)

    cur.execute(
        """
        SELECT l.id, h.hypothetical_question 
        FROM law_unit l 
        JOIN hiq h ON l.policy_id = h.policy_id AND l.id = h.law_unit_id
    """
    )
    rows = cur.fetchall()
    print("Law unit hypothetical questions:", rows)

    cur.execute(
        """
        SELECT h.embedding, h.hypothetical_question 
        FROM hiq h 
        WHERE h.embedding IS NOT NULL
    """
    )
    rows = cur.fetchall()
    print("HIQ with embeddings:", rows)

    # POLICY CHECK PROCESSED
    policy_check = "TEST_HAAH"

    cur.execute("SELECT 1 FROM policy WHERE id = %s", (policy_check,))
    rows = cur.fetchall()
    print("Policy exists:", len(rows) == 1)

    cur.execute(
        """
        SELECT fact FROM sel_fact s 
        JOIN policy p ON s.policy_id = p.id 
        WHERE p.id = %s
    """,
        (policy_check,),
    )
    rows = cur.fetchall()
    list_self_fact = [x[0] for x in rows]
    print("Self facts:", list_self_fact)

    # LIST EMBED
    cur.execute(
        """
        SELECT embedding FROM sel_fact 
        WHERE policy_id = %s AND embedding IS NOT NULL
    """,
        (policy_check,),
    )
    rows = cur.fetchall()
    list_embed_decode = [vector_to_list(embed[0]) for embed in rows]
    print("Number of embeddings:", len(list_embed_decode))

    cur.close()
    conn.close()


def judge(process_policy_json, documnet_url: str = "Documents/scanned_doc.json"):
    doc_path = BASE_DIR / documnet_url
    print("DOC PATH:", doc_path)
    print("EXISTS:", doc_path.exists())

    with open(doc_path, "r", encoding="utf-8") as f:
        document = json.load(f)
    request_json = {"policy": process_policy_json, "customer_info": document}
    r = requests.post(judge_url, json=request_json, timeout=270)
    r_json = r.json()
    return r_json


#
# def test_doc(documnet_url: str = "Documents/scanned_doc.json"):
#     doc_path = BASE_DIR / documnet_url
#     print("DOC PATH:", doc_path)
#     print("EXISTS:", doc_path.exists())
#     with open(doc_path, "rb") as f:
#         raw = f.read()
#     print("FIRST 30 BYTES:", raw[:30])
#     text = raw.decode("utf-8-sig")
#     return json.loads(text)

# test_doc()


async def save_tex_policy(file, policy_id):
    import_dir = BASE_DIR / "import"
    import_dir.mkdir(parents=True, exist_ok=True)

    # Start with the base filename
    base_name = policy_id
    file_path = import_dir / f"{base_name}.txt"

    # If file exists, find the next available number
    if file_path.exists():
        counter = 2
        while True:
            file_path = import_dir / f"{base_name}({counter}).txt"
            if not file_path.exists():
                break
            counter += 1

    # Write the file content
    if hasattr(file, "read"):
        # If file is a file-like object (e.g., UploadFile)
        content = await file.read()
        if isinstance(content, bytes):
            content = content.decode("utf-8")
    else:
        # If file is already a string
        content = file

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

    return file_path


app = FastAPI()

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/upload_policy")
async def process_policy_flow(policy_id, subject, file: UploadFile):

    await save_tex_policy(file=file, policy_id=policy_id)

    file_path = f"import/{policy_id}.txt"

    MAX_TRY = 3
    num_try = 0
    while num_try < MAX_TRY:
        try:
            segments = segment_document(file_path)
            num_try = 0
            break
        except:
            num_try += 1
            if num_try == MAX_TRY:
                raise EOFError("Number of try exceeded")

    while num_try < MAX_TRY:
        try:
            self_fact = process_self_fact(segments_list=segments)
            num_try = 0
            break
        except:
            num_try += 1
            if num_try == MAX_TRY:
                raise EOFError("Number of try exceeded")

    while num_try < MAX_TRY:
        try:
            policy_processed_json = process_policy(
                segments=segments,
                self_fact=self_fact,
                policy_id=policy_id,
                subject=subject,
            )
            num_try = 0
            break
        except:
            num_try += 1
            if num_try == MAX_TRY:
                raise EOFError("Number of try exceeded")

    while num_try < MAX_TRY:
        try:
            num = store_to_db(process_policy_json=policy_processed_json)
            num_try = 0
            break
        except:
            num_try += 1
            if num_try == MAX_TRY:
                raise EOFError("Number of try exceeded")

    while num_try < MAX_TRY:
        try:
            check_db(num)
            num_try = 0
            break
        except:
            num_try += 1
            if num_try == MAX_TRY:
                raise EOFError("Number of try exceeded")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=6868)
