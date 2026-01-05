from Segmentation import *
from Retrival import *
from KnowGraph import *
from ComplianceUnit import *
from Judge import *


def test_case():
    seg = Segmentation()
    doc = """
    The Client is a logistics company operating a network of warehouses, cross-docks, and distribution partners throughout the North American region, supporting a mix of time-sensitive and regulated product flows. As the Client’s volume and facility count increased, it encountered recurring challenges with fragmented inventory visibility, inconsistent receiving and put-away practices across sites, and delays caused by manual reconciliation between warehouse records and downstream operational reporting. The Client has determined that improving end-to-end inventory traceability and exception handling is necessary to maintain service levels, reduce shrink and write-offs, and strengthen audit readiness across its facilities.

The Developer has experience designing and implementing software solutions intended to support complex inventory environments, including structured movement tracking, configurable workflow controls, and reporting outputs aligned with operational governance requirements. In connection with the Client’s evaluation of potential systems, the parties engaged in preliminary discussions regarding a software platform referred to as “BioTrack,” including demonstrations, high-level technical discussions, and exchanges of operational and process information to assess whether the platform could be adapted to the Client’s multi-site warehouse needs.

Given that these discussions required the sharing of non-public operational details, facility processes, and commercial information, the parties entered into a Non-Disclosure Agreement dated January 14, 2023, to protect confidential information exchanged in the course of their evaluation and planning. After executing the Non-Disclosure Agreement, the Client provided additional detail regarding its warehouse workflows, internal reporting cadence, exception patterns, and integration constraints, and the parties discussed a phased approach for configuration, onboarding, training, and support.

The Client desires to obtain a license to use BioTrack in connection with its North American operations, with the intent of improving inventory tracking and warehouse execution across its facilities, including internal performance analytics and customer-facing shipment-visibility reporting generated from BioTrack outputs. The parties further wish to set forth the terms and conditions under which the Developer will provide implementation and related professional services, and under which the Client will be granted access to the software and associated documentation, as may be further described in statements of work executed under this Master Services Agreement.

The parties acknowledge their prior Non-Disclosure Agreement dated January 14, 2023, and intend that confidentiality protections continue to apply to applicable information shared under or in connection with this Master Services Agreement, including any technical, commercial, and operational information disclosed during implementation and ongoing support.

NOW, THEREFORE, in consideration of the mutual covenants and promises contained herein, and other good and valuable consideration, the receipt and sufficiency of which are hereby acknowledged, the parties agree as follows…
    """
    docs = seg.segment_whole_to_chunks(doc=doc, chunk_size=500, overlap=50)
    print(f"segmented into {len(docs)}")
    # Initialize
    kg = Knowlegde()
    kg.full_pipeline_chunks_to_knowledge(docs=docs)
    for rel in kg.knowledge:
        print(f"- {rel}")
    print(kg.get_chain("Developer"))

    context_chunks = seg.process_markdown(markdown_document=doc, chunk_size=250)
    print(f"Context chunk size: {len(context_chunks)}")
    retriever = Retriever()
    result = retriever.retrieve_knowledge_dict(context_chunks, kg)

    # print("=" * 50)
    # print("Test: Retriever Knowledge Extraction")
    # print("=" * 50)
    # print(f"\nSegment: {context_chunks}")
    # print(f"\nKnowledge Graph Nodes: {list(kg.graph.keys())}")
    # print(f"\nRetrieved Facts:\n{result}")
    # print("=" * 50)

    for r in result:
        print(f" Context:{r} \n")
        print(f"fact{result[r]}")
    POLICY_PATH = r"json_snapshots/policy_test.json"
    c = ComplianceUnit(file_path=POLICY_PATH, llm=LLM, embed_model=EMBED_MODEL)

    canidate = Canidate()
    dict = canidate.retrieve(doc_dict=result, cu=c)  # prompt to policy
    judge = Judgement(prompt_policy=dict)
    judge.judge_full()


if __name__ == "__main__":
    test_case()
