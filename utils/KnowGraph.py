from dataclasses import dataclass
from typing import List, Dict, Tuple
import os
from pathlib import Path
from utils.Segmentation import *

# from FlagEmbedding import BGEM3FlagModel
import json
from openai import OpenAI

MODEL_ID = "gpt-oss:120b-cloud"
BASE_URL = "http://localhost:11434/v1"
LLM = OpenAI(
    base_url=BASE_URL,
    api_key="eb8c507c1e284d8f9bf5867f42b6cd8b.saBEZsoabpJuz3J6pIwAW1uk",
)
import os


@dataclass
class Entity:
    name: str


@dataclass
class Relation:
    relation: str


def write_to_file(filename, content, mode="w"):
    # Validate mode
    if mode not in ("w", "a"):
        raise ValueError("Mode must be 'w' (write) or 'a' (append).")

    try:
        # Open file with UTF-8 encoding
        with open(filename, mode, encoding="utf-8") as file:
            if isinstance(content, list):
                # Ensure each item ends with a newline
                file.writelines(
                    line if line.endswith("\n") else line + "\n" for line in content
                )
            elif isinstance(content, str):
                file.write(content)
            else:
                raise TypeError("Content must be a string or a list of strings.")
        print(f"Content successfully written to '{filename}' in mode '{mode}'.")

    except (OSError, IOError) as e:
        print(f"Error writing to file: {e}")


class Knowlegde:
    def __init__(self, llm=LLM):
        self.llm = llm
        self.graph: Dict[str, List[Tuple[Relation, Entity]]] = {}  # For tracability
        self.knowledge: List[Tuple[Entity, Relation, Entity]] = []

    def add_connection_str(self, s, r, o):
        self.add_connection(Entity(s), Relation(r), Entity(o))

    def llm_generate(self, prompt: str, model: str = MODEL_ID) -> str:
        resp = self.llm.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You write a json file for enity relation",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()

    def get_all(self):
        result = ""
        for s, r, o in self.knowlegde:
            result = "-" + s.name + " " + r.relation + " " + o.name + "\n"

    def full_pipeline_chunks_to_knowledge(
        self,
        docs,
        prefix_path=r"prompts/prompt_relation.txt",
        json_path=r"json_snapshots/context_snap.json",
        skip_length=30,
    ):
        print(docs)
        for doc in docs:
            if len(doc) < skip_length:
                continue
            self.get_relation_json_single(
                doc=doc, prefix_path=prefix_path, out_path=json_path
            )
            self.process_json_to_knowledge(json_path=json_path)

    def get_relation_json_single(
        self,
        doc,
        prefix_path=r"prompts/prompt_relation.txt",
        out_path=r"json_snapshots/context_snap.json",
    ):
        if not os.path.exists(prefix_path):
            raise FileNotFoundError(f"File not found: {prefix_path}")
        try:
            with open(prefix_path, "r", encoding="utf-8") as file:
                prefix = file.read()
        except:
            print("ERROR DISMISSING")
            pass
        full_prompt = prefix + "\n" + doc
        print(f"prompting: \n {full_prompt}")
        json_content = self.llm_generate(prompt=full_prompt)
        write_to_file(filename=out_path, mode="w", content=json_content)

    def process_json_to_knowledge(self, json_path=r"json_snapshots/context_snap.json"):
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
        except FileNotFoundError:
            print("File not found!")
        except json.JSONDecodeError:
            print("Invalid JSON format!")

        for item in data.get("extracted_data", []):
            rel = item.get("relation", {})
            self.add_connection_str(
                rel.get("subject", "Unknown"),
                rel.get("relationship", "Unknown"),
                rel.get("object", "Unknown"),
            )

    def add_connection(self, ent1, rel, ent2):
        self.knowledge.append((ent1, rel, ent2))
        if ent1.name not in self.graph:
            self.graph[ent1.name] = []
        self.graph[ent1.name].append((rel, ent2))

    def get_relations(self) -> str:
        """
        Output all relations as human-readable statements.
        Format: Entity1 relation Entity2
        Example: Alice knows Bob. Bob works at Google.
        """
        if not self.knowledge:
            return "No relations found."

        relations = []
        for ent1, rel, ent2 in self.knowledge:
            statement = f"{ent1.name} {rel.relation} {ent2.name}"
            relations.append(statement)

        return "\n".join(relations)

    def get_chain(self, start_name: str) -> str:
        # 1. Create a list to hold the lines of text
        output_lines = [start_name]

        # 2. Pass this list to the recursive function
        self._dfs(start_name, visited=set(), lines=output_lines)

        # 3. Join everything into a single string and return it
        return "\n".join(output_lines)

    def get_relation_set(self, start_name: str) -> List[str]:
        """
        Get all relations reachable from a start node as a set of string relations.
        Format: Entity1 relation Entity2
        Example: Alice knows Bob. Bob works at Google.
        Returns a list of relation strings.
        """
        relations = []
        visited = set()
        self._collect_relations(start_name, visited, relations)
        return relations

    def _collect_relations(
        self, current_name: str, visited: set, relations: list, max_depth=5, level=0
    ):
        """Recursively collect all relations from a starting node."""
        if current_name in visited or level >= max_depth:
            return
        visited.add(current_name)

        # Check for connections from current node
        if current_name in self.graph:
            for rel, entity in self.graph[current_name]:
                # Add the relation as a string statement
                statement = f"{current_name} {rel.relation} {entity.name}"
                relations.append(statement)

                # Recursively collect from the connected entity
                self._collect_relations(
                    entity.name, visited, relations, max_depth, level + 1
                )

    def _dfs(self, current_name: str, visited: set, lines: list, level=0, max_depth=5):
        if current_name in visited:
            return
        visited.add(current_name)

        # Check for connections
        if current_name in self.graph:
            for rel, entity in self.graph[current_name]:
                # Indentation logic from your image
                if level >= max_depth:
                    continue
                indent = "   " * level

                # Append the formatted string to the list instead of printing
                lines.append(f"{indent}└── [{rel.relation}] ──> {entity.name}")

                # Recursion
                self._dfs(entity.name, visited, lines, level + 1)


class ContextConverter:
    def __init__(self, doc):
        self.doc = doc
        self.segmented_doc = []
        self.segmented_doc = self.segmenting(doc)

    def segmenting(self, doc=None) -> List[str]:
        """Segment document into paragraphs only."""
        if doc is None:
            doc = self.doc

        if len(doc) == 0:
            return []

        # Split by paragraphs (double newlines)
        paragraphs = doc.split("\n\n")

        # Clean up whitespace and filter empty paragraphs
        segments = [para.strip() for para in paragraphs if para.strip()]

        return segments


# TEST
def test_json():
    graph = Knowlegde()
    graph.process_json_to_knowledge(json_path=r"json_snapshots/context_snap.json")

    for rel in graph.knowledge:
        print(f"- {rel}")
    print(graph.get_chain("Developer"))


def test_context_converter():
    """Test ContextConverter segmentation and grouping"""
    # Sample document with multiple paragraphs and sentences
    sample_doc = """
    The rain stopped as suddenly as it began, leaving the street glossy like a mirror that couldn’t decide what to reflect. Mina stepped over a shallow puddle and noticed a paper boat drifting toward the gutter, its edges softening but its little fold-lines still stubbornly sharp. She picked it up, unfolded it carefully, and found a message written in hurried pencil: “If you’re reading this, follow the smell of oranges.” Somewhere down the block, a warm citrus scent floated through the cool air—impossible in this weather—so she tucked the damp paper into her pocket and started walking, feeling like the city had just blinked at her.
    """

    print("=" * 60)
    print("Test: ContextConverter")
    print("=" * 60)

    # Test 1: Basic segmentation
    print("\n1. Basic Segmentation (by sentences):")
    converter = ContextConverter(sample_doc)
    print(f"Total segments: {len(converter.segmented_doc)}")
    for i, segment in enumerate(converter.segmented_doc):
        print(f"  Segment {i+1}: {segment[:60]}...")

    # Test 2: Group similar segments
    print("\n2. Grouping Similar Segments (threshold=0.75):")
    grouped = converter.group_similar(similarity_threshold=0.75)
    print(f"Grouped segments: {len(grouped)}")
    for i, segment in enumerate(grouped):
        print(f"  Group {i+1}: {segment[:80]}...")

    # Test 3: Empty document
    print("\n3. Empty Document Test:")
    empty_converter = ContextConverter("")
    print(f"Empty doc segments: {empty_converter.segmented_doc}")

    # Test 4: Single segment
    print("\n4. Single Segment Test:")
    single_converter = ContextConverter("This is a single sentence.")
    print(f"Single segment: {single_converter.segmented_doc}")

    print("=" * 60)


def test_Knowledege():

    doc = """"
    WHEREAS, [Developer Legal Name] (“Developer”) has conceived, architected, developed, and continues to maintain a proprietary software platform currently marketed as “BioTrack” (the “Software”), together with its related components (including any mobile applications, web portals, APIs, dashboards, reporting tools, workflow engines, data models, and technical and user documentation) and any updates, upgrades, patches, and enhancements made available by Developer from time to time (collectively, “BioTrack”), which is designed to provide operational visibility and control over inventory movement across multi-site supply chains—particularly in environments requiring structured handling, traceability, reconciliation, and audit-ready reporting;

WHEREAS, Client is a logistics and warehousing company operating distribution and fulfillment facilities that handle, store, pick, pack, and ship products on behalf of third parties, including customers in regulated and quality-sensitive industries, and Client desires to deploy a centralized system to reduce inventory variance, improve warehouse throughput and accuracy, strengthen chain-of-custody and traceability practices, and standardize warehouse processes across multiple sites;

WHEREAS, Client desires to obtain a limited license (or subscription right, as applicable) to access and use BioTrack for Client’s internal business purposes solely in connection with Client’s warehouse operations and related logistics workflows within the North American region (the “Territory”), including (as applicable) inbound receiving, put-away, storage location management, lot/batch tracking, expiration management, cycle counts, returns handling, shipment verification, exception management, and operational reporting;

WHEREAS, the parties acknowledge that implementation of BioTrack may require configuration and professional services, and may include integration with Client’s existing systems and third-party solutions (such as ERP, WMS, TMS, EDI, barcode/RFID scanning systems, customer portals, or business intelligence tools), and the parties desire to define a clear framework governing (i) the scope and delivery of implementation, integration, training, and support services, (ii) service levels and responsibilities, (iii) data access and security expectations, and (iv) change control procedures for new requirements or expansions;

WHEREAS, the parties previously entered into a Non-Disclosure Agreement dated January 14, 2023 (the “NDA”) in order to exchange confidential and proprietary information during evaluation and business discussions, and the parties now desire to formalize their commercial relationship through this Master Services Agreement (this “Agreement”), which sets forth the terms and conditions under which Developer will provide BioTrack and any related services to Client, including commercial terms, confidentiality, intellectual property ownership, compliance with applicable laws, risk allocation, and other obligations of the parties;

NOW, THEREFORE, in consideration of the mutual promises and covenants contained herein, and other good and valuable consideration, the receipt and sufficiency of which are acknowledged, the parties agree as follows:
    """

    docs = [docs]
    # Initialize
    kg = Knowlegde()
    kg.get_relation_json_test(doc=doc)

    # 1. Define Entities
    alice = Entity("Alice")
    bob = Entity("Bob")
    google = Entity("Google")
    house = Entity("House")

    # 2. Build the connections
    # Link 1: Alice -> Knows -> Bob
    kg.add_connection(alice, Relation("knows"), bob)

    # Link 2: Bob -> Works At -> Google
    kg.add_connection(bob, Relation("works_at"), google)

    kg.add_connection(bob, Relation("live_in"), house)

    # 3. Visualize the chain
    print(kg.get_relation_set("Alice"))


def test_multi():
    seg = Segmentation()
    doc = """"
    WHEREAS, the Developer has designed and owns certain proprietary software known as 'BioTrack', which is used for tracking inventory in pharmaceutical supply chains; and WHEREAS, the Client is a logistics company that desires to license this software to manage its warehouse operations in the North American region; and WHEREAS, the parties previously entered into a Non-Disclosure Agreement dated January 14, 2023, and now wish to formalize their commercial relationship under the terms of this Master Services Agreement; NOW, THEREFORE, the parties agree as follows...
    """
    docs = seg.segment_whole_to_chunks(doc=doc, chunk_size=500, overlap=50)
    print(f"segmented into {len(docs)}")
    # Initialize
    kg = Knowlegde()
    kg.full_pipeline_chunks_to_knowledge(docs=docs)
    for rel in kg.knowledge:
        print(f"- {rel}")
    print(kg.get_chain("Developer"))


if __name__ == "__main__":
    # test_context_converter()
    # print("\n")
    # test_Knowledege()
    # test_json()
    test_multi()
