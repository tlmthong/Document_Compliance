from __future__ import annotations
import json
from dataclasses import dataclass, field
from uuid import uuid4
from enum import Enum
from typing import Optional, Set, List, Any, Dict
import os
from collections import deque
from FlagEmbedding import BGEM3FlagModel
from openai import OpenAI


MODEL_ID = "gpt-oss:120b-cloud"
EMBED_MODEL = BGEM3FlagModel("BAAI/bge-m3")
BASE_URL = "http://localhost:11434/v1"
LLM = OpenAI(
    base_url=BASE_URL,
    api_key="eb8c507c1e284d8f9bf5867f42b6cd8b.saBEZsoabpJuz3J6pIwAW1uk",
)


@dataclass
class Constraint(str, Enum):
    PROHIBITED = "STRICT"
    OBLIGTATION = "OBLIGATION"
    PERMISSION = "PERMISSION"
    EXCEPTION = "EXCEPTION"


@dataclass(frozen=True)
class PolicyUnit:
    id: str
    subject: str
    constraint: tuple = field(default_factory=tuple)
    context: str = ""
    references: tuple = field(default_factory=tuple)


def _make_unique_synthetic_id(used: Set[str], prefix: str = "SYN") -> str:
    # Generates: SYN001, SYN002, ...
    n = 1
    while True:
        candidate = f"{prefix}{n:03d}"
        if candidate not in used:
            used.add(candidate)
            return candidate
        n += 1


def _ensure_list_of_str(value: Any, field_name: str) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(x, str) for x in value):
        return value
    raise ValueError(
        f"Field '{field_name}' must be a string or list[str], got {type(value).__name__}"
    )


def _ensure_list_of_constraint(value: Any, field_name: str) -> List[Constraint]:
    """Convert constraint values to Constraint enum."""
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if isinstance(value, list):
        constraints = []
        for v in value:
            if isinstance(v, str):
                # Try to match string to Constraint enum
                try:
                    constraints.append(Constraint(v))
                except ValueError:
                    # If not a valid enum value, try matching by name
                    try:
                        constraints.append(Constraint[v.upper()])
                    except KeyError:
                        # Default to storing as-is if not valid enum
                        constraints.append(v)
            else:
                constraints.append(v)
        return constraints
    raise ValueError(
        f"Field '{field_name}' must be a string or list[str], got {type(value).__name__}"
    )


class ComplianceUnit:
    def __init__(self, file_path, llm=LLM, embed_model=EMBED_MODEL):
        self.llm = llm
        self.embed_model = embed_model
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                self.data = json.load(file)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Invalid JSON format: {e}", e.doc, e.pos)

        self.policy = self.process_policy()
        self.embedding = dict()
        for p in self.policy:
            self.prompt_HyDE(embedding_dict=self.embedding, policy_unit=p)

    def llm_generate(self, prompt: str, model: str = MODEL_ID) -> str:
        resp = self.llm.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You write a hypothetical passage for retrieval (HyDE).",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()

    def embed_text(self, text: str) -> list[float]:
        out = self.embed_model.encode(
            [text], return_dense=True, return_sparse=False, return_colbert_vecs=False
        )
        # out["dense_vecs"] is shape (1, dim)
        return out["dense_vecs"]

    # [0].tolist()

    def prompt_HyDE(
        self,
        embedding_dict,
        policy_unit,
        prompt_hyde_path=r"prompts/prompt_HyDE.txt",
        model=MODEL_ID,
    ):
        if not os.path.exists(prompt_hyde_path):
            raise FileNotFoundError(f"File not found: {prompt_hyde_path}")
        try:
            with open(prompt_hyde_path, "r", encoding="utf-8") as file:
                prefix = file.read()
        except:
            print("ERROR DISMISSING")
            pass
        full_context = (
            f"subject: {getattr(policy_unit, 'subject', '')}\n"
            f"context: {getattr(policy_unit, 'context', '')}\n"
            f"constraint: {getattr(policy_unit, 'constraint', [])}\n"
            f"references: {getattr(policy_unit, 'references', [])}\n"
        )
        print(full_context)
        full_prompt = prefix + full_context
        hyde_text = self.llm_generate(full_prompt, model=model)
        print(f"context: {policy_unit.context}")
        print(f"Hyde Tex: {hyde_text}\n")

        context_vec = self.embed_text(policy_unit.context)
        hyde_vec = self.embed_text(hyde_text)
        embedding_dict[policy_unit] = (context_vec, hyde_text)

    def process_policy(self, synthetic_prefix: str = "SYN"):
        if self.data == None:
            raise ValueError("No policy")
        units = self.data.get("compliance_units")
        if not isinstance(units, list):
            raise ValueError("Expected 'compliance units' to be a list.")
        used_ids: Set[str] = set()
        for u in units:
            if isinstance(u, dict) and isinstance(u.get("id"), str) and u["id"].strip():
                used_ids.add(u["id"].strip())
        out: List[PolicyUnit] = []
        for i, u in enumerate(units):
            if not isinstance(u, dict):
                raise ValueError(f"Unit at index {i} must be a dict/object.")

            unit_id = u.get("id")
            if not isinstance(unit_id, str) or not unit_id.strip():
                unit_id = _make_unique_synthetic_id(used_ids, prefix=synthetic_prefix)
            else:
                unit_id = unit_id.strip()
                used_ids.add(unit_id)

            subject = u.get("subject")
            if not isinstance(subject, str) or not subject.strip():
                raise ValueError(f"Unit {unit_id}: missing/invalid 'subject'.")

            context = u.get("context")
            if not isinstance(context, str):
                raise ValueError(f"Unit {unit_id}: missing/invalid 'context'.")

            constraints = _ensure_list_of_constraint(u.get("constraint"), "constraint")
            references = _ensure_list_of_str(u.get("references"), "references")

            out.append(
                PolicyUnit(
                    id=unit_id,
                    subject=subject.strip(),
                    constraint=tuple(constraints),
                    context=context,
                    references=tuple(references),
                )
            )
        return out

    def build_reference_graph(self, units) -> Dict[str, List[str]]:
        # directed edges: unit.id -> referenced ids
        return {u.id: list(getattr(u, "references", []) or []) for u in units}

    def n_hop_policy_ids(
        self, start_id: str, max_hops: int = 2, *, include_start: bool = True
    ) -> Set[str]:
        """
        Follow PolicyUnit.references up to max_hops hops (default 2).
        Returns a set of reachable policy IDs.
        """
        if max_hops < 0:
            raise ValueError("max_hops must be >= 0")
        graph = self.build_reference_graph(self.policy)

        visited: Set[str] = set()
        if include_start:
            visited.add(start_id)

        q = deque([(start_id, 0)])  # (node_id, depth)

        while q:
            node, depth = q.popleft()
            if depth == max_hops:
                continue

            for nxt in graph.get(node, []):
                if nxt not in visited:
                    visited.add(nxt)
                    q.append((nxt, depth + 1))

        return visited

    def n_hop_policy_units(
        self, start_id: str, max_hops: int = 2, *, include_start: bool = True
    ):
        """
        Same as n_hop_policy_ids, but returns PolicyUnit objects (preserves only those found).
        """
        id_to_unit = {u.id: u for u in self.policy}
        ids = self.n_hop_policy_ids(
            start_id, max_hops=max_hops, include_start=include_start
        )
        return [id_to_unit[i] for i in ids if i in id_to_unit]


def testCompliance():
    PATH = r"json_snapshots/policy_test.json"
    c = ComplianceUnit(file_path=PATH)
    list_policy_ids = c.n_hop_policy_ids("A1")
    id_to_unit = {u.id: u for u in c.policy}
    for policy_id in list_policy_ids:
        if policy_id in id_to_unit:
            print(id_to_unit[policy_id].id)
    print(len(c.embedding))


if __name__ == "__main__":
    testCompliance()
