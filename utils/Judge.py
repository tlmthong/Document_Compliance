from openai import OpenAI
from typing import Dict
import os
import json

MODEL_ID = "gpt-oss:120b-cloud"
BASE_URL = "http://localhost:11434/v1"
LLM = OpenAI(
    base_url=BASE_URL,
    api_key="eb8c507c1e284d8f9bf5867f42b6cd8b.saBEZsoabpJuz3J6pIwAW1uk",
)


class Judgement:
    # prompt policy dict(context+fact : [policies])
    def __init__(self, prompt_policy: Dict, llm=LLM):
        self.llm = llm
        self.prompt_policy = prompt_policy

    def llm_generate(self, prompt: str, model: str = MODEL_ID) -> str:
        resp = self.llm.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You decide if the context and fact comply with policy.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()

    def judge_test_whole(self, doc, json_path=r"json_snapshots/policy_test.json"):
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"File not found: {json_path}")
        try:
            with open(json_path, "r", encoding="utf-8") as file:
                json_text = file.read()
        except:
            print("ERROR DISMISSING")
            pass
        testprompt = """"
        You are an automated Compliance Engine. You will receive two inputs:
1. A "Reference Document" (Policy/Rules).
2. A "Input Batch" (A JSON list of items/contexts to evaluate).

### TASK
Iterate through every item in the "Input Batch". Compare the content of each item against the "Reference Document" to determine compliance.

### CLASSIFICATION CRITERIA
For each item, assign one of the following decisions based *strictly* on the Reference Document:
- **COMPLY**: The item explicitly follows the rules or permissions.
- **NOT_COMPLY**: The item violates a prohibition or fails a mandatory requirement.
- **NOT_RELATE**: The item is irrelevant to the domain of the Reference Document.

### OUTPUT FORMAT
You must return a single valid JSON Array containing objects representing the evaluation of the input list in the exact same order.
Do not include markdown formatting (like ```json).

Each object in the array must follow this schema:
{
  "id": "The ID from the input item (if present, otherwise use the index)",
  "decision": "COMPLY" | "NOT_COMPLY" | "NOT_RELATE",
  "reason": "Concise justification citing the Reference Document."
}
        """
        testprompt += f"\n POLICY JSON: \n {json_text} \n Text to analyze: \n {doc}"
        print(self.llm_generate(prompt=testprompt, model=MODEL_ID))

    def judge_single(
        self, prompt, policy_unit, prefix_path=r"prompts/prompt_judge.txt"
    ):
        if not os.path.exists(prefix_path):
            raise FileNotFoundError(f"File not found: {prefix_path}")
        try:
            with open(prefix_path, "r", encoding="utf-8") as file:
                prefix = file.read()
        except:
            print("ERROR DISMISSING")
            pass
        policy_context_text = (
            f"id: {getattr(policy_unit, 'id', '')}\n"
            f"subject: {getattr(policy_unit, 'subject', '')}\n"
            f"context: {getattr(policy_unit, 'context', '')}\n"
            f"constraint: {getattr(policy_unit, 'constraint', [])}\n"
            f"references: {getattr(policy_unit, 'references', [])}\n"
        )
        full_prompt = prefix + prompt + "\n Policy: \n" + policy_context_text
        return self.llm_generate(prompt=full_prompt, model=MODEL_ID)

    def judge_full(self):
        if self.prompt_policy is None:
            raise ValueError()
        print("NUMBER OF PROMPTS", len(self.prompt_policy))
        for message in self.prompt_policy:
            print(f"Message:{message} ")
            for policy in self.prompt_policy[message]:
                print(f"Policy id : {policy.id} \n")
                print(f"- {self.judge_single(prompt = message, policy_unit= policy)}\n")


def test():
    doc = """
    The Client is a logistics company operating a network of warehouses, cross-docks, and distribution partners throughout the North American region, supporting a mix of time-sensitive and regulated product flows. As the Client’s volume and facility count increased, it encountered recurring challenges with fragmented inventory visibility, inconsistent receiving and put-away practices across sites, and delays caused by manual reconciliation between warehouse records and downstream operational reporting. The Client has determined that improving end-to-end inventory traceability and exception handling is necessary to maintain service levels, reduce shrink and write-offs, and strengthen audit readiness across its facilities.

The Developer has experience designing and implementing software solutions intended to support complex inventory environments, including structured movement tracking, configurable workflow controls, and reporting outputs aligned with operational governance requirements. In connection with the Client’s evaluation of potential systems, the parties engaged in preliminary discussions regarding a software platform referred to as “BioTrack,” including demonstrations, high-level technical discussions, and exchanges of operational and process information to assess whether the platform could be adapted to the Client’s multi-site warehouse needs.

Given that these discussions required the sharing of non-public operational details, facility processes, and commercial information, the parties entered into a Non-Disclosure Agreement dated January 14, 2023, to protect confidential information exchanged in the course of their evaluation and planning. After executing the Non-Disclosure Agreement, the Client provided additional detail regarding its warehouse workflows, internal reporting cadence, exception patterns, and integration constraints, and the parties discussed a phased approach for configuration, onboarding, training, and support.

The Client desires to obtain a license to use BioTrack in connection with its North American operations, with the intent of improving inventory tracking and warehouse execution across its facilities, including internal performance analytics and customer-facing shipment-visibility reporting generated from BioTrack outputs. The parties further wish to set forth the terms and conditions under which the Developer will provide implementation and related professional services, and under which the Client will be granted access to the software and associated documentation, as may be further described in statements of work executed under this Master Services Agreement.

The parties acknowledge their prior Non-Disclosure Agreement dated January 14, 2023, and intend that confidentiality protections continue to apply to applicable information shared under or in connection with this Master Services Agreement, including any technical, commercial, and operational information disclosed during implementation and ongoing support.

NOW, THEREFORE, in consideration of the mutual covenants and promises contained herein, and other good and valuable consideration, the receipt and sufficiency of which are hereby acknowledged, the parties agree as follows…
    """
    judge = Judgement(prompt_policy=None)
    judge.judge_test_whole(doc=doc)


if __name__ == "__main__":
    test()
