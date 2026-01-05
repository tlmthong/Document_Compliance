from FlagEmbedding import BGEM3FlagModel
from KnowGraph import *
from ComplianceUnit import *
from typing import List, Dict, Tuple
import numpy as np

# from sentence_transformers import SentenceTransformer
# from transformers import AutoTokenizer
# from transformers import AutoModelForSequenceClassification
# import torch
from sklearn.metrics.pairwise import cosine_similarity

EMBED_MODEL = BGEM3FlagModel("BAAI/bge-m3")


class Retriever:
    def __init__(self):
        pass

    def retrieve_knowledge_dict(self, list_seg_doc, k_graph):
        """
        map segmented docs to corresponding knowledge
        """
        knowledge_dict = dict()
        for seg_doc in list_seg_doc:
            knowledge_dict[seg_doc] = self.retrieve_knowledge(seg_doc, k_graph)
        return knowledge_dict

    def retrieve_knowledge(self, seg_doc, k_graph):
        set_seg = set(seg_doc.split())
        set_graph = set(k_graph.graph.keys())
        subjects = set_seg.intersection(set_graph)
        fact_set = set()
        fact = ""
        for s in subjects:
            print(k_graph.get_relation_set(s))
            fact_set = fact_set.union(k_graph.get_relation_set(s))
            print(fact_set)
        for f in fact_set:
            fact += f
            fact += ". "
        return fact


class Canidate:
    def __init__(self):
        pass

    def retrieve(
        self, doc_dict: Dict, cu: ComplianceUnit, threshold=0.7, model=EMBED_MODEL
    ):

        list_doct = list(doc_dict.keys())
        list_full_context = []
        for doct in doc_dict.keys():
            list_full_context.append(doct + "\n Fact: \n" + doc_dict[doct])

        # Embed documents
        embed_result = model.encode(
            list_doct, return_dense=True, return_sparse=False, return_colbert_vecs=False
        )
        embed_docs = embed_result["dense_vecs"]  # Extract the dense vectors

        canidate = dict()
        for idx_doc, embed_doc in enumerate(embed_docs):
            # Reshape to 2D array for cosine_similarity
            embed_doc_2d = np.array(embed_doc).reshape(1, -1)
            temp_dict = dict()

            for idx_policy, policy in enumerate(cu.embedding.keys()):
                embed_policy, hyde_text = cu.embedding[policy]
                # embed_policy is a vector, hyde_text is a string that needs to be embedded
                # Reshape policy embedding to 2D array
                embed_policy_2d = np.array(embed_policy).reshape(1, -1)

                # Embed the hyde_text if it's a string
                if isinstance(hyde_text, str):
                    hyde_embed_result = model.encode(
                        [hyde_text],
                        return_dense=True,
                        return_sparse=False,
                        return_colbert_vecs=False,
                    )
                    embed_hyde = hyde_embed_result["dense_vecs"][0]
                else:
                    embed_hyde = hyde_text

                embed_hyde_2d = np.array(embed_hyde).reshape(1, -1)
                ep = cosine_similarity(embed_doc_2d, embed_policy_2d)[0][0]
                print(ep)
                eh = cosine_similarity(embed_doc_2d, embed_hyde_2d)[0][0]
                print(eh)
                if ep > threshold or eh > threshold:
                    print("CANIDATE FOUND!")
                    full_context = list_full_context[idx_doc]
                    if full_context not in canidate:
                        canidate[full_context] = []
                    canidate[full_context].append(policy)
                temp_dict[max([ep, eh])] = policy
            full_context = list_full_context[idx_doc]
            if full_context not in canidate:
                canidate[full_context] = []
                canidate[full_context].append(temp_dict[max(temp_dict.keys())])

        return canidate  # this is a dict: context to policy object


def test_retriever():
    """Test the Retriever class with sample knowledge graph"""
    # 1. Create a knowledge graph
    kg = Knowlegde()

    # 2. Add some sample entities and relations
    alice = Entity("Alice")
    bob = Entity("Bob")
    google = Entity("Google")
    house = Entity("House")

    kg.add_connection(alice, Relation("knows"), bob)

    kg.add_connection(bob, Relation("works_at"), google)

    kg.add_connection(bob, Relation("lives_in"), house)

    # 3. Create a sample document segment
    seg_doc = """"Alice knows Bob well—well enough to notice when he’s stressed and when he’s proud of himself. She listens to his stories, especially about his work at Google, and she’s the kind of friend who shows up consistently, not just when it’s convenient.

Bob works at Google and lives in a house that feels like his quiet reset zone. His days are busy, but his life outside work is simple: home, rest, and keeping things steady when everything else moves fast.
    """

    # 4. Test the retriever
    retriever = Retriever()
    converter = ContextConverter(seg_doc)
    result = retriever.retrieve_knowledge_dict(converter.segmented_doc, kg)

    print("=" * 50)
    print("Test: Retriever Knowledge Extraction")
    print("=" * 50)
    print(f"\nSegment: {seg_doc}")
    print(f"\nKnowledge Graph Nodes: {list(kg.graph.keys())}")
    print(f"\nRetrieved Facts:\n{result}")
    print("=" * 50)


if __name__ == "__main__":
    test_retriever()
