print("starting")
from FlagEmbedding import BGEM3FlagModel


EMBED_MODEL = BGEM3FlagModel("BAAI/bge-m3")

target = "What is the specific term length or duration of the deposit account as stated in the document?"
term = "Term is 5 months"

print(EMBED_MODEL.encode(target)["dense_vecs"] @ EMBED_MODEL.encode(term)["dense_vecs"])
