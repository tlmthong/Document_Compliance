import sqlite3
import sqlite_vec

db = sqlite3.connect("data/policy.db")
db.enable_load_extension(True)
sqlite_vec.load(db)
db.enable_load_extension(False)
db.execute("PRAGMA foreign_keys = ON;")
db.execute(
    """
CREATE TABLE policy (
  id   VARCHAR(100) PRIMARY KEY,
  subject TEXT NOT NULL,
  content TEXT NOT NULL
);
"""
)
db.execute(
    """
CREATE TABLE law_unit (
  id   VARCHAR(100) NOT NULL,
  policy_id VARCHAR(100) NOT NULL,
  content   TEXT NOT NULL,
  PRIMARY KEY (id, policy_id),
  FOREIGN KEY (policy_id) REFERENCES policy(id) ON DELETE CASCADE
);
"""
)
db.execute(
    """
CREATE TABLE hiq (
  id    INTEGER PRIMARY KEY,
  law_unit_id          VARCHAR(100) NOT NULL,
  policy_id VARCHAR(100) NOT NULL,
  hypothetical_question TEXT NOT NULL,
  FOREIGN KEY (law_unit_id) REFERENCES law_unit(id) ON DELETE CASCADE,
  FOREIGN KEY (policy_id) REFERENCES law_unit(policy_id) ON DELETE CASCADE
);
"""
)


db.execute(
    """
CREATE TABLE law_references (
  law_unit_id          VARCHAR(100) NOT NULL,
  policy_id VARCHAR(100) NOT NULL,
  reference VARCHAR(100) NOT NULL,
  PRIMARY KEY (law_unit_id, policy_id, reference)
  FOREIGN KEY (law_unit_id) REFERENCES law_unit(id) ON DELETE CASCADE,
  FOREIGN KEY (policy_id) REFERENCES law_unit(policy_id) ON DELETE CASCADE
);
"""
)

db.execute(
    """
CREATE TABLE law_cross_references (
  law_unit_id VARCHAR(100) NOT NULL,
  policy_id VARCHAR(100) NOT NULL,
  cross_reference VARCHAR(100) NOT NULL,
  PRIMARY KEY (law_unit_id, policy_id, cross_reference)
  FOREIGN KEY (law_unit_id) REFERENCES law_unit(id) ON DELETE CASCADE,
  FOREIGN KEY (policy_id) REFERENCES law_unit(policy_id) ON DELETE CASCADE
);
"""
)


db.execute(
    """
CREATE TABLE sel_fact (
  id        INTEGER PRIMARY KEY,
  policy_id VARCHAR(100) NOT NULL,
  fact      TEXT NOT NULL,
  FOREIGN KEY (policy_id) REFERENCES policy(id) ON DELETE CASCADE
);

"""
)


db.execute(
    """
-- ======================
-- Vector tables (sqlite-vec)
-- rowid == the id of the entity row
-- ======================

-- Embeddings for HIQ questions
CREATE VIRTUAL TABLE hiq_vec USING vec0(
  embedding float[1024] distance_metric=cosine
);


"""
)

db.execute(
    """
-- Embeddings for self facts
CREATE VIRTUAL TABLE sel_fact_vec USING vec0(
  embedding float[1024] distance_metric=cosine
);
"""
)
rows = db.execute(
    """
    SELECT * FROM POLICY
"""
).fetchall()

print(rows)
