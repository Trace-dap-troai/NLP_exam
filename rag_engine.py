# --------------------------------------------------
# BMW AI CEO Agent
# Knowledge Repository + Strategic Intelligence Engine
# --------------------------------------------------

import os
import re  # <--------->
import json
import chromadb
import pandas as pd
import ollama
from sentence_transformers import SentenceTransformer

CSV_FILE = "bmw_live_data.csv"
CACHE_FILE = "bmw_cached_report.json"
COLLECTION_NAME = "bmw_strategic_intel"


class AICEoAgent:

    def __init__(self):
        print("Initializing AI CEO Agent...")
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME
        )

    # --------------------------------------------------
    # INDEX DATA
    # --------------------------------------------------
    def index_data(self, csv_path=CSV_FILE):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"{csv_path} not found")

        print("Loading CSV data...")
        df = pd.read_csv(csv_path)

        documents = df["content"].fillna("").tolist()
        #ids = [f"doc_{i}" for i in range(len(df))]
        ids = df["id"].astype(str).tolist()
        print("Rows:", len(df))
        print("IDs:", ids[:20])
        print("Unique IDs:", len(set(ids)))
        print("Total IDs:", len(ids))
        metadatas = []

        for _, row in df.iterrows():
            metadatas.append(
                {
                    "source": str(row["source"]),
                    "url": str(row["link"]),
                    "sentiment": str(row["sentiment_label"]),
                }
            )

        print("Generating embeddings...")
        embeddings = self.embedding_model.encode(documents).tolist()

        if self.collection.count() > 0:
            self.client.delete_collection(name=COLLECTION_NAME)
            self.collection = self.client.create_collection(
                name=COLLECTION_NAME
            )

        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids,
        )
        print(f"Indexed {self.collection.count()} documents.")
    
    def ask_question(self, question):

        context, _ = self.retrieve_context(question)

        prompt = f"""
You are a BMW Strategic Intelligence Advisor.

Based only on the context below, answer the question.

CONTEXT:
{context}

QUESTION:
{question}

Provide a concise executive answer.
"""

        response = ollama.chat(
            model="gemma3:4b",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response["message"]["content"]

    # --------------------------------------------------
    # SEMANTIC SEARCH 
    # --------------------------------------------------
    def retrieve_context(self, query, n_results=7):
        query_embedding = self.embedding_model.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding, n_results=n_results
        )

        context = ""
        evidence = []

        docs = results["documents"][0]
        metas = results["metadatas"][0]

        for doc, meta in zip(docs, metas):
            source = meta.get("source", "Unknown")
            url = meta.get("url", "N/A")
            #snippet = doc[:600]

            #Head-Tail Chunking
            #First 750 characters (Contains Event/Hook) + Last 750 characters (Contains Conclusion/Action)
            if len(doc) > 1500:
                snippet = doc[:750] + "\n[... Content omitted ...]\n" + doc[-750:]
            else:
                snippet = doc

            context += f"[SOURCE={source}] [URL={url}] {snippet}\n\n"
            evidence.append({"source": source, "url": url})

        return context, evidence

    # --------------------------------------------------
    # BUILD PROMPT 
    # --------------------------------------------------
    def build_prompt(self, context):
        prompt = f"""
You are the Chief Strategic Intelligence Advisor for the CEO of BMW Group.
Based strictly on the extracted context below, provide a professional strategic analysis.

CONTEXT:
{context}

STRICT RULES:
1. Return ONLY a valid JSON object matching the schema below.
2. Do NOT output empty strings "" or empty lists []. You MUST synthesize facts from the context.
3. Overwrite the placeholder text inside the quotes with your actual detailed analysis.

EXACT JSON SCHEMA TO FILL:

{{
  "market_intelligence": {{
    "competitor_activities": [
      "Detail action taken by Mercedes, Tesla, or Audi based on context",
      "Detail another competitor move found in context"
    ],
    "emerging_technologies": [
      "Identify an emerging EV battery or software tech from context"
    ]
  }},
  "opportunities": [
    {{
      "title": "Name of the strategic opportunity",
      "impact": "High / Medium - Explain benefit to BMW Group",
      "confidence": "High",
      "evidence": "Quote specific data or event from context"
    }}
  ],
  "risks": [
    {{
      "title": "Name of the critical threat or risk",
      "category": "Supply Chain / Market / Tech",
      "severity": "Critical / High",
      "confidence": "High",
      "evidence": "Cite exact warning from context"
    }}
  ],
  "trends": [
    {{
      "title": "Key Industry Trend",
      "description": "Detailed explanation of how this trend alters the automotive landscape"
    }}
  ],
  "recommendations": [
    {{
      "action": "Specific management directive for BMW Board",
      "priority": "P1 (Urgent) / P2",
      "supporting_evidence": "Quote exact context linking to this action",
      "expected_impact": "What key metric improves",
      "risk_assessment": "Downside if this action fails"
    }}
  ],
  "ceo_briefing": {{
    "what_happened": "Executive summary of the live data retrieved.",
    "why_it_matters": "Why the CEO needs to act on this immediately.",
    "what_next": "Bulleted 3-step action plan for the Board of Management."
  }}
}}
"""
        return prompt

    # --------------------------------------------------
    # LLM REPORT (Regex)
    # --------------------------------------------------
    def generate_report(self):
        query = "BMW strategy, EV market, competitor activities, Mercedes, Audi, Tesla, technology trends, supply chain risks"
        context, evidence = self.retrieve_context(query)
        prompt = self.build_prompt(context)

        try:
            target_model = "gemma3:4b"
            print(
                f"Generating strategic report using {target_model} (Free-text mode)..."
            )

            response = ollama.chat(
                model=target_model,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": 0.1, 
                    "top_p": 0.2,
                    "num_ctx": 6144}
            )

            raw_content = response["message"]["content"]

            #regex 
            match = re.search(r"\{.*\}", raw_content, re.DOTALL)
            if match:
                clean_json_str = match.group(0)
            else:
                clean_json_str = raw_content

            # strict=False avoid enter, space
            try:
                report = json.loads(clean_json_str, strict=False)
            except Exception:
                sanitized_str = re.sub(
                    r"[\x00-\x1f\x7f-\x9f]", "", clean_json_str
                )
                report = json.loads(sanitized_str, strict=False)

            report["retrieved_sources"] = evidence

            with open(CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

            print("Report cached successfully!")
            return report

        except Exception as e:
            print(f"\n[!] Caught JSON parsing error: {e}")
            print("--- Please review ---")
            print(
                raw_content[:600] if "raw_content" in locals() else "No content"
            )

            if os.path.exists(CACHE_FILE):
                with open(CACHE_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            return None


# --------------------------------------------------
# MAIN
# --------------------------------------------------
if __name__ == "__main__":
    agent = AICEoAgent()
    agent.index_data()
    agent.generate_report()
    print("Knowledge Repository Ready.")