# --------------------------------------------------
# BMW AI CEO Agent
# Knowledge Repository + Strategic Intelligence Engine
# Senior Architecture: 7-Step Workflow + Deterministic Guardrail
# --------------------------------------------------

import os
import re
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
        if self.collection.count() > 0:
            print(f"Collection already has {self.collection.count()} documents. Skipping re-index.")
            return
        print("Loading CSV data...")
        df = pd.read_csv(csv_path)

        documents = df["content"].fillna("").tolist()
        ids = df["id"].astype(str).tolist()
        print("Rows:", len(df))
        print("IDs:", ids[:7])
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
            messages=[{"role": "user", "content": prompt}]
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

        if not results["documents"]:
            return context, evidence

        docs = results["documents"][0]

        metas = results["metadatas"][0]

        for doc, meta in zip(docs, metas):
            source = meta.get("source", "Unknown")
            url = meta.get("url", "N/A")

            context += f"[SOURCE={source}] {doc}\n\n"
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
      "confidence": "High / Medium / Low",
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

    # =====================================================================
    # AGENT CAPABILITIES MODULES (4 Autonomous Modules)
    # =====================================================================

    def plan_task(self, goal):
        print(f"\n[AGENT PLANNER] Receiving executive directive: '{goal}'...")
        prompt = f"""
        You are the Chief Strategy Planner for BMW Group.
        Based on the CEO goal: "{goal}"
        Generate EXACTLY 3 English keywords to search in our automotive news database.
        Return ONLY 3 keywords separated by commas. Do not write anything else.
        Example: BMW EV sales, Mercedes battery tech, supply chain disruption
        """
        try:
            res = ollama.chat(
                model="gemma3:4b",
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.1}
            )
            raw_text = res["message"]["content"]
            kw_list = [k.strip() for k in raw_text.replace('\n', ',').split(',') if len(k.strip()) > 3][:3]
            if not kw_list:
                raise ValueError("Empty planner output")
        except Exception as e:
            print(f" [!] Planner fallback triggered due to: {e}")
            kw_list = ["BMW EV strategy", "competitor market share", "supply chain risk"]

        return {
            "goal": goal,
            "strategy_keywords": kw_list,
            "agent_mode": "AUTONOMOUS_PLANNING"
        }

    def analyze_evidence(self, context):
        print("[AGENT ANALYZER] Scanning market signals from raw retrieved knowledge...")
        ctx_lower = context.lower()
        analysis = {
            "has_competitor_threats": any(x in ctx_lower for x in ["tesla", "mercedes", "audi", "byd", "porsche"]),
            "has_tech_disruption": any(x in ctx_lower for x in ["solid-state", "battery", "ai", "autonomous", "software"]),
            "has_supply_risks": any(x in ctx_lower for x in ["shortage", "strike", "recall", "delay", "tariff", "tax"]),
            "context_volume": len(context)
        }
        return analysis

    def decide_next_action(self, analysis):
        print("[AGENT DECISION] Executing autonomous reasoning on strategic focus...")
        actions = []
        if analysis.get("has_competitor_threats"):
            actions.append("Deep-dive into competitor aggressive pricing & EV launches")
        if analysis.get("has_tech_disruption"):
            actions.append("Prioritize R&D response to emerging battery/AI tech")
        if analysis.get("has_supply_risks"):
            actions.append("Issue P1 Urgent Board Warning regarding supply chain/recalls")

        if not actions:
            actions.append("Maintain baseline monitoring of BMW Group market share")

        return actions

    def validate_report(self, draft_json_str, source_context):
        print("[AGENT VALIDATOR] Activating AI Auditor for pre-presentation cross-check...")

        intro_part = (
            "You are an Executive AI Auditor for BMW Group.\n"
            "Here is the GROUND-TRUTH RETRIEVED CONTEXT from our database:\n"
            + str(source_context) + "\n\n"
            "Here is the DRAFT JSON strategic report generated for the CEO:\n"
            + str(draft_json_str) + "\n\n"
        )

        rules_part = """CRITICAL AUDIT TASK:
1. Verify it is valid JSON.
2. Ensure ALL required keys exist: "market_intelligence", "opportunities", "risks", "trends", "recommendations", "ceo_briefing".
3. STRICT ANTI-HALLUCINATION RULE: Scan the ENTIRE document. Check EVERY "evidence" and "supporting_evidence" field. If any field quotes outlets not in the GROUND-TRUTH CONTEXT (like marketplace.org, bloomberg, reuters, tradingkey, investopedia), REPLACE IT with a real sentence copied from the GROUND-TRUTH CONTEXT above.
4. LOGICAL ALIGNMENT RULE: Ensure the evidence directly proves the title.
RETURN ONLY THE VALID JSON OBJECT. NO MARKDOWN."""

        val_prompt = intro_part + rules_part

        try:
            res = ollama.chat(
                model="gemma3:4b",
                messages=[{"role": "user", "content": val_prompt}],
                options={"temperature": 0.0, "top_p": 0.1, "num_ctx": 8192}
            )
            raw_val = res["message"]["content"]
            match = re.search(r"\{.*\}", raw_val, re.DOTALL)
            return match.group(0) if match else draft_json_str
        except Exception as e:
            print(f" [!] Validator skipped due to model timeout: {e}")
            return draft_json_str

    # --------------------------------------------------
    # LLM AGENT REPORT ENGINE (7-Step Pipeline)
    # --------------------------------------------------
    def generate_report(self):
        print("\n" + "="*63)
        print(" INITIALIZING BMW STRATEGIC INTELLIGENCE AGENT (7-STEP WORKFLOW)")
        print("="*60)

        # STEP 1: GOAL
        ceo_goal = "Formulate an urgent strategic intelligence report for the BMW Group Board of Management."
        print(f"[STEP 1 - GOAL] Assigned Directive: '{ceo_goal}'")

        # STEP 2: PLAN
        print("[STEP 2 - PLAN] Formulating execution strategy...")
        plan_data = self.plan_task(ceo_goal)
        keywords = plan_data["strategy_keywords"]
        print(f" -> Targeted Search Matrix: {keywords}")

        # STEP 3: RETRIEVE
        print("[STEP 3 - RETRIEVE] Triggering Vector DB Retrieval Tools...")
        aggregated_context = ""
        all_evidence = []
        for kw in keywords:
            ctx, ev = self.retrieve_context(kw, n_results=3)
            aggregated_context += f"\n--- Knowledge Domain: [{kw}] ---\n{ctx}\n"
            all_evidence.extend(ev)

        unique_evidence = [dict(t) for t in {tuple(d.items()) for d in all_evidence}]

        # STEP 4: ANALYZE
        print("[STEP 4 - ANALYZE] Parsing semantic evidence structures...")
        intel_analysis = self.analyze_evidence(aggregated_context)

        # STEP 5: DECIDE
        print("[STEP 5 - DECIDE] Synthesizing strategic reasoning paths...")
        strategic_directives = self.decide_next_action(intel_analysis)
        print(f" -> Selected Directives: {strategic_directives}")

        # STEP 6: RECOMMEND
        print("[STEP 6 - RECOMMEND] Compiling directives into Executive JSON Report...")

        base_prompt = self.build_prompt(aggregated_context)
        enhanced_prompt = base_prompt + f"\n\nAGENT STRATEGIC DIRECTIVES TO PRIORITIZE:\n" + str(strategic_directives)

        try:
            target_model = "gemma3:4b"
            response = ollama.chat(
                model=target_model,
                messages=[{"role": "user", "content": enhanced_prompt}],
                options={"temperature": 0.1, "top_p": 0.2, "num_ctx": 6144}
            )
            raw_content = response["message"]["content"]
            match = re.search(r"\{.*\}", raw_content, re.DOTALL)
            clean_draft = match.group(0) if match else raw_content

            # STEP 7: VALIDATE
            print("[STEP 7 - VALIDATE] Auditing output data integrity...")
            validated_json_str = self.validate_report(clean_draft, aggregated_context)

            try:
                report = json.loads(validated_json_str, strict=False)
            except Exception:
                print(" [!] Warning: Validator distorted syntax, restoring baseline Draft...")
                report = json.loads(clean_draft, strict=False)


            report["retrieved_sources"] = unique_evidence
            report["agent_metadata"] = {
                "workflow": "Goal->Plan->Retrieve->Analyze->Decide->Recommend->Validate->Guardrail",
                "planned_keywords": keywords,
                "autonomous_decisions": strategic_directives
            }

            with open(CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

            print("\n" + "="*63)
            print(" [SUCCESS] STRATEGIC REPORT GENERATED AND CACHED SUCCESSFULLY!")
            print("="*63 + "\n")
            return report

        except Exception as e:
            print(f"\n[!!!!] CRITICAL PIPELINE FAILURE: {e}")
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