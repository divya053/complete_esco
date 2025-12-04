# rfp_analyzer_routes.py
"""
RFP Analyzer routes for Flask app - Exact port from rfp_analyser_12_ollama_latest_version.py
Uses Ollama Llama3 with 23-point weighted scoring and 5-level decision matrix
"""

import os
import json
import tempfile
import re
from typing import List, Dict, Any
from flask import Blueprint, render_template, request, jsonify, flash, redirect, url_for, current_app
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
from MySQLdb.cursors import DictCursor
from datetime import datetime, timedelta
import torch
import pandas as pd
import fitz  # PyMuPDF
import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
from doctr.models import ocr_predictor
from doctr.io import DocumentFile
from docx import Document as DocxDocument
import ollama
from dotenv import load_dotenv

load_dotenv()

# Create blueprint
rfp_bp = Blueprint('rfp_analyzer', __name__, url_prefix='/rfp-analyzer')

# ========================== Config ==========================
COMPANIES = ["IKIO", "METCO", "SUNSPRINT"]
COMPANY_DB_DIR = "company_db"
UPLOAD_FOLDER = "uploads/rfp"
ALLOWED_EXTENSIONS = {'pdf', 'txt', 'docx', 'json'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(COMPANY_DB_DIR, exist_ok=True)

# Ollama Configuration
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))
LLM_NUM_CTX = int(os.getenv("LLM_NUM_CTX", "8192"))
LLM_INPUT_BUDGET = int(os.getenv("LLM_INPUT_BUDGET", "6000"))
LLM_OUTPUT_BUDGET = int(os.getenv("LLM_OUTPUT_BUDGET", "1200"))
EVAL_INPUT_BUDGET = int(os.getenv("EVAL_INPUT_BUDGET", "6000"))
EVAL_OUTPUT_BUDGET = int(os.getenv("EVAL_OUTPUT_BUDGET", "600"))

# Initialize Ollama client
ollama_client = ollama.Client(host=OLLAMA_HOST)

# Retrieval settings
TOP_K = 4
DEFAULT_BATCH_PAGES = 10
RERANK_CANDIDATES = 50

# ========================== Checklist (23) ==========================
CHECKLIST_23 = [
    ("Full solicitation received (RFP, RFQ, drawings, specs, addenda) in the bid document?", "C"),
    ("U.S. federal/state/local procurement requirements reviewed?", "C"),
    ("All bid instructions, deadlines, and addenda understood?", "C"),
    ("Any ambiguities or missing info required? Clarification with the contracting officer/owner required?", "NC"),
    ("What is the project state and is company registered and licensed for work in the project state?", "C"),
    ("Do company meet all mandatory minimum qualifications (experience, capabilities, safety record, etc.) as per the project?", "C"),
    ("Can company meet Diversity/Small Business goals (e.g., SBE, WBE, MBE) if mandatory?", "C"),
    ("Do project require security clearances (for federal/defense projects) and are these obtainable for the company?", "C"),
    ("Is any Environmental/Permitting risks (NEPA, state laws) are required and are these understood and manageable by the company?", "C"),
    ("Is Project scope is clear and achievable with our company's capabilities?", "C"),
    ("Is any Labor relations/Union requirements? If so, are these requirements understood and manageable?", "C"),
    ("Is Site investigation conducted or required or sufficient data (geotech, survey) are available in bid document? If not, what is required?", "NC"),
    ("What are Major technical, commercial, and execution risks required in the bid? Are they identified and manageable by the company?", "C"),
    ("Adequate time and internal resources are available for quality proposal preparation by the company?", "C"),
    ("We have a competitive advantage or unique value proposition for this bid from the company?", "NC"),
    ("Internal team and key partners (especially engineering) are available?", "C"),
    ("Critical vendors, equipment, and subcontractors are available and can comply with project terms?", "C"),
    ("Is any adherence to 'BABA/BAA' or other domestic content rules is required in the BID and is it possible for the company to fulfil this?", "C"),
    ("Is any Contractual Terms & Conditions (indemnity, liability, payment) are required in the BID? Are these acceptable by the company?", "C"),
    ("Is any Required bonds (bid, performance, payment) and insurances in the BID can be secured by the company?", "C"),
    ("Is Preliminary pricing is feasible and within a competitive range for a target profit margin for the company?", "C"),
    ("Is Project aligning with our company's current strategic goals and risk profile?", "C"),
    ("Is Management/Executive buy-in for bidding this project has been secured?", "C"),
]

# Scoring Functions
def score_item(eval_str: str, criticality: str):
    if eval_str == "Yes":
        return 2 if criticality == "C" else 1
    elif eval_str == "Partial":
        return 1 if criticality == "C" else 0.5
    else:
        return 0

MAX_SCORE = sum(2 if c == "C" else 1 for _, c in CHECKLIST_23)

def get_decision_category(compliance_percent: float) -> str:
    if compliance_percent >= 90:
        return "Strongly-Go"
    elif compliance_percent >= 80:
        return "Go"
    elif compliance_percent >= 70:
        return "Conditional Go"
    elif compliance_percent >= 60:
        return "Evaluate Again"
    else:
        return "No-Go"

def get_decision_color(decision: str) -> str:
    colors = {
        "Strongly-Go": "#28a745",
        "Go": "#5cb85c",
        "Conditional Go": "#ffc107",
        "Evaluate Again": "#ff9800",
        "No-Go": "#dc3545"
    }
    return colors.get(decision, "#6c757d")

# ========================== Token Utilities ==========================
def count_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, int(len(text) / 4))  # Rough approximation

def trim_text_to_token_limit(text: str, max_tokens: int) -> str:
    approx_chars = max_tokens * 4
    return text[:approx_chars] if len(text) > approx_chars else text

# ========================== OCR and Text Extraction ==========================
def extract_pages_with_pymupdf(file) -> List[str]:
    """Extract text per page using PyMuPDF"""
    pages = []
    try:
        with fitz.open(stream=file.read(), filetype="pdf") as doc:
            for page in doc:
                t = page.get_text("text")
                pages.append(t or "")
        file.seek(0)
    except Exception:
        pages = []
        file.seek(0)
    return pages

def extract_pages_with_doctr(pdf_bytes: bytes) -> List[str]:
    """OCR fallback using docTR"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ocr_predictor(pretrained=True).to(device)
    doc = DocumentFile.from_pdf(pdf_bytes)
    result = model(doc)
    return [page.render() for page in result.pages]

def extract_page_texts(file) -> List[str]:
    """Try PyMuPDF first, fallback to OCR if needed"""
    pages = extract_pages_with_pymupdf(file)
    
    # If most pages empty, use OCR
    if not pages or sum(1 for p in pages if p.strip()) < max(1, int(0.2 * len(pages))):
        file.seek(0)
        pdf_bytes = file.read()
        pages = extract_pages_with_doctr(pdf_bytes)
        file.seek(0)
    
    return pages

# ========================== Ollama JSON Helper ==========================
def ollama_json(prompt: str, out_tokens: int) -> dict:
    """Call Ollama and parse JSON response"""
    try:
        resp = ollama_client.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": LLM_TEMPERATURE,
                "num_ctx": LLM_NUM_CTX,
                "num_predict": int(min(out_tokens, LLM_OUTPUT_BUDGET)),
            },
            stream=False,
        )
        text = resp.get("message", {}).get("content", "")
    except Exception as e:
        return {"error": f"Ollama error: {e}"}
    
    # Try strict JSON first
    try:
        return json.loads(text)
    except Exception:
        # Try to extract JSON from text
        m = re.search(r'(\{.*\}|\[.*\])', text, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
        return {"raw": text.strip()}

# ========================== Batch Summarization ==========================
def summarize_batch_with_llama(batch_text: str, batch_idx: int) -> dict:
    """Summarize a batch of pages"""
    batch_text = trim_text_to_token_limit(batch_text, LLM_INPUT_BUDGET)
    
    prompt = f"""You are summarizing a group of RFP pages (batch #{batch_idx}).
Return ONLY JSON with:
{{
 "summary": "dense summary capturing scope, deliverables, milestones, locations, due dates, budget, requirements",
 "key_requirements": ["..."],
 "technical_requirements": ["..."],
 "bid_requirements": ["..."],
 "other_keywords": ["..."]
}}

TEXT:
{batch_text}
"""
    return ollama_json(prompt, LLM_OUTPUT_BUDGET)

def build_master_summary(batch_summaries: List[dict]) -> dict:
    """Build master summary from batch summaries"""
    merged: Dict[str, List[str]] = {
        "key_requirements": [],
        "technical_requirements": [],
        "bid_requirements": [],
        "other_keywords": [],
    }
    
    parts = []
    for i, b in enumerate(batch_summaries, start=1):
        parts.append(f"=== Batch {i} ===\n{b.get('summary','')}\n")
        for k in merged.keys():
            merged[k].extend(b.get(k, []))
    
    batches_text = trim_text_to_token_limit("\n".join(parts), LLM_INPUT_BUDGET)
    
    prompt = f"""Consolidate these RFP batch summaries into one master summary.
Return ONLY JSON:
{{
 "project_type": "...",
 "scope": "...",
 "primary_domain": "...",
 "complexity": "Low|Medium|High",
 "summary": "Comprehensive summary",
 "due_date": "...",
 "total_cost": "...",
 "key_requirements": [],
 "technical_requirements": [],
 "bid_requirements": []
}}

BATCH SUMMARIES:
{batches_text}
"""
    return ollama_json(prompt, LLM_OUTPUT_BUDGET)

# ========================== Scorer ==========================
class WeightedBidScorer23:
    def __init__(self, checklist):
        self.checklist = checklist
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.emb_model = SentenceTransformer("BAAI/bge-large-en-v1.5", device=device)
            # Cross-encoder reranker for higher-precision ranking
            self.reranker = CrossEncoder("BAAI/bge-reranker-large", device=device)
        except Exception:
            self.emb_model = SentenceTransformer("BAAI/bge-large-en-v1.5")
            # Fallback without cross-encoder if loading fails
            self.reranker = None
    
    def retrieve_relevant_context(self, company_texts: List[str], requirement: str) -> str:
        if not company_texts:
            return ""
        
        req_vec = self.emb_model.encode([requirement], convert_to_numpy=True)
        text_vecs = self.emb_model.encode(company_texts, convert_to_numpy=True)
        
        req_norm = np.linalg.norm(req_vec, axis=1, keepdims=True) + 1e-12
        txt_norms = np.linalg.norm(text_vecs, axis=1, keepdims=True) + 1e-12
        sims = (text_vecs @ req_vec.T)[:, 0] / (txt_norms[:, 0] * req_norm[0, 0])
        
        # First-stage retrieval: take broader candidate set
        cand_k = min(RERANK_CANDIDATES, len(company_texts))
        cand_idx = sims.argsort()[-cand_k:][::-1]
        cand_texts = [company_texts[i] for i in cand_idx]
        
        # Optional cross-encoder reranking
        if self.reranker is not None and cand_texts:
            pairs = [[requirement, t] for t in cand_texts]
            scores = self.reranker.predict(pairs)
            reranked = sorted(zip(cand_texts, scores), key=lambda x: x[1], reverse=True)
            top_texts = [t for t, _ in reranked[:TOP_K]]
        else:
            top_texts = cand_texts[:TOP_K]
        
        snippets = [trim_text_to_token_limit(t, 200) for t in top_texts]
        return "\n\n--- COMPANY CHUNK ---\n\n".join(snippets)
    
    def evaluate_item(self, requirement: str, criticality: str, rfp_summary: dict,
                      company_name: str, company_texts: List[str], check_rfp_only=False) -> dict:
        if check_rfp_only:
            context = trim_text_to_token_limit(rfp_summary.get("summary", ""), EVAL_INPUT_BUDGET)
            context_source = "RFP-only"
        else:
            rfp_ctx = trim_text_to_token_limit(rfp_summary.get("summary", ""), EVAL_INPUT_BUDGET // 2)
            company_ctx = self.retrieve_relevant_context(company_texts, requirement)
            context = rfp_ctx + "\n\nCompany Evidence:\n" + company_ctx
            context = trim_text_to_token_limit(context, EVAL_INPUT_BUDGET)
            context_source = "RFP+Company"
        
        prompt = f"""Evaluate requirement for company "{company_name}".

Requirement: {requirement} ({criticality})
Context:
{context}

Rules:
- Use ONLY the provided context.
- If insufficient evidence, evaluation="Unknown", comments="Unknown (not in documents)".
- Return ONLY JSON:
{{
 "requirement":"{requirement}",
 "criticality":"{criticality}",
 "evaluation":"Yes|Partial|No|Unknown",
 "comments":"short rationale"
}}
"""
        data = ollama_json(prompt, EVAL_OUTPUT_BUDGET)
        return {
            "requirement": data.get("requirement", requirement),
            "criticality": data.get("criticality", criticality),
            "evaluation": data.get("evaluation", "Unknown"),
            "comments": data.get("comments", "Unknown"),
            "score": score_item(data.get("evaluation", "Unknown"), criticality),
            "context_source": context_source
        }
    
    def score_company(self, rfp_summary: dict, company_name: str, company_texts: List[str]) -> List[Dict]:
        evaluations = []
        for idx, (req, crit) in enumerate(self.checklist):
            check_rfp_only = idx < 4  # First 4 use RFP-only
            evaluations.append(
                self.evaluate_item(req, crit, rfp_summary, company_name, company_texts, check_rfp_only)
            )
        return evaluations

# Initialize scorer
scorer = WeightedBidScorer23(CHECKLIST_23)

# ========================== Routes ==========================
@rfp_bp.route('/')
@login_required
def analyzer_home():
    """Main RFP Analyzer interface"""
    if not current_user.is_admin:
        flash("Access Denied - Admin only", "danger")
        return redirect(url_for('role_dashboard'))
    
    return render_template('rfp_analyzer_main.html')

@rfp_bp.route('/analyze', methods=['POST'])
@login_required
def analyze_rfp():
    """Process RFP and analyze with companies"""
    if not current_user.is_admin:
        return jsonify({'error': 'Access Denied'}), 403
    
    try:
        rfp_file = request.files.get('rfp_file')
        if not rfp_file:
            return jsonify({'error': 'RFP file required'}), 400
        
        # Check if company files were uploaded
        ikio_file = request.files.get('ikio_context')
        metco_file = request.files.get('metco_context')
        sunsprint_file = request.files.get('sunsprint_context')
        
        # Build company knowledge bases from uploaded files if provided
        if ikio_file or metco_file or sunsprint_file:
            if ikio_file:
                build_company_kb_from_file(ikio_file, "IKIO")
            if metco_file:
                build_company_kb_from_file(metco_file, "METCO")
            if sunsprint_file:
                build_company_kb_from_file(sunsprint_file, "SUNSPRINT")
        
        # Extract pages from RFP
        pages = extract_page_texts(rfp_file)
        
        if not pages or len(pages) == 0:
            return jsonify({'error': 'Could not extract text from PDF. Please ensure it is a valid PDF.'}), 400
        
        # Summarize in batches
        batches = [pages[i:i+DEFAULT_BATCH_PAGES] for i in range(0, len(pages), DEFAULT_BATCH_PAGES)]
        batch_summaries = []
        for i, grp in enumerate(batches, start=1):
            batch_text = "\n\n--- PAGE BREAK ---\n\n".join(grp)
            summary = summarize_batch_with_llama(batch_text, i)
            if summary and not summary.get('error'):
                batch_summaries.append(summary)
        
        if not batch_summaries:
            return jsonify({'error': 'Failed to summarize RFP. Please check Ollama is running.'}), 500
        
        # Build master summary
        rfp_summary = build_master_summary(batch_summaries)
        
        if not rfp_summary or rfp_summary.get('error'):
            return jsonify({'error': 'Failed to build master summary. Please check Ollama is running.'}), 500
        
        # Score companies
        results = {}
        totals = {}
        missing_companies = []
        
        for company in COMPANIES:
            db_path = os.path.join(COMPANY_DB_DIR, f"{company}.json")
            if not os.path.exists(db_path):
                missing_companies.append(company)
                continue
            
            try:
                with open(db_path, "r", encoding="utf-8") as f:
                    db = json.load(f)
                
                company_texts = db.get("texts", [])
                if not company_texts:
                    missing_companies.append(f"{company} (empty)")
                    continue
                
                evaluations = scorer.score_company(rfp_summary, company, company_texts)
                results[company] = evaluations
                totals[company] = sum(x["score"] for x in evaluations)
            except Exception as e:
                missing_companies.append(f"{company} (error: {str(e)})")
                continue
        
        if not results:
            return jsonify({
                'error': f'No company knowledge bases found. Missing: {", ".join(missing_companies)}. Please upload company context files or create JSON files in company_db/ directory.',
                'missing_companies': missing_companies
            }), 400
        
        # Determine winner
        if totals:
            best = max(totals, key=totals.get)
            best_compliance = round((totals[best] / MAX_SCORE) * 100, 2)
            best_decision = get_decision_category(best_compliance)
        else:
            best = None
            best_compliance = 0
            best_decision = "No-Go"
        
        # Format results for frontend
        formatted_results = {}
        for company, evals in results.items():
            score = sum(x["score"] for x in evals)
            compliance = round((score / MAX_SCORE) * 100, 2)
            formatted_results[company] = {
                "evaluations": evals,
                "totalScore": score,
                "maxScore": MAX_SCORE,
                "compliance": compliance,
                "decision": get_decision_category(compliance)
            }
        
        # Save RFP analysis results to bid_incoming and sync to go_bids
        try:
            # Access mysql from current_app
            mysql = getattr(current_app, 'mysql', None)
            
            if mysql and hasattr(mysql, 'connection'):
                cur = mysql.connection.cursor(DictCursor)
                
                # Extract RFP information from summary
                rfp_name = rfp_file.filename if rfp_file else "RFP Analysis"
                rfp_name = rfp_name.replace('.pdf', '').replace('.txt', '').replace('.docx', '')
                
                # Get project details from RFP summary
                project_type = rfp_summary.get('project_type', '')
                scope_text = rfp_summary.get('scope_summary', '') or rfp_summary.get('summary', '')
                if len(scope_text) > 500:
                    scope_text = scope_text[:500] + "..."
                
                # Use best company and decision
                comp_name = best if best else ''
                decision = best_decision if best_decision else 'No-Go'
                scoring = int(best_compliance) if best_compliance else 0
                
                # Default due date (30 days from now)
                due_date = (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')
                
                # Insert into bid_incoming
                cur.execute("""
                    INSERT INTO bid_incoming (b_name, due_date, state, scope, type, scoring, comp_name, decision, summary)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    rfp_name,
                    due_date,
                    'analyzer',
                    scope_text,
                    project_type,
                    scoring,
                    comp_name,
                    decision,
                    json.dumps(rfp_summary) if isinstance(rfp_summary, dict) else str(rfp_summary)
                ))
                
                bid_id = cur.lastrowid
                mysql.connection.commit()
                
                # Automatically sync to go_bids only if decision is GO
                if (decision or '').upper() == 'GO':
                    cur2 = mysql.connection.cursor(DictCursor)
                    cur2.execute("SELECT g_id FROM go_bids WHERE id=%s", (bid_id,))
                    row = cur2.fetchone()
                    args = (rfp_name, due_date, 'business', scope_text, project_type, scoring, comp_name, decision, json.dumps(rfp_summary) if isinstance(rfp_summary, dict) else str(rfp_summary))
                    if row:
                        cur2.execute("""UPDATE go_bids SET b_name=%s,due_date=%s,state=%s,scope=%s,
                                        type=%s,scoring=%s,company=%s,decision=%s,summary=%s WHERE id=%s""", (*args, bid_id))
                    else:
                        cur2.execute("""INSERT INTO go_bids (id,b_name,due_date,state,scope,type,scoring,company,decision,summary)
                                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                                     (bid_id, *args))
                    mysql.connection.commit()
                    cur2.close()
                cur.close()
        except Exception as db_error:
            # Log error but don't fail the RFP analysis
            print(f"Error saving to database: {str(db_error)}")
        
        return jsonify({
            'success': True,
            'rfp_summary': rfp_summary,
            'results': formatted_results,
            'winner': {
                'company': best,
                'compliance': best_compliance,
                'decision': best_decision
            },
            'missing_companies': missing_companies if missing_companies else None
        })
    
    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

def build_company_kb_from_file(file, company_name: str):
    """Build company knowledge base from uploaded file"""
    try:
        filename = secure_filename(file.filename)
        ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
        
        text = ""
        if ext == 'txt':
            text = file.read().decode('utf-8')
        elif ext == 'docx':
            doc = DocxDocument(file)
            text = "\n".join([p.text for p in doc.paragraphs])
        elif ext == 'json':
            data = json.load(file)
            if isinstance(data, dict) and 'texts' in data:
                # Already in correct format
                db_path = os.path.join(COMPANY_DB_DIR, f"{company_name}.json")
                with open(db_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f)
                return
            else:
                text = json.dumps(data)
        else:
            text = file.read().decode('utf-8', errors='ignore')
        
        # Split text into chunks (approx 500 words each)
        words = text.split()
        chunks = []
        chunk_size = 500
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i+chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        
        # Save as JSON
        db_path = os.path.join(COMPANY_DB_DIR, f"{company_name}.json")
        with open(db_path, 'w', encoding='utf-8') as f:
            json.dump({"texts": chunks}, f, indent=2)
        
        file.seek(0)
    except Exception as e:
        print(f"Error building KB for {company_name}: {e}")
        file.seek(0)

# Export blueprint
__all__ = ['rfp_bp']
