
"""
╔══════════════════════════════════════════════════════════════╗
║     TEACHER PERFORMANCE EVALUATION SYSTEM                    ║
║     Flask + SQLite + ML (scikit-learn: TF-IDF + LR)         ║
║     Crimson Theme Edition — ML-Upgraded + Block System       ║
╚══════════════════════════════════════════════════════════════╝
INSTALL REQUIREMENTS:    pip install flask werkzeug scikit-learn numpy
RUN:
    python app.py
"""
from flask import (Flask, render_template_string, request, redirect,
                   url_for, session, flash, jsonify)
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3, os, re, json, secrets, smtplib
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from functools import wraps
import threading
# ── scikit-learn ML ──
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")
# ─────────────────────────────────────────────
#  APP CONFIG
# ─────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = "TPES_SECRET_KEY_2025_change_in_prod"
DB = "tpes.db"
# ─────────────────────────────────────────────
#  EMAIL CONFIG
# ─────────────────────────────────────────────
SMTP_HOST     = "smtp.gmail.com"
SMTP_PORT     = 587
SMTP_USER     = "nicolejanegallardo242@gmail.com"
SMTP_PASSWORD = "qvlb atee wqtq rlka"
FROM_EMAIL    = "TPES System <nicolejanegallardo242@gmail.com>"
OTP_STORE: dict = {}

# ─────────────────────────────────────────────
#  BLOCK / SEMESTER CONFIG
# ─────────────────────────────────────────────
BLOCKS = [
    "BSCS 1", "BSCS 2", "BSCS 3", "BSCS 4",
    "ACT 1", "ACT 2",
    "BSED English 1", "BSED English 2", "BSED English 3", "BSED English 4",
    "BSED Mathematics 1", "BSED Mathematics 2", "BSED Mathematics 3", "BSED Mathematics 4",
    "BSED Filipino 1", "BSED Filipino 2", "BSED Filipino 3", "BSED Filipino 4",
    "BEED 1", "BEED 2", "BEED 3", "BEED 4",
]

DEPARTMENTS = ["BSCS", "ACT", "BSED English", "BSED Mathematics", "BSED Filipino", "BEED"]

SEMESTERS = ["1st Semester 2024-2025", "2nd Semester 2024-2025",
             "Summer 2025", "1st Semester 2025-2026", "2nd Semester 2025-2026"]

# ═══════════════════════════════════════════════════════════════
#  REAL ML ENGINE
# ═══════════════════════════════════════════════════════════════
TRAINING_DATA = [
    ("excellent teacher explains concepts very clearly and is always prepared", 5, "Excellent"),
    ("outstanding professor highly knowledgeable inspiring and patient", 5, "Excellent"),
    ("best teacher i have had very engaging and organized lessons", 5, "Excellent"),
    ("brilliant educator dedicated and professional always helpful", 5, "Excellent"),
    ("amazing teaching methods very effective and thorough explanations", 5, "Excellent"),
    ("wonderful teacher motivates students and gives clear feedback", 5, "Excellent"),
    ("superb instructor very skilled talented and approachable", 5, "Excellent"),
    ("fantastic professor makes learning enjoyable and accessible", 4, "Excellent"),
    ("exceptional teacher always prepared and fair in assessments", 5, "Excellent"),
    ("perfect teaching style very clear organized and professional", 4, "Excellent"),
    ("teacher is very knowledgeable and explains everything perfectly", 5, "Excellent"),
    ("love this class teacher is the best very engaging", 5, "Excellent"),
    ("inspired me to learn more highly recommend this professor", 5, "Excellent"),
    ("good teacher explains well but sometimes goes too fast", 4, "Good"),
    ("generally effective and organized could be more engaging", 4, "Good"),
    ("solid instructor good knowledge delivery and fair grading", 4, "Good"),
    ("helpful and accessible most of the time good overall", 4, "Good"),
    ("well prepared and knowledgeable good pace in lessons", 4, "Good"),
    ("clear explanations and good classroom management", 4, "Good"),
    ("good teaching style approachable and professional", 3, "Good"),
    ("mostly effective teacher with good command of subject", 4, "Good"),
    ("decent instructor explains concepts well minor issues", 4, "Good"),
    ("good teacher fair in grading and mostly engaging", 4, "Good"),
    ("average teacher sometimes unclear explanations", 3, "Average"),
    ("mediocre teaching style not very engaging", 3, "Average"),
    ("okay professor nothing special could improve delivery", 3, "Average"),
    ("teaching is average could be more organized", 3, "Average"),
    ("sometimes helpful but inconsistent in explanations", 3, "Average"),
    ("decent enough but not inspiring or very effective", 3, "Average"),
    ("average performance needs to improve engagement", 3, "Average"),
    ("class is okay teacher is not very dynamic", 3, "Average"),
    ("moderate teacher fair but could do much better", 2, "Average"),
    ("teaching is passable but lacks clarity and structure", 3, "Average"),
    ("poor explanations often unclear and confusing", 2, "Needs Improvement"),
    ("often unprepared and late to class disappointing", 2, "Needs Improvement"),
    ("difficult to follow lessons disorganized and slow", 2, "Needs Improvement"),
    ("unhelpful and unresponsive to student concerns", 2, "Needs Improvement"),
    ("unfair grading strict and biased teacher", 2, "Needs Improvement"),
    ("boring lectures monotone voice no engagement", 2, "Needs Improvement"),
    ("teacher is often absent and leaves no material", 2, "Needs Improvement"),
    ("ineffective teaching method confusing and hard to follow", 2, "Needs Improvement"),
    ("disappointing instructor lacks organization", 2, "Needs Improvement"),
    ("needs to improve communication and clarity", 1, "Needs Improvement"),
    ("terrible teacher worst class I have attended ever", 1, "Poor"),
    ("awful teaching no structure boring and rude", 1, "Poor"),
    ("very poor performance lazy and does not teach well", 1, "Poor"),
    ("complete failure as a teacher unhelpful and disorganized", 1, "Poor"),
    ("horrible experience teacher is always late and unprepared", 1, "Poor"),
    ("worst professor very unfair and biased in grading", 1, "Poor"),
    ("very bad teacher cannot explain anything clearly at all", 1, "Poor"),
    ("terrible experience do not recommend this teacher at all", 1, "Poor"),
    ("awful monotone boring and completely disorganized class", 1, "Poor"),
    ("poor teacher fails to deliver material and is unreliable", 1, "Poor"),
]

class TPESClassifier:
    LABELS = ["Poor", "Needs Improvement", "Average", "Good", "Excellent"]
    def __init__(self):
        self.text_pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=2000, sublinear_tf=True, min_df=1)),
            ("clf", LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced", solver="lbfgs", random_state=42)),
        ])
        self._trained = False
        self._train()
    def _train(self):
        texts  = [d[0] for d in TRAINING_DATA]
        labels = [d[2] for d in TRAINING_DATA]
        self.text_pipeline.fit(texts, labels)
        self._trained = True
    def predict(self, comments: str, avg_score: float) -> dict:
        if   avg_score >= 4.5: score_label = "Excellent"
        elif avg_score >= 3.5: score_label = "Good"
        elif avg_score >= 2.5: score_label = "Average"
        elif avg_score >= 1.5: score_label = "Needs Improvement"
        else:                  score_label = "Poor"
        if not comments or len(comments.strip()) < 10:
            return {"label": score_label, "confidence": self._score_confidence(avg_score),
                    "text_proba": {}, "score_label": score_label, "method": "score_only"}
        proba = self.text_pipeline.predict_proba([comments])[0]
        classes = self.text_pipeline.classes_
        text_proba_dict = {c: round(float(p)*100, 1) for c, p in zip(classes, proba)}
        score_idx = self.LABELS.index(score_label)
        score_dist = np.zeros(len(self.LABELS))
        for i in range(len(self.LABELS)):
            score_dist[i] = np.exp(-0.5 * ((i - score_idx) / 0.8) ** 2)
        score_dist /= score_dist.sum()
        text_dist = np.array([text_proba_dict.get(lbl, 0)/100 for lbl in self.LABELS])
        text_weight = min(0.65, 0.35 + len(comments.split()) * 0.01)
        combined = text_weight * text_dist + (1 - text_weight) * score_dist
        final_idx = int(np.argmax(combined))
        return {"label": self.LABELS[final_idx], "confidence": round(float(combined[final_idx])*100, 1),
                "text_proba": text_proba_dict, "score_label": score_label, "method": "ensemble"}
    def _score_confidence(self, avg: float) -> float:
        dists = [abs(avg - b) for b in [1.5, 2.5, 3.5, 4.5]]
        return round(min(95, 50 + min(dists) * 35), 1)
    def get_suggestion(self, avg_score: float, label: str, confidence: float) -> str:
        conf_note = f" (ML confidence: {confidence}%)" if confidence else ""
        suggestions = {
            "Excellent": f"Outstanding performance{conf_note}. This teacher demonstrates exemplary teaching effectiveness. Nominate for the Faculty Excellence Award and invite to mentor junior educators. Consider assigning advanced or honors classes.",
            "Good": f"Good overall performance{conf_note}. The teacher demonstrates solid command of the subject with effective delivery. Encourage participation in advanced pedagogical workshops and peer observation programs to reach excellence.",
            "Average": f"Average performance detected{conf_note}. While competency is present, there is clear room for improvement in teaching methodology and student engagement. Recommend enrolling in instructional design training and schedule a mid-term coaching session.",
            "Needs Improvement": f"Below-average performance identified{conf_note}. Significant gaps exist in teaching effectiveness and student satisfaction. Assign a senior faculty mentor, conduct a formal classroom observation, and schedule a detailed performance review within 30 days.",
            "Poor": f"Critical performance concern detected{conf_note}. Immediate intervention is required. Initiate a Performance Improvement Plan (PIP), arrange intensive coaching, and conduct bi-weekly supervisor evaluations. Student welfare should be the top priority.",
        }
        return suggestions.get(label, "No suggestion available.")

_classifier = None
def get_classifier() -> TPESClassifier:
    global _classifier
    if _classifier is None:
        _classifier = TPESClassifier()
    return _classifier

def get_performance_color(prediction):
    return {"Excellent":"#22c55e","Good":"#f59e0b","Average":"#f97316","Needs Improvement":"#dc2626","Poor":"#7f1d1d"}.get(prediction, "#6b7280")

def row_get(row, key, default=None):
    try:
        return row[key] if key in row.keys() else default
    except Exception:
        return default

# ─────────────────────────────────────────────
#  DATABASE
# ─────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn

def init_db():
    conn = get_db()
    c = conn.cursor()
    c.executescript("""
    CREATE TABLE IF NOT EXISTS users (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        name        TEXT NOT NULL,
        email       TEXT UNIQUE NOT NULL,
        password    TEXT NOT NULL,
        role        TEXT NOT NULL CHECK(role IN ('admin','teacher','student')),
        department  TEXT DEFAULT '',
        block       TEXT DEFAULT '',
        semester    TEXT DEFAULT '',
        created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    CREATE TABLE IF NOT EXISTS student_teacher (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id  INTEGER NOT NULL,
        teacher_id  INTEGER NOT NULL,
        block       TEXT NOT NULL,
        semester    TEXT NOT NULL,
        subject     TEXT DEFAULT '',
        created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(student_id, teacher_id, block, semester),
        FOREIGN KEY(student_id) REFERENCES users(id) ON DELETE CASCADE,
        FOREIGN KEY(teacher_id) REFERENCES users(id) ON DELETE CASCADE
    );
    CREATE TABLE IF NOT EXISTS question (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        question_text TEXT NOT NULL,
        created_at    DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    CREATE TABLE IF NOT EXISTS evaluation (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        teacher_id  INTEGER NOT NULL,
        student_id  INTEGER NOT NULL,
        question_id INTEGER NOT NULL,
        score       INTEGER NOT NULL CHECK(score BETWEEN 1 AND 5),
        comment     TEXT DEFAULT '',
        created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(teacher_id)  REFERENCES users(id) ON DELETE CASCADE,
        FOREIGN KEY(student_id)  REFERENCES users(id) ON DELETE CASCADE,
        FOREIGN KEY(question_id) REFERENCES question(id) ON DELETE CASCADE
    );
    CREATE TABLE IF NOT EXISTS teacher_suggestion (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        teacher_id      INTEGER NOT NULL UNIQUE,
        suggestion_text TEXT DEFAULT '',
        prediction      TEXT DEFAULT '',
        average_score   REAL DEFAULT 0,
        confidence      REAL DEFAULT 0,
        ml_method       TEXT DEFAULT '',
        text_proba      TEXT DEFAULT '{}',
        created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(teacher_id) REFERENCES users(id) ON DELETE CASCADE
    );
    """)

    # Migrate existing DB: add new columns if missing
    for tbl, col, defn in [
        ("users", "block", "TEXT DEFAULT ''"),
        ("users", "semester", "TEXT DEFAULT ''"),
        ("users", "status", "TEXT DEFAULT 'Regular'"),
    ]:
        try:
            c.execute(f"ALTER TABLE {tbl} ADD COLUMN {col} {defn}")
        except sqlite3.OperationalError:
            pass

    existing = c.execute("SELECT id FROM users WHERE role='admin'").fetchone()
    if not existing:
        c.execute("INSERT INTO users (name,email,password,role,department) VALUES (?,?,?,?,?)",
                  ("Administrator","admin@tpes.edu",generate_password_hash("admin123"),"admin","Administration"))

    if not c.execute("SELECT id FROM question").fetchone():
        for q in [
            "How effectively does the teacher explain concepts?",
            "How well does the teacher manage the classroom?",
            "How accessible is the teacher for student concerns?",
            "How well-prepared is the teacher for each class?",
            "How fairly does the teacher assess student work?"
        ]:
            c.execute("INSERT INTO question (question_text) VALUES (?)", (q,))

    for col, definition in [("confidence","REAL DEFAULT 0"),("ml_method","TEXT DEFAULT ''"),("text_proba","TEXT DEFAULT '{}'")]:
        try:
            c.execute(f"ALTER TABLE teacher_suggestion ADD COLUMN {col} {definition}")
        except sqlite3.OperationalError:
            pass

    conn.commit()
    conn.close()

# ─────────────────────────────────────────────
#  AUTH DECORATORS
# ─────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to continue.", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated

def role_required(*roles):
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            if session.get("role") not in roles:
                flash("Access denied.", "danger")
                return redirect(url_for("dashboard"))
            return f(*args, **kwargs)
        return decorated
    return decorator

# ─────────────────────────────────────────────
#  HELPER — recalculate teacher ML result
# ─────────────────────────────────────────────
def recalc_teacher(teacher_id):
    conn = get_db(); c = conn.cursor()
    try:
        rows = c.execute("SELECT e.score, e.comment FROM evaluation e WHERE e.teacher_id=?", (teacher_id,)).fetchall()
        if not rows:
            c.execute("DELETE FROM teacher_suggestion WHERE teacher_id=?", (teacher_id,))
            conn.commit()
            return
        scores   = [r["score"] for r in rows]
        comments = " ".join(r["comment"] or "" for r in rows)
        avg      = round(sum(scores)/len(scores), 2)
        clf      = get_classifier()
        result   = clf.predict(comments, avg)
        pred, conf, method, tproba = result["label"], result["confidence"], result["method"], json.dumps(result["text_proba"])
        suggestion = clf.get_suggestion(avg, pred, conf)
        existing = c.execute("SELECT id FROM teacher_suggestion WHERE teacher_id=?", (teacher_id,)).fetchone()
        if existing:
            c.execute("UPDATE teacher_suggestion SET suggestion_text=?,prediction=?,average_score=?,confidence=?,ml_method=?,text_proba=?,created_at=CURRENT_TIMESTAMP WHERE teacher_id=?",
                      (suggestion, pred, avg, conf, method, tproba, teacher_id))
        else:
            c.execute("INSERT INTO teacher_suggestion (teacher_id,suggestion_text,prediction,average_score,confidence,ml_method,text_proba) VALUES (?,?,?,?,?,?,?)",
                      (teacher_id, suggestion, pred, avg, conf, method, tproba))
        conn.commit()
    except Exception as e:
        conn.rollback()
        app.logger.error(f"recalc_teacher error: {e}")
    finally:
        conn.close()

# ════════════════════════════════════════════════════════════════
#  CRIMSON THEME CSS
# ════════════════════════════════════════════════════════════════
BASE_CSS = """
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
:root {
  --bg:#0d0608; --bg2:#110a0b; --surface:#1a0d0f; --surface2:#241215; --surface3:#2e1618;
  --border:#3d1a1d; --border2:#521f23;
  --crimson:#dc143c; --crimson-lt:#e8365a; --crimson-dk:#9b0e29;
  --blood:#8b0000; --rose:#ff6b7a; --blush:#ffb3bc;
  --gold:#c9963a; --gold-lt:#e8b45a; --gold-dk:#8a6520; --amber:#f59e0b;
  --text:#f5e8e9; --text-dim:#c9a8aa; --muted:#7a5055; --muted2:#5a3035;
  --success:#22c55e; --warning:#f59e0b; --danger:#ff4444; --info:#fb923c;
  --radius:10px; --radius-lg:16px; --radius-xl:22px;
  --shadow:0 8px 32px rgba(139,0,0,0.25); --shadow-lg:0 20px 60px rgba(139,0,0,0.35);
  --glow:0 0 24px rgba(220,20,60,0.3); --transition:all 0.22s cubic-bezier(0.4,0,0.2,1);
}
html { font-size: 15px; }
body { font-family:'Crimson Pro',Georgia,serif; background:var(--bg); color:var(--text); min-height:100vh; overflow-x:hidden; letter-spacing:0.01em; }
body::before { content:''; position:fixed; inset:0; z-index:0; background:radial-gradient(ellipse 70% 55% at 15% 5%,rgba(139,0,0,0.18) 0%,transparent 55%),radial-gradient(ellipse 50% 40% at 90% 95%,rgba(220,20,60,0.1) 0%,transparent 50%); pointer-events:none; }
h1,h2,h3,h4,h5 { font-family:'Playfair Display',Georgia,serif; font-weight:700; letter-spacing:-0.02em; }
a { color:var(--rose); text-decoration:none; transition:var(--transition); }
a:hover { color:var(--blush); }
::-webkit-scrollbar { width:4px; }
::-webkit-scrollbar-track { background:var(--surface); }
::-webkit-scrollbar-thumb { background:var(--border2); border-radius:99px; }
.layout { display:flex; min-height:100vh; position:relative; z-index:1; }
.sidebar { width:248px; flex-shrink:0; background:var(--surface); border-right:1px solid var(--border); display:flex; flex-direction:column; position:fixed; top:0; left:0; bottom:0; z-index:100; box-shadow:4px 0 24px rgba(0,0,0,0.4); }
.sidebar::before { content:''; position:absolute; left:0; top:0; bottom:0; width:2px; background:linear-gradient(180deg,transparent,var(--crimson),var(--gold),transparent); opacity:0.6; }
.sidebar-header { padding:28px 20px 22px; border-bottom:1px solid var(--border); }
.sidebar-logo { font-family:'Playfair Display',Georgia,serif; font-weight:900; font-size:1.15rem; color:var(--text); letter-spacing:0.05em; display:flex; align-items:center; gap:12px; text-transform:uppercase; }
.sidebar-logo-mark { width:36px; height:36px; background:linear-gradient(135deg,var(--crimson),var(--crimson-dk)); border-radius:8px; display:flex; align-items:center; justify-content:center; font-size:0.85rem; color:#fff; font-weight:900; box-shadow:0 4px 12px rgba(220,20,60,0.45); flex-shrink:0; }
.sidebar-wordmark { display:flex; flex-direction:column; gap:1px; }
.sidebar-wordmark-main { font-size:0.95rem; font-weight:900; color:var(--text); letter-spacing:0.12em; }
.sidebar-wordmark-sub { font-size:0.55rem; color:var(--muted); letter-spacing:0.2em; text-transform:uppercase; }
.sidebar-role { font-family:'JetBrains Mono',Consolas,monospace; font-size:0.6rem; color:var(--crimson); text-transform:uppercase; letter-spacing:0.2em; margin-top:10px; display:flex; align-items:center; gap:6px; }
.sidebar-role::before { content:''; display:inline-block; width:5px; height:5px; border-radius:50%; background:var(--crimson); box-shadow:0 0 6px var(--crimson); animation:pulse-dot 2s infinite; }
@keyframes pulse-dot { 0%,100%{opacity:1;transform:scale(1)}50%{opacity:.5;transform:scale(.8)} }
.sidebar-nav { flex:1; padding:18px 14px; overflow-y:auto; }
.nav-section { margin-bottom:28px; }
.nav-label { font-family:'JetBrains Mono',Consolas,monospace; font-size:0.58rem; color:var(--muted); text-transform:uppercase; letter-spacing:0.22em; padding:0 10px; margin-bottom:10px; display:flex; align-items:center; gap:8px; }
.nav-label::after { content:''; flex:1; height:1px; background:var(--border); }
.nav-link { display:flex; align-items:center; gap:11px; padding:9px 12px; border-radius:9px; color:var(--muted); font-size:0.92rem; cursor:pointer; transition:var(--transition); margin-bottom:3px; border:1px solid transparent; font-family:'Playfair Display',Georgia,serif; position:relative; }
.nav-link:hover { color:var(--text); background:var(--surface2); border-color:var(--border); }
.nav-link.active { color:var(--rose); background:rgba(220,20,60,0.12); border-color:rgba(220,20,60,0.25); }
.nav-link.active::before { content:''; position:absolute; left:-1px; top:20%; bottom:20%; width:2px; border-radius:99px; background:var(--crimson); box-shadow:0 0 8px var(--crimson); }
.nav-icon { font-size:0.95rem; width:18px; text-align:center; flex-shrink:0; opacity:0.8; }
.sidebar-footer { padding:16px 14px; border-top:1px solid var(--border); }
.user-info { display:flex; align-items:center; gap:10px; padding:10px 12px; border-radius:10px; background:var(--surface2); border:1px solid var(--border); margin-bottom:10px; }
.avatar { width:34px; height:34px; border-radius:50%; background:linear-gradient(135deg,var(--crimson-dk),var(--crimson)); display:flex; align-items:center; justify-content:center; font-weight:700; font-size:0.78rem; color:#fff; flex-shrink:0; font-family:'Playfair Display',Georgia,serif; }
.user-name { font-size:0.85rem; font-weight:600; color:var(--text); }
.user-role { font-family:'JetBrains Mono',Consolas,monospace; font-size:0.58rem; color:var(--muted); text-transform:uppercase; }
.main { flex:1; margin-left:248px; padding:36px 40px; min-height:100vh; }
.page-header { margin-bottom:32px; animation:fadeSlideDown 0.5s ease forwards; padding-bottom:24px; border-bottom:1px solid var(--border); position:relative; }
.page-header::after { content:''; position:absolute; bottom:-1px; left:0; width:60px; height:1px; background:linear-gradient(90deg,var(--crimson),transparent); }
.page-title { font-size:1.9rem; font-weight:900; color:var(--text); font-family:'Playfair Display',Georgia,serif; }
.page-title-accent { color:var(--crimson); }
.page-subtitle { font-size:0.9rem; color:var(--muted); margin-top:5px; font-style:italic; }
.page-actions { display:flex; gap:10px; margin-top:16px; flex-wrap:wrap; }
.card { background:var(--surface); border:1px solid var(--border); border-radius:var(--radius-lg); padding:24px; transition:var(--transition); position:relative; overflow:hidden; }
.card:hover { border-color:var(--border2); box-shadow:var(--shadow); }
.cards-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(200px,1fr)); gap:16px; margin-bottom:32px; }
.stat-card { background:var(--surface); border:1px solid var(--border); border-radius:var(--radius-lg); padding:22px; display:flex; flex-direction:column; gap:10px; position:relative; overflow:hidden; animation:fadeIn 0.5s ease forwards; transition:var(--transition); }
.stat-card:hover { transform:translateY(-3px); box-shadow:var(--shadow); }
.stat-card.crimson { border-color:rgba(220,20,60,0.3); }
.stat-card.crimson::before { background:radial-gradient(circle at 80% 0%,rgba(220,20,60,0.15) 0%,transparent 60%); content:''; position:absolute; inset:0; }
.stat-card.gold { border-color:rgba(201,150,58,0.3); }
.stat-card.gold::before { background:radial-gradient(circle at 80% 0%,rgba(201,150,58,0.12) 0%,transparent 60%); content:''; position:absolute; inset:0; }
.stat-card.rose { border-color:rgba(255,107,122,0.3); }
.stat-card.rose::before { background:radial-gradient(circle at 80% 0%,rgba(255,107,122,0.12) 0%,transparent 60%); content:''; position:absolute; inset:0; }
.stat-card.blood { border-color:rgba(139,0,0,0.4); }
.stat-card.blood::before { background:radial-gradient(circle at 80% 0%,rgba(139,0,0,0.2) 0%,transparent 60%); content:''; position:absolute; inset:0; }
.stat-icon { font-size:1.4rem; position:relative; z-index:1; }
.stat-value { font-family:'Playfair Display',Georgia,serif; font-size:2.2rem; font-weight:900; color:var(--text); position:relative; z-index:1; line-height:1; }
.stat-label { font-family:'JetBrains Mono',Consolas,monospace; font-size:0.62rem; color:var(--muted); text-transform:uppercase; letter-spacing:0.15em; position:relative; z-index:1; }
.btn { display:inline-flex; align-items:center; gap:7px; padding:9px 20px; border-radius:8px; font-size:0.88rem; font-weight:600; cursor:pointer; transition:var(--transition); border:none; font-family:'Playfair Display',Georgia,serif; white-space:nowrap; }
.btn-primary { background:linear-gradient(135deg,var(--crimson),var(--crimson-dk)); color:#fff; box-shadow:0 4px 14px rgba(220,20,60,0.4); }
.btn-primary:hover { background:linear-gradient(135deg,var(--crimson-lt),var(--crimson)); transform:translateY(-1px); color:#fff; }
.btn-secondary { background:var(--surface2); color:var(--text-dim); border:1px solid var(--border); }
.btn-secondary:hover { border-color:var(--border2); color:var(--text); background:var(--surface3); }
.btn-gold { background:linear-gradient(135deg,var(--gold),var(--gold-dk)); color:#1a0d0f; font-weight:700; }
.btn-danger { background:rgba(220,20,60,0.12); color:var(--rose); border:1px solid rgba(220,20,60,0.25); }
.btn-danger:hover { background:rgba(220,20,60,0.22); }
.btn-sm { padding:5px 13px; font-size:0.8rem; border-radius:6px; }
.table-wrap { overflow-x:auto; border-radius:var(--radius-lg); border:1px solid var(--border); background:var(--surface); }
table { width:100%; border-collapse:collapse; }
thead { background:var(--surface2); }
th { padding:13px 18px; text-align:left; font-family:'JetBrains Mono',Consolas,monospace; font-size:0.62rem; color:var(--muted); text-transform:uppercase; letter-spacing:0.15em; border-bottom:1px solid var(--border); }
td { padding:14px 18px; font-size:0.95rem; color:var(--text); border-bottom:1px solid rgba(61,26,29,0.4); vertical-align:middle; }
tr:last-child td { border-bottom:none; }
tbody tr:hover { background:var(--surface2); }
.form-group { margin-bottom:20px; }
label { display:block; margin-bottom:7px; font-family:'JetBrains Mono',Consolas,monospace; font-size:0.65rem; color:var(--muted); text-transform:uppercase; letter-spacing:0.15em; }
input[type=text],input[type=email],input[type=password],select,textarea { width:100%; background:var(--surface2); border:1px solid var(--border); border-radius:8px; padding:11px 14px; color:var(--text); font-family:'Playfair Display',Georgia,serif; font-size:1rem; outline:none; transition:var(--transition); }
input:focus,select:focus,textarea:focus { border-color:var(--crimson); box-shadow:0 0 0 3px rgba(220,20,60,0.12); }
input::placeholder,textarea::placeholder { color:var(--muted2); }
textarea { resize:vertical; min-height:85px; }
select option { background:var(--surface); }
.pw-wrap { position:relative; }
.pw-wrap input { padding-right:44px; }
.pw-toggle { position:absolute; right:12px; top:50%; transform:translateY(-50%); background:none; border:none; cursor:pointer; color:var(--text); font-size:1rem; padding:4px; transition:color 0.18s; line-height:1; display:flex; align-items:center; justify-content:center; }
.pw-toggle:hover { color:var(--rose); }
.alert { padding:12px 16px; border-radius:9px; margin-bottom:16px; font-size:0.95rem; display:flex; align-items:center; gap:10px; animation:fadeIn 0.4s ease; }
.alert-success { background:rgba(34,197,94,0.1); border:1px solid rgba(34,197,94,0.2); color:#4ade80; }
.alert-danger   { background:rgba(220,20,60,0.12); border:1px solid rgba(220,20,60,0.25); color:var(--rose); }
.alert-warning  { background:rgba(245,158,11,0.1); border:1px solid rgba(245,158,11,0.2); color:var(--amber); }
.alert-info     { background:rgba(251,146,60,0.1); border:1px solid rgba(251,146,60,0.2); color:var(--info); }
.badge { display:inline-block; padding:3px 10px; border-radius:99px; font-family:'JetBrains Mono',Consolas,monospace; font-size:0.6rem; text-transform:uppercase; letter-spacing:0.08em; border:1px solid transparent; }
.badge-admin   { background:rgba(220,20,60,0.15); color:var(--rose); border-color:rgba(220,20,60,0.2); }
.badge-teacher { background:rgba(201,150,58,0.15); color:var(--gold-lt); border-color:rgba(201,150,58,0.2); }
.badge-student { background:rgba(255,179,188,0.12); color:var(--blush); border-color:rgba(255,179,188,0.15); }
.badge-excellent { background:rgba(34,197,94,0.12); color:#4ade80; border-color:rgba(34,197,94,0.2); }
.badge-good      { background:rgba(201,150,58,0.15); color:var(--gold-lt); border-color:rgba(201,150,58,0.2); }
.badge-average   { background:rgba(249,115,22,0.12); color:#fb923c; border-color:rgba(249,115,22,0.2); }
.badge-needs     { background:rgba(220,20,60,0.12); color:var(--rose); border-color:rgba(220,20,60,0.2); }
.badge-poor      { background:rgba(139,0,0,0.2); color:#ff6666; border-color:rgba(139,0,0,0.3); }
.badge-block     { background:rgba(255,107,122,0.1); color:var(--blush); border-color:rgba(255,107,122,0.18); }
.stars { display:flex; gap:5px; }
.star { font-size:1.4rem; cursor:pointer; color:var(--border2); transition:color 0.12s; user-select:none; }
.star.filled,.star:hover { color:var(--gold); text-shadow:0 0 8px rgba(201,150,58,0.6); }
.rating-input { display:none; }
.score-bar-wrap { display:flex; align-items:center; gap:12px; }
.score-bar { flex:1; height:6px; background:var(--surface2); border-radius:99px; overflow:hidden; }
.score-fill { height:100%; border-radius:99px; background:linear-gradient(90deg,var(--crimson-dk),var(--crimson),var(--rose)); transition:width 1.2s cubic-bezier(0.4,0,0.2,1); }
.auth-page { min-height:100vh; display:flex; align-items:center; justify-content:center; padding:24px; background:radial-gradient(ellipse 70% 55% at 10% 5%,rgba(139,0,0,0.25) 0%,transparent 50%),var(--bg); }
.auth-box { width:100%; max-width:430px; background:var(--surface); border:1px solid var(--border); border-radius:var(--radius-xl); padding:44px; box-shadow:var(--shadow-lg); animation:fadeSlideUp 0.5s ease forwards; position:relative; }
.auth-box::before { content:''; position:absolute; top:0; left:0; right:0; height:2px; background:linear-gradient(90deg,transparent,var(--crimson),var(--gold),var(--crimson),transparent); }
.modal-overlay { display:none; position:fixed; inset:0; background:rgba(0,0,0,0.8); z-index:500; align-items:center; justify-content:center; padding:24px; backdrop-filter:blur(6px); }
.modal-overlay.open { display:flex; }
.modal { background:var(--surface); border:1px solid var(--border2); border-radius:var(--radius-xl); padding:30px; width:100%; max-width:500px; max-height:90vh; overflow-y:auto; animation:fadeSlideUp 0.3s ease; position:relative; }
.modal::before { content:''; position:absolute; top:0; left:0; right:0; height:1px; background:linear-gradient(90deg,transparent,var(--crimson),transparent); }
.modal-header { display:flex; align-items:center; justify-content:space-between; margin-bottom:22px; }
.modal-title { font-size:1.15rem; font-weight:700; font-family:'Playfair Display',Georgia,serif; }
.modal-close { width:30px; height:30px; border-radius:50%; background:var(--surface2); border:1px solid var(--border); color:var(--muted); cursor:pointer; font-size:0.9rem; display:flex; align-items:center; justify-content:center; transition:var(--transition); }
.modal-close:hover { color:var(--rose); }
.section { animation:fadeIn 0.5s ease forwards; margin-bottom:28px; }
.section-header { display:flex; align-items:center; justify-content:space-between; margin-bottom:16px; flex-wrap:wrap; gap:10px; }
.section-title { font-size:1.05rem; font-weight:700; font-family:'Playfair Display',Georgia,serif; }
.empty { text-align:center; padding:64px 24px; color:var(--muted); }
.empty-icon { font-size:2.5rem; margin-bottom:14px; opacity:0.4; }
.empty-msg  { font-size:0.9rem; font-style:italic; }
.hamburger { display:none; position:fixed; top:14px; left:14px; z-index:200; background:var(--surface); border:1px solid var(--border2); border-radius:8px; padding:8px 10px; cursor:pointer; color:var(--text); }
.separator { border:none; margin:22px 0; height:1px; background:linear-gradient(90deg,transparent,var(--border),transparent); }
.ml-panel { background:var(--surface2); border:1px solid var(--border); border-radius:var(--radius-lg); padding:20px 22px; margin-top:18px; position:relative; }
.ml-panel::before { content:'⬡ ML ENGINE'; position:absolute; top:-10px; left:16px; background:var(--surface2); padding:0 8px; font-family:'JetBrains Mono',Consolas,monospace; font-size:0.58rem; color:var(--crimson); letter-spacing:0.18em; }
.ml-method-tag { display:inline-flex; align-items:center; gap:6px; background:rgba(220,20,60,0.08); border:1px solid rgba(220,20,60,0.18); border-radius:99px; padding:3px 12px; font-family:'JetBrains Mono',Consolas,monospace; font-size:0.6rem; color:var(--rose); text-transform:uppercase; letter-spacing:0.12em; margin-bottom:14px; }
.ml-method-tag::before { content:''; width:5px; height:5px; border-radius:50%; background:var(--crimson); box-shadow:0 0 5px var(--crimson); }
.proba-row { display:flex; align-items:center; gap:10px; margin-bottom:10px; }
.proba-label { font-family:'JetBrains Mono',Consolas,monospace; font-size:0.65rem; color:var(--muted); width:130px; flex-shrink:0; text-transform:uppercase; }
.proba-bar-wrap { flex:1; height:6px; background:var(--surface3); border-radius:99px; overflow:hidden; }
.proba-bar-fill { height:100%; border-radius:99px; transition:width 1.3s cubic-bezier(0.4,0,0.2,1); }
.proba-pct { font-family:'JetBrains Mono',Consolas,monospace; font-size:0.65rem; color:var(--text-dim); width:38px; text-align:right; }
.confidence-ring { display:flex; flex-direction:column; align-items:center; justify-content:center; margin-bottom:16px; }
.confidence-value { font-family:'Playfair Display',Georgia,serif; font-size:2.4rem; font-weight:900; color:var(--text); line-height:1; }
.confidence-pct { font-family:'JetBrains Mono',Consolas,monospace; font-size:0.62rem; color:var(--muted); text-transform:uppercase; letter-spacing:0.15em; margin-top:4px; }
.result-card { background:var(--surface); border:1px solid var(--border); border-radius:var(--radius-lg); padding:24px; position:relative; overflow:hidden; }
.result-card::before { content:''; position:absolute; top:0; left:0; right:0; height:1px; background:linear-gradient(90deg,transparent,var(--crimson),var(--gold),var(--crimson),transparent); }
/* Block card */
.block-group { background:var(--surface2); border:1px solid var(--border); border-radius:var(--radius-lg); margin-bottom:18px; overflow:hidden; }
.block-group-header { display:flex; align-items:center; justify-content:space-between; padding:14px 20px; border-bottom:1px solid var(--border); cursor:pointer; }
.block-group-header:hover { background:var(--surface3); }
.block-title { font-family:'Playfair Display',Georgia,serif; font-size:1rem; font-weight:700; display:flex; align-items:center; gap:10px; }
.block-meta { font-family:'JetBrains Mono',Consolas,monospace; font-size:0.62rem; color:var(--muted); }
.block-body { padding:0 20px 16px; }
/* Password match indicator */
.pw-match-ok  { border-color: var(--success) !important; box-shadow: 0 0 0 3px rgba(34,197,94,0.12) !important; }
.pw-match-err { border-color: var(--danger)  !important; box-shadow: 0 0 0 3px rgba(255,68,68,0.12)  !important; }
.pw-error-msg { font-family:'JetBrains Mono',Consolas,monospace; font-size:0.62rem; color:var(--rose); margin-top:6px; display:none; }
@keyframes fadeIn { from{opacity:0}to{opacity:1} }
@keyframes fadeSlideDown { from{opacity:0;transform:translateY(-14px)}to{opacity:1;transform:translateY(0)} }
@keyframes fadeSlideUp { from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:translateY(0)} }
@media (max-width:768px) { .sidebar { transform:translateX(-100%); } .sidebar.open { transform:translateX(0); } .main { margin-left:0; padding:80px 16px 28px; } .hamburger { display:flex; } .cards-grid { grid-template-columns:1fr 1fr; } }
@media (max-width:480px) { .cards-grid { grid-template-columns:1fr; } .auth-box { padding:28px 22px; } }
.text-muted { color:var(--muted); font-size:0.9rem; }
.mt-2{margin-top:8px} .mt-3{margin-top:16px} .mt-4{margin-top:24px}
.flex{display:flex;align-items:center} .flex-between{display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px}
.gap-2{gap:8px} .gap-3{gap:12px}
"""

BASE_JS = """
const ham = document.getElementById('hamburger');
const sidebar = document.getElementById('sidebar');
if (ham && sidebar) {
  ham.addEventListener('click', () => sidebar.classList.toggle('open'));
  document.addEventListener('click', e => { if (!sidebar.contains(e.target) && e.target !== ham) sidebar.classList.remove('open'); });
}
document.querySelectorAll('.alert').forEach(a => {
  setTimeout(() => { a.style.opacity='0'; a.style.transition='opacity 0.5s'; setTimeout(()=>a.remove(),500); }, 4500);
});
function openModal(id)  { document.getElementById(id).classList.add('open'); }
function closeModal(id) { document.getElementById(id).classList.remove('open'); }
document.querySelectorAll('.modal-overlay').forEach(m => {
  m.addEventListener('click', e => { if(e.target===m) m.classList.remove('open'); });
});
document.querySelectorAll('.stars').forEach(starsEl => {
  const inp = document.getElementById(starsEl.dataset.input);
  const stars = starsEl.querySelectorAll('.star');
  stars.forEach((s,i) => {
    s.addEventListener('click', () => { if(inp) inp.value = i+1; stars.forEach((ss,j) => ss.classList.toggle('filled', j<=i)); });
    s.addEventListener('mouseenter', () => stars.forEach((ss,j) => ss.classList.toggle('filled', j<=i)));
  });
  starsEl.addEventListener('mouseleave', () => { const v = inp ? parseInt(inp.value)||0 : 0; stars.forEach((ss,j) => ss.classList.toggle('filled', j<v)); });
});
document.querySelectorAll('.score-fill, .proba-bar-fill').forEach(el => {
  const w = el.dataset.width || '0';
  el.style.width = '0%';
  setTimeout(() => el.style.width = w + '%', 300);
});
"""

def sidebar_html(role, name, active=""):
    initials = "".join(w[0].upper() for w in name.split()[:2]) if name else "?"
    if role == "admin":
        nav = f"""
        <div class="nav-section">
          <div class="nav-label">Overview</div>
          <a href="/admin/dashboard" class="nav-link {'active' if active=='dashboard' else ''}"><span class="nav-icon">⬡</span> Dashboard</a>
        </div>
        <div class="nav-section">
          <div class="nav-label">Management</div>
          <a href="/admin/users" class="nav-link {'active' if active=='users' else ''}"><span class="nav-icon">◉</span> Users</a>
          <a href="/admin/evaluations" class="nav-link {'active' if active=='evaluations' else ''}"><span class="nav-icon">◎</span> Evaluations</a>
          <a href="/admin/questions" class="nav-link {'active' if active=='questions' else ''}"><span class="nav-icon">◇</span> Questions</a>
          <a href="/admin/results" class="nav-link {'active' if active=='results' else ''}"><span class="nav-icon">◆</span> ML Results</a>
        </div>"""
    elif role == "teacher":
        nav = f"""
        <div class="nav-section">
          <div class="nav-label">Overview</div>
          <a href="/teacher/dashboard" class="nav-link {'active' if active=='dashboard' else ''}"><span class="nav-icon">⬡</span> Dashboard</a>
        </div>
        <div class="nav-section">
          <div class="nav-label">My Data</div>
          <a href="/teacher/evaluations" class="nav-link {'active' if active=='evaluations' else ''}"><span class="nav-icon">◎</span> My Evaluations</a>
          <a href="/teacher/results" class="nav-link {'active' if active=='results' else ''}"><span class="nav-icon">◆</span> My Results</a>
          <a href="/teacher/students" class="nav-link {'active' if active=='students' else ''}"><span class="nav-icon">◉</span> My Students</a>
        </div>"""
    else:
        nav = f"""
        <div class="nav-section">
          <div class="nav-label">Overview</div>
          <a href="/student/dashboard" class="nav-link {'active' if active=='dashboard' else ''}"><span class="nav-icon">⬡</span> Dashboard</a>
        </div>
        <div class="nav-section">
          <div class="nav-label">Evaluate</div>
          <a href="/student/evaluate" class="nav-link {'active' if active=='evaluate' else ''}"><span class="nav-icon">✦</span> Evaluate Teacher</a>
          <a href="/student/history" class="nav-link {'active' if active=='history' else ''}"><span class="nav-icon">◎</span> My Submissions</a>
        </div>"""
    return f"""
    <button class="hamburger" id="hamburger">☰</button>
    <aside class="sidebar" id="sidebar">
      <div class="sidebar-header">
        <div class="sidebar-logo">
          <div class="sidebar-logo-mark">TP</div>
          <div class="sidebar-wordmark">
            <div class="sidebar-wordmark-main">TPES</div>
            <div class="sidebar-wordmark-sub">Evaluation System</div>
          </div>
        </div>
        <div class="sidebar-role">{role} portal</div>
      </div>
      <nav class="sidebar-nav">{nav}</nav>
      <div class="sidebar-footer">
        <a href="/profile" class="user-info" style="text-decoration:none;display:flex;align-items:center;gap:10px;padding:10px 12px;border-radius:10px;background:var(--surface2);border:1px solid var(--border);margin-bottom:10px;transition:var(--transition);" onmouseover="this.style.borderColor='var(--border2)'" onmouseout="this.style.borderColor='var(--border)'">
  <div class="avatar">{initials}</div>
  <div><div class="user-name">{name[:18]}</div><div class="user-role">{role}</div></div>
</a>
        <a href="/logout-confirm" class="btn btn-secondary" style="width:100%;justify-content:center;font-size:0.85rem;">⎋ &nbsp;Logout</a>
      </div>
    </aside>"""

def flash_html():
    from flask import get_flashed_messages
    msgs = get_flashed_messages(with_categories=True)
    if not msgs: return ""
    html = ""
    for cat, msg in msgs:
        icon = {"success":"✓","danger":"✕","warning":"⚠","info":"ℹ"}.get(cat,"•")
        html += f'<div class="alert alert-{cat}">{icon} {msg}</div>'
    return html

def page(title, content, role, name, active=""):
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title} — TPES</title>
<link rel="preconnect" href="https://fonts.googleapis.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=Crimson+Pro:ital,wght@0,400;0,600;1,400&family=JetBrains+Mono:wght@400;500&display=swap" media="print" onload="this.media='all'">
<style>{BASE_CSS}</style>
</head>
<body>
<div class="layout">
  {sidebar_html(role, name, active)}
  <main class="main">
    {flash_html()}
    {content}
  </main>
</div>
<script>{BASE_JS}</script>
</body>
</html>"""

# ─────────────────────────────────────────────
#  PASSWORD MATCH JS (reusable)
# ─────────────────────────────────────────────
PW_MATCH_JS = """
function setupPwMatch(pwId, confirmId, errId) {
  const pw      = document.getElementById(pwId);
  const confirm = document.getElementById(confirmId);
  const err     = document.getElementById(errId);
  if (!pw || !confirm || !err) return;
  function check() {
    if (!confirm.value) { confirm.classList.remove('pw-match-ok','pw-match-err'); err.style.display='none'; return; }
    const match = pw.value === confirm.value;
    confirm.classList.toggle('pw-match-ok',  match);
    confirm.classList.toggle('pw-match-err', !match);
    err.style.display = match ? 'none' : 'block';
  }
  confirm.addEventListener('input', check);
  pw.addEventListener('input', check);
}
"""

AUTH_JS = """
document.querySelectorAll('.alert').forEach(a=>{
  setTimeout(()=>{a.style.opacity='0';a.style.transition='opacity 0.5s';setTimeout(()=>a.remove(),500);},4500);
});
document.querySelectorAll('.pw-toggle').forEach(btn=>{
  btn.addEventListener('click',()=>{
    const inp=btn.previousElementSibling;
    if(!inp)return;
    const isHidden=inp.type==='password';
    inp.type=isHidden?'text':'password';
    btn.textContent=isHidden?'HIDE':'SHOW';
  });
});
function switchTab(tab) {
  document.getElementById('loginPanel').style.display  = tab==='login'  ? 'block' : 'none';
  document.getElementById('registerPanel').style.display = tab==='register' ? 'block' : 'none';
  document.getElementById('tabLogin').classList.toggle('auth-tab-active',    tab==='login');
  document.getElementById('tabRegister').classList.toggle('auth-tab-active', tab==='register');
}
// Dynamic teacher list based on role selection
function onRoleChange(sel) {
  const isStudent = sel.value === 'student';
  const isTeacher = sel.value === 'teacher';
  const ids = ['statusSection','teacherSection','blockSection','semSection'];
  const teacherIds = ['teacherBlockSection','teacherSemSection'];
  ids.forEach(id => { const el = document.getElementById(id); if(el) el.style.display = isStudent ? 'block' : 'none'; });
  teacherIds.forEach(id => { const el = document.getElementById(id); if(el) el.style.display = isTeacher ? 'block' : 'none'; });
}
// Password match validation
function setupPwMatch(pwId, confirmId, errId) {
  const pw      = document.getElementById(pwId);
  const confirm = document.getElementById(confirmId);
  const err     = document.getElementById(errId);
  if (!pw || !confirm || !err) return;
  function check() {
    if (!confirm.value) { confirm.classList.remove('pw-match-ok','pw-match-err'); err.style.display='none'; return; }
    const match = pw.value === confirm.value;
    confirm.classList.toggle('pw-match-ok',  match);
    confirm.classList.toggle('pw-match-err', !match);
    err.style.display = match ? 'none' : 'block';
  }
  confirm.addEventListener('input', check);
  pw.addEventListener('input', check);
}
// Block form submit if passwords don't match
function guardPwForm(formId, pwId, confirmId) {
  const form = document.getElementById(formId);
  if (!form) return;
  form.addEventListener('submit', function(e) {
    const pw      = document.getElementById(pwId);
    const confirm = document.getElementById(confirmId);
    if (pw && confirm && pw.value !== confirm.value) {
      e.preventDefault();
      confirm.classList.add('pw-match-err');
      const err = document.getElementById(confirmId + '_err');
      if (err) err.style.display = 'block';
      confirm.focus();
    }
  });
}
document.addEventListener('DOMContentLoaded', function() {
  setupPwMatch('login_pw',  'login_confirm_pw',  'login_pw_err');
  setupPwMatch('reg_pw',    'reg_confirm_pw',    'reg_pw_err');
  guardPwForm('loginForm',    'login_pw', 'login_confirm_pw');
  guardPwForm('registerForm', 'reg_pw',   'reg_confirm_pw');
});
"""

BASE_CSS_TAG = f"<style>{BASE_CSS}</style>"

# ─────────────────────────────────────────────
#  AUTH FORMS
# ─────────────────────────────────────────────
def _login_form():
    return """
    <form method="POST" action="/login" autocomplete="off" id="loginForm">
        <div class="form-group">
            <label>Email Address</label>
            <input type="email" name="email" placeholder="you@example.com" required autocomplete="off">
        </div>
        <div class="form-group">
            <label>Password</label>
            <div class="pw-wrap">
                <input type="password" name="password" id="login_pw" placeholder="••••••••" required autocomplete="new-password">
                <button type="button" class="pw-toggle" style="font-family:'JetBrains Mono',monospace;font-size:0.55rem;letter-spacing:0.1em;color:var(--text);">SHOW</button>
            </div>
        </div>
        <div class="form-group">
            <label>Confirm Password</label>
            <div class="pw-wrap">
                <input type="password" name="confirm_password" id="login_confirm_pw" placeholder="••••••••" required autocomplete="new-password">
                <button type="button" class="pw-toggle" style="font-family:'JetBrains Mono',monospace;font-size:0.55rem;letter-spacing:0.1em;color:var(--text);">SHOW</button>
            </div>
            <div class="pw-error-msg" id="login_confirm_pw_err">✕ &nbsp;Passwords do not match</div>
            <div style="text-align:right;margin-top:6px;">
                <a href="/forgot-password" style="font-size:0.78rem;color:var(--muted);font-family:'JetBrains Mono',monospace;letter-spacing:0.05em;">Forgot password?</a>
            </div>
        </div>
        <button type="submit" class="btn btn-primary" style="width:100%;">Sign In →</button>
    </form>"""

def _register_form():
    block_opts = "".join(f'<option value="{b}">{b}</option>' for b in BLOCKS)
    sem_opts   = "".join(f'<option value="{s}">{s}</option>' for s in SEMESTERS)
    dept_opts  = "".join(f'<option value="{d}">{d}</option>' for d in DEPARTMENTS)
    return f"""
    <form method="POST" action="/register" autocomplete="off" id="registerForm">
      <div class="form-group">
        <label>Full Name</label>
        <input type="text" name="name" placeholder="Jane Smith" required autocomplete="off">
      </div>
      <div class="form-group">
        <label>Email</label>
        <input type="email" name="email" placeholder="you@institution.edu" required autocomplete="off">
      </div>
      <div class="form-group">
        <label>Password</label>
        <div class="pw-wrap">
          <input type="password" name="password" id="reg_pw" placeholder="Min. 6 characters" required autocomplete="new-password">
          <button type="button" class="pw-toggle" style="font-family:'JetBrains Mono',monospace;font-size:0.55rem;letter-spacing:0.1em;color:var(--text);">SHOW</button>
        </div>
      </div>
      <div class="form-group">
        <label>Confirm Password</label>
        <div class="pw-wrap">
          <input type="password" name="confirm_password" id="reg_confirm_pw" placeholder="Repeat your password" required autocomplete="new-password">
          <button type="button" class="pw-toggle" style="font-family:'JetBrains Mono',monospace;font-size:0.55rem;letter-spacing:0.1em;color:var(--text);">SHOW</button>
        </div>
        <div class="pw-error-msg" id="reg_confirm_pw_err">✕ &nbsp;Passwords do not match</div>
      </div>
      <div class="form-group">
        <label>Role</label>
        <select name="role" onchange="onRoleChange(this)" id="roleSelect">
          <option value="student" selected>Student</option>
          <option value="teacher">Teacher</option>
        </select>
      </div>
      <div class="form-group">
        <label>Department</label>
        <select name="department" id="deptSelect" required>
          <option value="">— Select Department —</option>
          {dept_opts}
        </select>
      </div>
      <!-- Student-only fields -->
      <div id="statusSection" class="form-group" style="display:none">
        <label>Student Status</label>
        <select name="status" id="statusSelect" onchange="onStatusChange(this)">
          <option value="Regular">Regular</option>
          <option value="Irregular">Irregular</option>
        </select>
      </div>
      <div id="blockSection" class="form-group">
        <label>Block / Section</label>
        <select name="block" id="blockSelect">
          <option value="">— Select your block —</option>
          {block_opts}
        </select>
      </div>
      <div id="semSection" class="form-group">
        <label>Semester</label>
        <select name="semester">
          <option value="">— Select semester —</option>
          {sem_opts}
        </select>
      </div>
      <div id="teacherSection" class="form-group">
        <label>My Teacher(s)</label>
        <div id="teacherLoadWrap">
          <select name="teacher_ids" id="teacherMultiSelect" multiple
            style="min-height:90px;font-size:0.88rem;"
            title="Hold Ctrl/Cmd to select multiple teachers">
          </select>
          <div style="font-family:'JetBrains Mono',monospace;font-size:0.58rem;color:var(--muted);margin-top:5px;letter-spacing:0.05em;">Hold Ctrl / ⌘ to select multiple</div>
        </div>
      </div>

      <!-- Teacher-only fields -->
      <div id="teacherBlockSection" class="form-group" style="display:none">
        <label>Blocks I Handle <span style="color:var(--muted);font-size:0.6rem;">(hold Ctrl/⌘ for multiple)</span></label>
        <select name="teacher_blocks" id="teacherBlockSelect" multiple style="min-height:110px;font-size:0.88rem;">
          {block_opts}
        </select>
      </div>
      <div id="teacherSemSection" class="form-group" style="display:none">
        <label>Semester</label>
        <select name="teacher_semester">
          <option value="">— Select semester —</option>
          {sem_opts}
        </select>
      </div>

     <button type="submit" class="btn btn-primary" style="width:100%;padding:12px;font-size:1rem;margin-top:4px;">Create Account &nbsp;→</button>
    </form>
    <script>
      // Run immediately after form renders
      (function() {{
        var s = document.getElementById('statusSection');
        var tb = document.getElementById('teacherBlockSection');
        var ts = document.getElementById('teacherSemSection');
        var bl = document.getElementById('blockSection');
        var se = document.getElementById('semSection');
        var te = document.getElementById('teacherSection');
        // Default is student — show student fields, hide teacher fields
        if(s)  s.style.display  = 'block';
        if(bl) bl.style.display = 'block';
        if(se) se.style.display = 'block';
        if(te) te.style.display = 'block';
        if(tb) tb.style.display = 'none';
        if(ts) ts.style.display = 'none';
      }})();
    </script>

    <script>
    fetch('/api/teachers').then(r=>r.json()).then(teachers=>{{
      window._allTeachers = teachers;
      renderTeachers(teachers);
    }});

    function renderTeachers(teachers) {{
      const sel = document.getElementById('teacherMultiSelect');
      sel.innerHTML = '';
      teachers.forEach(t=>{{
        const o = document.createElement('option');
        o.value = t.id;
        o.textContent = t.name + (t.department ? ' — '+t.department : '');
        sel.appendChild(o);
      }});
    }}

    function onRoleChange(sel) {{
      const isStudent = sel.value === 'student';
      const isTeacher = sel.value === 'teacher';
      document.getElementById('statusSection').style.display        = isStudent ? 'block' : 'none';
      document.getElementById('teacherSection').style.display       = isStudent ? 'block' : 'none';
      document.getElementById('blockSection').style.display         = isStudent ? 'block' : 'none';
      document.getElementById('semSection').style.display           = isStudent ? 'block' : 'none';
      document.getElementById('teacherBlockSection').style.display  = isTeacher ? 'block' : 'none';
      document.getElementById('teacherSemSection').style.display    = isTeacher ? 'block' : 'none';
      // Reset teacher list when switching roles
      if (isStudent && window._allTeachers) renderTeachers(window._allTeachers);
    }}

    function onStatusChange(sel) {{
      const isIrregular = sel.value === 'Irregular';
      const dept = document.getElementById('deptSelect').value;
      if (isIrregular) {{
        // Irregular: show ALL teachers regardless of department
        if (window._allTeachers) renderTeachers(window._allTeachers);
      }} else {{
        // Regular: filter by department
        if (window._allTeachers && dept) {{
          renderTeachers(window._allTeachers.filter(t => t.department === dept));
        }} else if (window._allTeachers) {{
          renderTeachers(window._allTeachers);
        }}
      }}
    }}

    document.getElementById('deptSelect').addEventListener('change', function() {{
      const dept = this.value;
      const role = document.getElementById('roleSelect').value;
      const status = document.getElementById('statusSelect') ? document.getElementById('statusSelect').value : 'Regular';
      if (role === 'student' && window._allTeachers) {{
        if (status === 'Irregular') {{
          // Irregular students see all teachers
          renderTeachers(window._allTeachers);
        }} else {{
          const filtered = dept
            ? window._allTeachers.filter(t => t.department === dept)
            : window._allTeachers;
          renderTeachers(filtered);
        }}
      }}
    }});
    </script>"""

AUTH_PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__ — TPES</title>
<link rel="preconnect" href="https://fonts.googleapis.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;0,900;1,700&family=Crimson+Pro:ital,wght@0,400;0,600;1,400;1,600&family=JetBrains+Mono:wght@400;500&display=swap" media="print" onload="this.media='all'">
<style>
__CSS__
.split-page { min-height:100vh; display:grid; grid-template-columns:1fr 1fr; }
.split-left { background:linear-gradient(160deg,var(--crimson-dk) 0%,var(--blood) 40%,#0d0608 100%); padding:56px 52px; display:flex; flex-direction:column; justify-content:space-between; position:relative; overflow:hidden; }
.split-left::before { content:''; position:absolute; width:480px; height:480px; border-radius:50%; border:1px solid rgba(220,20,60,0.15); top:-120px; left:-120px; pointer-events:none; }
.split-left::after { content:''; position:absolute; width:320px; height:320px; border-radius:50%; border:1px solid rgba(201,150,58,0.12); bottom:60px; right:-80px; pointer-events:none; }
.split-brand { display:flex; align-items:center; gap:12px; position:relative; z-index:1; }
.split-brand-mark { width:40px; height:40px; background:rgba(255,255,255,0.12); border:1px solid rgba(255,255,255,0.2); border-radius:10px; display:flex; align-items:center; justify-content:center; font-family:'Playfair Display',serif; font-weight:900; font-size:0.9rem; color:#fff; }
.split-brand-name { font-family:'JetBrains Mono',monospace; font-size:0.7rem; color:rgba(255,255,255,0.7); text-transform:uppercase; letter-spacing:0.2em; }
.split-hero { position:relative; z-index:1; }
.split-label { font-family:'JetBrains Mono',monospace; font-size:0.6rem; color:rgba(255,179,188,0.7); text-transform:uppercase; letter-spacing:0.25em; margin-bottom:24px; }
.split-headline { font-family:'Playfair Display',serif; font-size:3.2rem; font-weight:900; color:#fff; line-height:1.1; margin-bottom:20px; }
.split-headline em { font-style:italic; color:var(--rose); }
.split-desc { font-family:'Crimson Pro',serif; font-size:1.05rem; color:rgba(255,255,255,0.65); line-height:1.75; max-width:380px; margin-bottom:36px; }
.split-pills { display:flex; flex-wrap:wrap; gap:10px; margin-bottom:36px; }
.split-pill { display:flex; align-items:center; gap:7px; background:rgba(255,255,255,0.07); border:1px solid rgba(255,255,255,0.12); border-radius:99px; padding:7px 16px; font-family:'Crimson Pro',serif; font-size:0.88rem; color:rgba(255,255,255,0.8); }
.split-pill::before { content:''; width:6px; height:6px; border-radius:50%; background:var(--crimson); flex-shrink:0; }
.split-stat { background:rgba(255,255,255,0.06); border:1px solid rgba(255,255,255,0.1); border-radius:14px; padding:18px 22px; display:inline-flex; flex-direction:column; gap:4px; position:relative; z-index:1; }
.split-stat-num { font-family:'Playfair Display',serif; font-size:2rem; font-weight:900; color:#fff; }
.split-stat-label { font-family:'JetBrains Mono',monospace; font-size:0.58rem; color:rgba(255,179,188,0.6); text-transform:uppercase; letter-spacing:0.15em; }
.split-footer { font-family:'JetBrains Mono',monospace; font-size:0.58rem; color:rgba(255,255,255,0.25); letter-spacing:0.1em; position:relative; z-index:1; }
.split-right { background:var(--bg); display:flex; align-items:center; justify-content:center; padding:48px 52px; overflow-y:auto; }
.split-right-inner { width:100%; max-width:420px; }
.auth-tabs { display:grid; grid-template-columns:1fr 1fr; background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:4px; gap:4px; margin-bottom:32px; }
.auth-tab { padding:10px; border-radius:8px; text-align:center; font-family:'Playfair Display',serif; font-size:0.9rem; color:var(--muted); cursor:pointer; border:none; background:transparent; transition:all 0.2s; font-weight:600; }
.auth-tab:hover { color:var(--text); }
.auth-tab-active { background:linear-gradient(135deg,var(--crimson),var(--crimson-dk)); color:#fff !important; box-shadow:0 4px 12px rgba(220,20,60,0.35); }
.auth-panel-title { font-family:'Playfair Display',serif; font-size:1.6rem; font-weight:900; color:var(--text); margin-bottom:6px; }
.auth-panel-sub { font-family:'Crimson Pro',serif; font-size:0.92rem; color:var(--muted); font-style:italic; margin-bottom:28px; }
input[type="password"]::-ms-reveal, input[type="password"]::-ms-clear { display:none !important; }
@media (max-width:860px) { .split-page { grid-template-columns:1fr; } .split-left { display:none; } .split-right { padding:40px 24px; } }
</style>
</head>
<body>
<div class="split-page">
  <div class="split-left">
    <div class="split-brand">
      <div class="split-brand-mark">TP</div>
      <div class="split-brand-name">TPES &nbsp;·&nbsp; Evaluation System</div>
    </div>
    <div class="split-hero">
      <div class="split-label">Academic Excellence Platform</div>
      <div class="split-headline">Teacher<br><em>performance</em><br>refined.</div>
      <div class="split-desc">Evaluate teaching quality, gather structured feedback, and generate meaningful insights for academic excellence.</div>
      <div class="split-pills">
        <div class="split-pill">Structured Feedback</div>
        <div class="split-pill">Block-Based Tracking</div>
        <div class="split-pill">ML Analytics</div>
      </div>
      <div class="split-stat">
        <div class="split-stat-num">98%</div>
        <div class="split-stat-label">Evaluation completion rate</div>
      </div>
    </div>
    <div class="split-footer">&copy; NICOLE GALLARDO &nbsp;·&nbsp; Teacher Performance Evaluation System</div>
  </div>
  <div class="split-right">
    <div class="split-right-inner">
      __FLASHES__
      <div class="auth-tabs">
        <button class="auth-tab __LOGIN_ACTIVE__" id="tabLogin" onclick="switchTab('login')">Login</button>
        <button class="auth-tab __REGISTER_ACTIVE__" id="tabRegister" onclick="switchTab('register')">Sign Up</button>
      </div>
      <div id="loginPanel" style="display:__LOGIN_DISPLAY__">
        <div class="auth-panel-title">Sign in to your account</div>
        <div class="auth-panel-sub">Access evaluations, reports, and performance insights.</div>
        __LOGIN_FORM__
      </div>
      <div id="registerPanel" style="display:__REGISTER_DISPLAY__">
        <div class="auth-panel-title">Create your account</div>
        <div class="auth-panel-sub">Join the platform and start evaluating today.</div>
        __REGISTER_FORM__
      </div>
    </div>
  </div>
</div>
<script>__AUTH_JS__</script>
</body>
</html>"""

# ─────────────────────────────────────────────
#  EMAIL + OTP
# ─────────────────────────────────────────────
def send_email(to_addr, subject, html_body):
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject; msg["From"] = FROM_EMAIL; msg["To"] = to_addr
        msg.attach(MIMEText(html_body, "html"))
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.ehlo(); server.starttls(); server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(SMTP_USER, to_addr, msg.as_string())
        return True
    except Exception as e:
        app.logger.error(f"Email error: {e}"); return False

def generate_otp(email):
    code = f"{secrets.randbelow(1000000):06d}"
    OTP_STORE[email] = {"code": code, "expires": datetime.utcnow() + timedelta(minutes=10)}
    return code

def verify_otp(email, code):
    entry = OTP_STORE.get(email)
    if not entry: return False
    if datetime.utcnow() > entry["expires"]: OTP_STORE.pop(email, None); return False
    if entry["code"] != code.strip(): return False
    OTP_STORE.pop(email, None); return True

def email_format_valid(email):
    return bool(re.match(r'^[^@\s]+@[^@\s]+\.[^@\s]+$', email))

# ─────────────────────────────────────────────
#  API: teacher list for registration
# ─────────────────────────────────────────────
@app.route("/api/teachers")
def api_teachers():
    conn = get_db()
    teachers = conn.execute("SELECT id, name, department FROM users WHERE role='teacher' ORDER BY name").fetchall()
    conn.close()
    return jsonify([{"id": t["id"], "name": t["name"], "department": t["department"] or ""} for t in teachers])

# ════════════════════════════════════════════════════════════════
#  AUTH ROUTES
# ════════════════════════════════════════════════════════════════
@app.route("/")
def index():
    if "user_id" in session: return redirect(url_for("dashboard"))
    return redirect(url_for("login"))

@app.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":
        email            = request.form.get("email","").strip().lower()
        password         = request.form.get("password","")
        confirm_password = request.form.get("confirm_password","")

        # ── Confirm password check ──
        if password != confirm_password:
            flash("Passwords do not match. Please try again.", "danger")
            return (AUTH_PAGE_TEMPLATE
                .replace("__TITLE__","Login").replace("__CSS__",BASE_CSS).replace("__FLASHES__",flash_html())
                .replace("__LOGIN_FORM__",_login_form()).replace("__REGISTER_FORM__",_register_form())
                .replace("__LOGIN_ACTIVE__","auth-tab-active").replace("__REGISTER_ACTIVE__","")
                .replace("__LOGIN_DISPLAY__","block").replace("__REGISTER_DISPLAY__","none")
                .replace("__AUTH_JS__",AUTH_JS))

        try:
            conn = get_db()
            user = conn.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
            conn.close()
            if user and check_password_hash(user["password"], password):
                session.update({"user_id":user["id"],"name":user["name"],"role":user["role"],"email":user["email"]})
                return redirect(url_for("dashboard"))
            flash("Invalid email or password.", "danger")
        except Exception as e:
            flash(f"Login error: {str(e)}", "danger")
    return (AUTH_PAGE_TEMPLATE
        .replace("__TITLE__","Login").replace("__CSS__",BASE_CSS).replace("__FLASHES__",flash_html())
        .replace("__LOGIN_FORM__",_login_form()).replace("__REGISTER_FORM__",_register_form())
        .replace("__LOGIN_ACTIVE__","auth-tab-active").replace("__REGISTER_ACTIVE__","")
        .replace("__LOGIN_DISPLAY__","block").replace("__REGISTER_DISPLAY__","none")
        .replace("__AUTH_JS__",AUTH_JS))

@app.route("/register", methods=["GET","POST"])
def register():
    if request.method == "POST":
        name             = request.form.get("name","").strip()
        email            = request.form.get("email","").strip().lower()
        password         = request.form.get("password","")
        confirm_password = request.form.get("confirm_password","")
        role             = request.form.get("role","student")
        department       = request.form.get("department","").strip()
        block            = request.form.get("block","").strip()
        semester         = request.form.get("semester","").strip()
        status           = request.form.get("status","Regular").strip()
        teacher_ids      = request.form.getlist("teacher_ids")

        if not all([name, email, password]):
            flash("All fields are required.", "danger")
        elif not re.match(r'^[^@\s]+@[^@\s]+\.[^@\s]+$', email):
            flash("Please enter a valid email address.", "danger")
        elif not department:
            flash("Please select your department.", "danger")
        elif role not in ("teacher", "student"):
            flash("Invalid role.", "danger")
        elif len(password) < 6:
            flash("Password must be at least 6 characters.", "danger")
        elif password != confirm_password:
            flash("Passwords do not match. Please try again.", "danger")
        elif role == "student" and not block:
            flash("Please select your block/section.", "danger")
        elif role == "student" and not semester:
            flash("Please select your semester.", "danger")
        else:
            try:
                conn = get_db()

                # For teachers: use first selected block for the users table (or empty),
                # actual multi-block stored via student_teacher links later
                teacher_blocks   = request.form.getlist("teacher_blocks")   # list of blocks
                teacher_semester = request.form.get("teacher_semester", "").strip()

                # For the users table, store first block + semester (for profile display)
                if role == "teacher":
                    primary_block    = teacher_blocks[0] if teacher_blocks else ""
                    primary_semester = teacher_semester
                else:
                    primary_block    = block
                    primary_semester = semester

                conn.execute(
                    "INSERT INTO users(name,email,password,role,department,block,semester,status) VALUES(?,?,?,?,?,?,?,?)",
                    (name, email, generate_password_hash(password), role, department, primary_block, primary_semester,
                     status if role == "student" else "")
                )
                user_id = conn.execute("SELECT id FROM users WHERE email=?", (email,)).fetchone()["id"]

                if role == "student" and teacher_ids:
                    # Manually selected teachers
                    for tid in teacher_ids:
                        try:
                            conn.execute(
                                "INSERT OR IGNORE INTO student_teacher(student_id,teacher_id,block,semester) VALUES(?,?,?,?)",
                                (user_id, int(tid), block, semester)
                            )
                        except Exception:
                            pass

                if role == "student" and block and semester:
                    # Auto-link: find teachers whose handled blocks+semester match this student
                    matching_teachers = conn.execute("""
                        SELECT DISTINCT st.teacher_id FROM student_teacher st
                        JOIN users u ON st.teacher_id = u.id
                        WHERE st.block = ? AND st.semester = ? AND u.department = ? AND u.role = 'teacher'
                    """, (block, semester, department)).fetchall()
                    for mt in matching_teachers:
                        try:
                            conn.execute(
                                "INSERT OR IGNORE INTO student_teacher(student_id,teacher_id,block,semester) VALUES(?,?,?,?)",
                                (user_id, mt["teacher_id"], block, semester)
                            )
                        except Exception:
                            pass

                if role == "teacher" and teacher_blocks and teacher_semester:
                    # Store each block the teacher handles as a sentinel row
                    # (student_id=0 means "teacher handles this block")
                    for tb in teacher_blocks:
                        try:
                            conn.execute(
                                "INSERT OR IGNORE INTO student_teacher(student_id,teacher_id,block,semester) VALUES(?,?,?,?)",
                                (0, user_id, tb, teacher_semester)
                            )
                        except Exception:
                            pass
                    # Auto-link existing students that match teacher's blocks+semester+dept
                    for tb in teacher_blocks:
                        matching_students = conn.execute("""
                            SELECT u.id FROM users u
                            WHERE u.role='student' AND u.block=? AND u.semester=? AND u.department=?
                        """, (tb, teacher_semester, department)).fetchall()
                        for ms in matching_students:
                            try:
                                conn.execute(
                                    "INSERT OR IGNORE INTO student_teacher(student_id,teacher_id,block,semester) VALUES(?,?,?,?)",
                                    (ms["id"], user_id, tb, teacher_semester)
                                )
                            except Exception:
                                pass

                conn.commit(); conn.close()
                flash("Account created. Please log in.", "success")
                return redirect(url_for("login"))
            except sqlite3.IntegrityError:
                flash("Email already registered.", "danger")
            except Exception as e:
                flash(f"Error: {str(e)}", "danger")

    return (AUTH_PAGE_TEMPLATE
        .replace("__TITLE__","Register").replace("__CSS__",BASE_CSS).replace("__FLASHES__",flash_html())
        .replace("__LOGIN_FORM__",_login_form()).replace("__REGISTER_FORM__",_register_form())
        .replace("__LOGIN_ACTIVE__","").replace("__REGISTER_ACTIVE__","auth-tab-active")
        .replace("__LOGIN_DISPLAY__","none").replace("__REGISTER_DISPLAY__","block")
        .replace("__AUTH_JS__",AUTH_JS))

@app.route("/forgot-password", methods=["GET","POST"])
def forgot_password():
    if request.method == "POST":
        email = request.form.get("email","").strip().lower()
        if not email_format_valid(email):
            flash("Please enter a valid email address.","danger")
        else:
            conn = get_db()
            user = conn.execute("SELECT id FROM users WHERE email=?", (email,)).fetchone()
            conn.close()
            if user:
                code = generate_otp(email)
                html = f'<div style="font-family:sans-serif;max-width:480px;margin:auto"><h2 style="color:#8b0000;">TPES Password Reset</h2><p>Your one-time verification code is:</p><div style="font-size:2.5rem;font-weight:bold;letter-spacing:0.3em;background:#f4f4f4;padding:16px 24px;border-radius:8px;display:inline-block;color:#333;">{code}</div><p style="color:#666;font-size:0.88rem;margin-top:12px;">This code expires in <strong>10 minutes</strong>.</p></div>'
                result = {"ok": False}
                def send_bg(): result["ok"] = send_email(email, "TPES — Password Reset Code", html)
                t = threading.Thread(target=send_bg); t.start(); t.join(timeout=6)
                if result["ok"]: flash("A 6-digit code has been sent to your email.", "success")
                else: flash(f"(Dev mode) Your reset code is: {code}", "info")
            else:
                flash("If that email is registered, a reset code was sent.", "info")
            return redirect(url_for("forgot_verify", email=email))
    page_html = f"""<div style="min-height:100vh;display:flex;align-items:center;justify-content:center;background:var(--bg);"><div class="auth-card" style="max-width:420px;width:100%;padding:2.5rem;"><h2>Forgot Password</h2>{flash_html()}<form method="POST" autocomplete="off"><div class="form-group"><label>Email Address</label><input type="email" name="email" placeholder="you@example.com" required></div><button type="submit" class="btn btn-primary" style="width:100%;">Send Reset Code →</button></form><div style="text-align:center;margin-top:1rem;"><a href="/login">← Back to Login</a></div></div></div>"""
    return render_template_string(f"<!DOCTYPE html><html><head><title>Forgot Password</title>{BASE_CSS_TAG}</head><body>{page_html}</body></html>")

@app.route("/forgot-verify", methods=["GET","POST"])
def forgot_verify():
    email = request.args.get("email","").strip().lower() or request.form.get("email","").strip().lower()
    if not email: return redirect(url_for("forgot_password"))
    if request.method == "POST":
        code = request.form.get("code","").strip()
        if verify_otp(email, code):
            session["pw_reset_email"] = email
            return redirect(url_for("forgot_reset"))
        flash("Invalid or expired code.","danger")
    page_html = f"""<div style="min-height:100vh;display:flex;align-items:center;justify-content:center;background:var(--bg);"><div class="auth-card" style="max-width:420px;width:100%;padding:2.5rem;"><h2>Enter Verification Code</h2><p style="color:var(--text-muted);font-size:0.88rem;margin-bottom:1.5rem;">Sent to <strong>{email}</strong>. Expires in 10 minutes.</p>{flash_html()}<form method="POST" autocomplete="off"><input type="hidden" name="email" value="{email}"><div class="form-group"><label>6-Digit Code</label><input type="text" name="code" placeholder="••••••" maxlength="6" required style="letter-spacing:0.35em;font-size:1.4rem;text-align:center;"></div><button type="submit" class="btn btn-primary" style="width:100%;">Verify Code →</button></form><div style="text-align:center;margin-top:1rem;"><a href="/forgot-password">Resend code</a> &nbsp;·&nbsp; <a href="/login">← Login</a></div></div></div>"""
    return render_template_string(f"<!DOCTYPE html><html><head><title>Verify Code</title>{BASE_CSS_TAG}</head><body>{page_html}</body></html>")

@app.route("/forgot-reset", methods=["GET","POST"])
def forgot_reset():
    email = session.get("pw_reset_email")
    if not email: flash("Session expired.","danger"); return redirect(url_for("forgot_password"))
    if request.method == "POST":
        pw = request.form.get("password",""); pw2 = request.form.get("password2","")
        if len(pw) < 6: flash("Password must be at least 6 characters.","danger")
        elif pw != pw2: flash("Passwords do not match.","danger")
        else:
            conn = get_db(); conn.execute("UPDATE users SET password=? WHERE email=?", (generate_password_hash(pw), email)); conn.commit(); conn.close()
            session.pop("pw_reset_email", None); flash("Password updated. Please log in.","success"); return redirect(url_for("login"))
    page_html = f"""<div style="min-height:100vh;display:flex;align-items:center;justify-content:center;background:var(--bg);"><div class="auth-card" style="max-width:420px;width:100%;padding:2.5rem;"><h2>Set New Password</h2>{flash_html()}<form method="POST" autocomplete="off"><div class="form-group"><label>New Password</label><div class="pw-wrap"><input type="password" name="password" id="pw1" placeholder="Min. 6 characters" required><button type="button" class="pw-toggle" onclick="var i=document.getElementById('pw1');i.type=i.type==='password'?'text':'password';" style="font-family:'JetBrains Mono',monospace;font-size:0.55rem;">SHOW</button></div></div><div class="form-group"><label>Confirm Password</label><div class="pw-wrap"><input type="password" name="password2" id="pw2" placeholder="Repeat password" required><button type="button" class="pw-toggle" onclick="var i=document.getElementById('pw2');i.type=i.type==='password'?'text':'password';" style="font-family:'JetBrains Mono',monospace;font-size:0.55rem;">SHOW</button></div></div><button type="submit" class="btn btn-primary" style="width:100%;">Update Password →</button></form></div></div>"""
    return render_template_string(f"<!DOCTYPE html><html><head><title>Reset Password</title>{BASE_CSS_TAG}</head><body>{page_html}</body></html>")

@app.route("/logout")
def logout():
    session.clear(); flash("You have been logged out.","info"); return redirect(url_for("login"))

@app.route("/logout-confirm")
@login_required
def logout_confirm():
    page_html = f"""<div style="min-height:100vh;display:flex;align-items:center;justify-content:center;background:var(--bg);"><div style="max-width:400px;width:100%;padding:2.5rem;background:var(--surface);border:1px solid var(--border);border-radius:var(--radius-xl);box-shadow:var(--shadow-lg);text-align:center;"><div style="width:64px;height:64px;border-radius:50%;background:rgba(220,20,60,0.12);border:1px solid rgba(220,20,60,0.3);display:flex;align-items:center;justify-content:center;margin:0 auto 20px;font-size:1.8rem;">⎋</div><h2 style="font-family:'Playfair Display',serif;color:var(--text);margin-bottom:8px;">Confirm Logout</h2><p style="color:var(--muted);font-size:0.92rem;margin-bottom:28px;font-style:italic;">Are you sure you want to log out?</p><div style="display:flex;gap:12px;justify-content:center;"><a href="/logout" class="btn btn-danger" style="padding:10px 28px;">Yes, Logout</a><a href="/dashboard" class="btn btn-secondary" style="padding:10px 28px;">Cancel</a></div></div></div>"""
    return render_template_string(f"<!DOCTYPE html><html><head><title>Logout</title>{BASE_CSS_TAG}</head><body>{page_html}</body></html>")

@app.route("/dashboard")
@login_required
def dashboard():
    role = session.get("role")
    if role == "admin":   return redirect(url_for("admin_dashboard"))
    if role == "teacher": return redirect(url_for("teacher_dashboard"))
    return redirect(url_for("student_dashboard"))

# ════════════════════════════════════════════════════════════════
#  ADMIN ROUTES
# ════════════════════════════════════════════════════════════════
@app.route("/admin/dashboard")
@login_required
@role_required("admin")
def admin_dashboard():
    conn = get_db()
    total_users    = conn.execute("SELECT COUNT(*) FROM users WHERE role!='admin'").fetchone()[0]
    total_teachers = conn.execute("SELECT COUNT(*) FROM users WHERE role='teacher'").fetchone()[0]
    total_students = conn.execute("SELECT COUNT(*) FROM users WHERE role='student'").fetchone()[0]
    total_evals = conn.execute("""
    SELECT COUNT(DISTINCT e.student_id || '-' || e.teacher_id)
    FROM evaluation e
    JOIN users u_t ON e.teacher_id = u_t.id
    JOIN users u_s ON e.student_id = u_s.id
""").fetchone()[0]
    recent = conn.execute("""
    SELECT u_t.id as teacher_id, u_t.name as teacher_name,
           COUNT(DISTINCT e.student_id) as student_count,
           ROUND(AVG(e.score),1) as avg_score,
           MAX(e.created_at) as created_at
    FROM evaluation e
    JOIN users u_t ON e.teacher_id=u_t.id
    GROUP BY e.teacher_id
    ORDER BY created_at DESC
    LIMIT 6""").fetchall()
    conn.close()
    rows = "".join(f"""<tr>
        <td><b>{r['teacher_name']}</b></td>
        <td style="color:var(--gold)">{'★'*round(r['avg_score'])}{'☆'*(5-round(r['avg_score']))} <span class="text-muted">({r['avg_score']}/5)</span></td>
        <td class="text-muted">{r['student_count']} student{'s' if r['student_count']!=1 else ''}</td>
        <td class="text-muted" style="font-family:'JetBrains Mono',monospace;font-size:0.75rem">{r['created_at'][:10]}</td>
        </tr>""" for r in recent)
    content = f"""
    <div class="page-header"><div class="page-title">Admin <span class="page-title-accent">Dashboard</span></div><div class="page-subtitle">System overview and performance metrics</div></div>
    <div class="cards-grid">
      <div class="stat-card crimson"><div class="stat-icon">◉</div><div class="stat-value">{total_users}</div><div class="stat-label">Total Users</div></div>
      <div class="stat-card gold"><div class="stat-icon">◈</div><div class="stat-value">{total_teachers}</div><div class="stat-label">Teachers</div></div>
      <div class="stat-card rose"><div class="stat-icon">✦</div><div class="stat-value">{total_students}</div><div class="stat-label">Students</div></div>
      <div class="stat-card blood"><div class="stat-icon">◎</div><div class="stat-value">{total_evals}</div><div class="stat-label">Evaluations</div></div>
    </div>
    <div class="section">
      <div class="section-header"><div class="section-title">Recent Evaluations</div><a href="/admin/evaluations" class="btn btn-secondary btn-sm">View All</a></div>
      <div class="table-wrap"><table>
        <thead><tr><th>#</th><th>Teacher</th><th>Student</th><th>Score</th><th>Date</th></tr></thead>
        <tbody>{'<tr><td colspan="5"><div class="empty"><div class="empty-icon">◌</div><div class="empty-msg">No evaluations yet</div></div></td></tr>' if not recent else rows}</tbody>
      </table></div>
    </div>"""
    return page("Dashboard", content, "admin", session["name"], "dashboard")

@app.route("/admin/users")
@login_required
@role_required("admin")
def admin_users():
    conn = get_db()
    users = conn.execute("SELECT id,name,email,role,department,block,semester,created_at FROM users WHERE role!='admin' ORDER BY created_at DESC").fetchall()
    conn.close()
    rows = "".join(f"""<tr>
        <td style="font-family:'JetBrains Mono',monospace;font-size:0.75rem;color:var(--muted)">#{u['id']}</td>
        <td><b>{u['name']}</b></td><td class="text-muted">{u['email']}</td>
        <td><span class="badge badge-{u['role']}">{u['role']}</span></td>
        <td class="text-muted">{u['department'] or '—'}</td>
        <td>{'<span class="badge badge-block">'+u['block']+'</span>' if u['block'] else '—'}</td>
        <td class="text-muted" style="font-size:0.8rem">{u['semester'] or '—'}</td>
        <td class="text-muted" style="font-family:'JetBrains Mono',monospace;font-size:0.75rem">{u['created_at'][:10]}</td>
        <td><form method="POST" action="/admin/users/delete/{u['id']}" onsubmit="return confirm('Delete {u['name']}?')">
          <button class="btn btn-danger btn-sm">✕ Delete</button></form></td>
        </tr>""" for u in users)
    content = f"""
    <div class="page-header"><div class="page-title">Registered <span class="page-title-accent">Users</span></div><div class="page-subtitle">All students and teachers</div></div>
    <div class="table-wrap"><table>
      <thead><tr><th>#</th><th>Name</th><th>Email</th><th>Role</th><th>Dept</th><th>Block</th><th>Semester</th><th>Joined</th><th>Action</th></tr></thead>
      <tbody>{'<tr><td colspan="9"><div class="empty"><div class="empty-icon">◌</div><div class="empty-msg">No users found</div></div></td></tr>' if not users else rows}</tbody>
    </table></div>"""
    return page("Users", content, "admin", session["name"], "users")

@app.route("/admin/users/delete/<int:uid>", methods=["POST"])
@login_required
@role_required("admin")
def admin_delete_user(uid):
    try:
        conn = get_db(); conn.execute("DELETE FROM users WHERE id=? AND role!='admin'", (uid,)); conn.commit(); conn.close()
        flash("User deleted.", "success")
    except Exception as e: flash(f"Error: {str(e)}", "danger")
    return redirect(url_for("admin_users"))

@app.route("/admin/evaluations")
@login_required
@role_required("admin")
def admin_evaluations():
    conn = get_db()
    evals = conn.execute("""
        SELECT e.id, u_t.id as teacher_id, u_t.name as teacher_name,
               e.student_id,
               q.question_text, e.score, e.comment, e.created_at
        FROM evaluation e
        JOIN users u_t ON e.teacher_id=u_t.id
        JOIN question q ON e.question_id=q.id
        ORDER BY u_t.name, e.created_at DESC""").fetchall()
    conn.close()

    from collections import OrderedDict
    teachers = OrderedDict()
    for e in evals:
        tid = e["teacher_id"]
        if tid not in teachers:
            teachers[tid] = {
                "name": e["teacher_name"],
                "evals": [],
                "score_sum": 0,
                "student_ids": set()
            }
        teachers[tid]["evals"].append(e)
        teachers[tid]["score_sum"] += e["score"]
        teachers[tid]["student_ids"].add(e["student_id"])

    summary_rows = ""
    detail_sections = ""
    for tid, data in teachers.items():
        student_count = len(data["student_ids"])
        avg = round(data["score_sum"] / len(data["evals"]), 1)
        stars_avg = "★" * round(avg) + "☆" * (5 - round(avg))
        summary_rows += f"""
        <tr id="summary-{tid}">
          <td><b>{data['name']}</b></td>
          <td style="color:var(--gold)">{stars_avg} <span class="text-muted">({avg}/5)</span></td>
          <td style="font-family:'JetBrains Mono',monospace;font-size:0.78rem;color:var(--muted)">
            {student_count} student{'s' if student_count != 1 else ''}
          </td>
          <td><button class="btn btn-secondary btn-sm" onclick="toggleDetails({tid})">▾ &nbsp;View All</button></td>
        </tr>
        <tr id="details-{tid}" style="display:none"><td colspan="4" style="padding:0">
          <div style="background:var(--surface2);border-top:1px solid var(--border);padding:16px 20px">
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.6rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.15em;margin-bottom:12px">Individual Evaluations — {data['name']}</div>
            <table style="width:100%;border-collapse:collapse"><thead><tr style="border-bottom:1px solid var(--border)">
              <th style="padding:8px 12px;text-align:left;font-family:'JetBrains Mono',monospace;font-size:0.58rem;color:var(--muted);text-transform:uppercase">#</th>
              <th style="padding:8px 12px;text-align:left;font-family:'JetBrains Mono',monospace;font-size:0.58rem;color:var(--muted);text-transform:uppercase">Question</th>
              <th style="padding:8px 12px;text-align:left;font-family:'JetBrains Mono',monospace;font-size:0.58rem;color:var(--muted);text-transform:uppercase">Score</th>
              <th style="padding:8px 12px;text-align:left;font-family:'JetBrains Mono',monospace;font-size:0.58rem;color:var(--muted);text-transform:uppercase">Comment</th>
              <th style="padding:8px 12px;text-align:left;font-family:'JetBrains Mono',monospace;font-size:0.58rem;color:var(--muted);text-transform:uppercase">Date</th>
              <th style="padding:8px 12px;text-align:left;font-family:'JetBrains Mono',monospace;font-size:0.58rem;color:var(--muted);text-transform:uppercase">Del</th>
            </tr></thead><tbody>"""
        for e in data["evals"]:
            detail_sections += f"""<tr style="border-bottom:1px solid rgba(61,26,29,0.3)">
                <td style="padding:10px 12px;font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:var(--muted)">#{e['id']}</td>
                <td style="padding:10px 12px;max-width:220px;font-size:0.88rem">{e['question_text'][:55]}{'...' if len(e['question_text']) > 55 else ''}</td>
                <td style="padding:10px 12px;color:var(--gold)">{'★' * e['score']}{'☆' * (5 - e['score'])}</td>
                <td style="padding:10px 12px;color:var(--text-dim);font-size:0.88rem">{e['comment'] or '—'}</td>
                <td style="padding:10px 12px;font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:var(--muted)">{e['created_at'][:10]}</td>
                <td style="padding:10px 12px"><form method="POST" action="/admin/evaluations/delete/{e['id']}" onsubmit="return confirm('Delete?')"><button class="btn btn-danger btn-sm">✕</button></form></td>
                </tr>"""
        detail_sections += "</tbody></table></div></td></tr>"

    content = f"""
    <div class="page-header">
      <div class="page-title">All <span class="page-title-accent">Evaluations</span></div>
      <div class="page-subtitle">One row per teacher — student count based · click View All to expand</div>
    </div>
    <div class="table-wrap"><table>
      <thead><tr>
        <th>Teacher</th>
        <th>Overall Avg Score</th>
        <th>Students Evaluated</th>
        <th>Action</th>
      </tr></thead>
      <tbody>{'<tr><td colspan="4"><div class="empty"><div class="empty-icon">◌</div><div class="empty-msg">No evaluations yet</div></div></td></tr>' if not teachers else summary_rows + detail_sections}</tbody>
    </table></div>
    <script>
    function toggleDetails(tid) {{
      const row = document.getElementById('details-' + tid);
      const btn = document.querySelector('#summary-' + tid + ' button');
      const isOpen = row.style.display !== 'none';
      row.style.display = isOpen ? 'none' : 'table-row';
      btn.innerHTML = isOpen ? '▾ &nbsp;View All' : '▴ &nbsp;Collapse';
    }}
    </script>"""
    return page("Evaluations", content, "admin", session["name"], "evaluations")

@app.route("/admin/evaluations/delete/<int:eid>", methods=["POST"])
@login_required
@role_required("admin")
def admin_delete_eval(eid):
    try:
        conn = get_db()
        eval_row = conn.execute("SELECT teacher_id FROM evaluation WHERE id=?", (eid,)).fetchone()
        conn.execute("DELETE FROM evaluation WHERE id=?", (eid,)); conn.commit(); conn.close()
        if eval_row: recalc_teacher(eval_row["teacher_id"])
        flash("Evaluation deleted.", "success")
    except Exception as e: flash(f"Error: {str(e)}", "danger")
    return redirect(url_for("admin_evaluations"))

@app.route("/admin/questions", methods=["GET","POST"])
@login_required
@role_required("admin")
def admin_questions():
    if request.method == "POST":
        action = request.form.get("action")
        if action == "add":
            qtext = request.form.get("question_text","").strip()
            if not qtext: flash("Question text required.", "danger")
            else:
                try:
                    conn = get_db(); conn.execute("INSERT INTO question(question_text)VALUES(?)", (qtext,)); conn.commit(); conn.close()
                    flash("Question added.", "success")
                except Exception as e: flash(f"Error: {str(e)}", "danger")
        elif action == "delete":
            qid = request.form.get("qid")
            try:
                conn = get_db(); conn.execute("DELETE FROM question WHERE id=?", (qid,)); conn.commit(); conn.close()
                flash("Question deleted.", "success")
            except Exception as e: flash(f"Error: {str(e)}", "danger")
        return redirect(url_for("admin_questions"))
    conn = get_db(); questions = conn.execute("SELECT * FROM question ORDER BY id").fetchall(); conn.close()
    rows = "".join(f"""<tr>
        <td style="font-family:'JetBrains Mono',monospace;font-size:0.75rem;color:var(--muted)">#{q['id']}</td>
        <td>{q['question_text']}</td>
        <td class="text-muted" style="font-family:'JetBrains Mono',monospace;font-size:0.75rem">{q['created_at'][:10]}</td>
        <td><form method="POST"><input type="hidden" name="action" value="delete"><input type="hidden" name="qid" value="{q['id']}">
          <button class="btn btn-danger btn-sm" onclick="return confirm('Delete?')">✕</button></form></td>
        </tr>""" for q in questions)
    content = f"""
    <div class="page-header"><div class="page-title">Evaluation <span class="page-title-accent">Questions</span></div>
      <div class="page-actions"><button class="btn btn-primary" onclick="openModal('addQ')">＋ Add Question</button></div></div>
    <div class="table-wrap"><table>
      <thead><tr><th>#</th><th>Question</th><th>Created</th><th>Action</th></tr></thead>
      <tbody>{'<tr><td colspan="4"><div class="empty"><div class="empty-icon">◌</div><div class="empty-msg">No questions yet</div></div></td></tr>' if not questions else rows}</tbody>
    </table></div>
    <div class="modal-overlay" id="addQ"><div class="modal">
      <div class="modal-header"><div class="modal-title">Add New Question</div><button class="modal-close" onclick="closeModal('addQ')">✕</button></div>
      <form method="POST"><input type="hidden" name="action" value="add">
        <div class="form-group"><label>Question Text</label><textarea name="question_text" required></textarea></div>
        <button type="submit" class="btn btn-primary">Add Question</button>
      </form></div></div>"""
    return page("Questions", content, "admin", session["name"], "questions")

@app.route("/admin/results")
@login_required
@role_required("admin")
def admin_results():
    conn = get_db()
    raw = conn.execute("""SELECT ts.*,u.name,u.email,u.department FROM teacher_suggestion ts
        JOIN users u ON ts.teacher_id=u.id ORDER BY ts.average_score DESC""").fetchall()
    results = [dict(r) for r in raw]
    conn.close()
    def badge_class(pred):
        return {"Excellent":"excellent","Good":"good","Average":"average","Needs Improvement":"needs","Poor":"poor"}.get(pred,"average")
    rows = "".join(f"""<tr>
        <td><b>{r['name']}</b></td><td class="text-muted">{r.get('department') or '—'}</td>
        <td><div class="score-bar-wrap"><div class="score-bar"><div class="score-fill" data-width="{round(r.get('average_score',0)/5*100,1)}"></div></div><b style="font-family:'Playfair Display',serif">{r.get('average_score',0)}/5</b></div></td>
        <td><span class="badge badge-{badge_class(r.get('prediction',''))}">{r.get('prediction','—')}</span></td>
        <td><span style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;color:var(--rose)">{r.get('confidence',0)}%</span></td>
        <td class="text-muted" style="max-width:180px">{(r.get('suggestion_text') or '')[:80]}…</td>
        <td class="text-muted" style="font-family:'JetBrains Mono',monospace;font-size:0.72rem">{(r.get('created_at') or '')[:10]}</td>
        </tr>""" for r in results)
    content = f"""
    <div class="page-header"><div class="page-title">ML <span class="page-title-accent">Results</span></div><div class="page-subtitle">scikit-learn TF-IDF + Logistic Regression — ensemble with numeric scores</div></div>
    <div class="table-wrap"><table>
      <thead><tr><th>Teacher</th><th>Dept</th><th>Avg Score</th><th>Prediction</th><th>Confidence</th><th>Suggestion</th><th>Updated</th></tr></thead>
      <tbody>{'<tr><td colspan="7"><div class="empty"><div class="empty-icon">◌</div><div class="empty-msg">No ML results yet.</div></div></td></tr>' if not results else rows}</tbody>
    </table></div>"""
    return page("ML Results", content, "admin", session["name"], "results")

# ════════════════════════════════════════════════════════════════
#  TEACHER ROUTES
# ════════════════════════════════════════════════════════════════
@app.route("/teacher/dashboard")
@login_required
@role_required("teacher")
def teacher_dashboard():
    tid = session["user_id"]; conn = get_db()
    count_evals = conn.execute("SELECT COUNT(DISTINCT student_id) FROM evaluation WHERE teacher_id=?", (tid,)).fetchone()[0]
    avg_r       = conn.execute("SELECT AVG(score) FROM evaluation WHERE teacher_id=?", (tid,)).fetchone()[0]
    avg_score   = round(avg_r, 2) if avg_r else 0
    ts_raw      = conn.execute("SELECT * FROM teacher_suggestion WHERE teacher_id=?", (tid,)).fetchone()
    ts          = dict(ts_raw) if ts_raw else None
    student_count = conn.execute("SELECT COUNT(DISTINCT student_id) FROM student_teacher WHERE teacher_id=? AND student_id != 0", (tid,)).fetchone()[0]
    conn.close()

    prediction = ts["prediction"] if ts else "—"
    suggestion = ts["suggestion_text"] if ts else "No evaluations received yet."
    avg_stored = ts["average_score"] if ts else 0
    confidence = ts.get("confidence", 0) if ts else 0
    ml_method  = ts.get("ml_method", "") if ts else ""
    text_proba = json.loads(ts["text_proba"]) if ts and ts.get("text_proba") else {}

    def badge_class(pred):
        return {"Excellent":"excellent","Good":"good","Average":"average","Needs Improvement":"needs","Poor":"poor"}.get(pred,"average")

    LABELS = ["Excellent","Good","Average","Needs Improvement","Poor"]
    COLORS = {"Excellent":"#22c55e","Good":"#f59e0b","Average":"#f97316","Needs Improvement":"#dc2626","Poor":"#7f1d1d"}
    proba_bars = ""
    if text_proba:
        for lbl in LABELS:
            pct = text_proba.get(lbl, 0)
            proba_bars += f"""<div class="proba-row"><div class="proba-label">{lbl}</div><div class="proba-bar-wrap"><div class="proba-bar-fill" data-width="{pct}" style="background:{COLORS[lbl]};width:0%"></div></div><div class="proba-pct">{pct}%</div></div>"""

    no_comment_badge = ('<span style="display:inline-flex;align-items:center;gap:6px;background:rgba(245,158,11,0.10);border:1px solid rgba(245,158,11,0.28);border-radius:99px;padding:3px 11px;font-family:\'JetBrains Mono\',monospace;font-size:0.58rem;color:#f59e0b;text-transform:uppercase;letter-spacing:0.12em;margin-left:8px;">&#9888; No Comment Submitted Yet</span>') if not text_proba else ""
    confidence_warning = ('<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.58rem;color:#f59e0b;text-align:center;margin-top:6px;letter-spacing:0.08em;">&#9888; Score-only</div>') if not text_proba else ""
    no_proba_box = ('<div style="display:flex;align-items:flex-start;gap:10px;padding:12px 14px;background:rgba(245,158,11,0.07);border:1px solid rgba(245,158,11,0.20);border-radius:8px;margin-top:6px;"><span style="font-size:1rem;color:#f59e0b;">&#9888;</span><span style="font-family:\'JetBrains Mono\',monospace;font-size:0.68rem;color:#f59e0b;line-height:1.6;">No comments submitted yet.<br><span style="color:var(--muted);">Comments enable NLP sentiment analysis.</span></span></div>')

    ml_section = ""
    if ts:
        ml_section = f"""
        <div class="ml-panel">
          <div class="ml-method-tag">{ml_method.replace('_',' ').upper() if ml_method else 'ML ENGINE'}</div>
          <div style="display:grid;grid-template-columns:100px 1fr;gap:16px;align-items:start">
            <div class="confidence-ring">
              <div class="confidence-value">{confidence}<span style="font-size:1.2rem">%</span></div>
              <div class="confidence-pct">Confidence</div>
              {confidence_warning}
            </div>
            <div>
              <div style="font-family:'JetBrains Mono',monospace;font-size:0.6rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.15em;margin-bottom:10px">Feedback Analysis{no_comment_badge}</div>
              {proba_bars if proba_bars else no_proba_box}
            </div>
          </div>
        </div>"""

    result_card = f"""
    <div class="result-card">
      <div class="flex-between" style="margin-bottom:18px">
        <div class="section-title">Performance Analysis</div>
        <span class="badge badge-{badge_class(prediction)}" style="font-size:0.72rem;padding:5px 14px">{prediction}</span>
      </div>
      <div class="score-bar-wrap" style="margin-bottom:16px">
        <div class="score-bar" style="height:8px"><div class="score-fill" data-width="{round(avg_stored/5*100,1) if avg_stored else 0}"></div></div>
        <b style="font-family:'Playfair Display',serif;font-size:1.2rem">{avg_stored}/5</b>
      </div>
      <p style="font-size:0.95rem;color:var(--text-dim);line-height:1.7;font-family:'Crimson Pro',serif;font-style:italic">{suggestion}</p>
      {ml_section}
    </div>"""

    content = f"""
    <div class="page-header">
      <div class="page-title">Welcome, <span class="page-title-accent">{session['name'].split()[0]}</span></div>
      <div class="page-subtitle">Your teaching performance overview</div>
    </div>
    <div class="cards-grid">
      <div class="stat-card gold"><div class="stat-icon">◎</div><div class="stat-value">{count_evals}</div><div class="stat-label">Total Evaluations</div></div>
      <div class="stat-card crimson"><div class="stat-icon">★</div><div class="stat-value">{avg_score}</div><div class="stat-label">Avg Score / 5</div></div>
      <div class="stat-card rose"><div class="stat-icon">◆</div><div class="stat-value">{confidence}%</div><div class="stat-label">ML Confidence</div></div>
      <div class="stat-card blood"><div class="stat-icon">◉</div><div class="stat-value">{student_count}</div><div class="stat-label">My Students</div></div>
    </div>
    {result_card}
    <div class="mt-3 flex gap-2" style="flex-wrap:wrap">
      <a href="/teacher/evaluations" class="btn btn-secondary">View Evaluations</a>
      <a href="/teacher/results" class="btn btn-secondary">Detailed Results</a>
      <a href="/teacher/students" class="btn btn-gold">◉ &nbsp;My Students</a>
    </div>"""
    return page("Dashboard", content, "teacher", session["name"], "dashboard")

@app.route("/teacher/evaluations")
@login_required
@role_required("teacher")
def teacher_evaluations():
    tid = session["user_id"]; conn = get_db()
    evals = conn.execute("""SELECT e.id,q.question_text,e.score,e.comment,e.created_at
        FROM evaluation e JOIN question q ON e.question_id=q.id
        WHERE e.teacher_id=? ORDER BY e.created_at DESC""", (tid,)).fetchall()
    conn.close()
    rows = "".join(f"""<tr>
        <td style="font-family:'JetBrains Mono',monospace;font-size:0.75rem;color:var(--muted)">#{e['id']}</td>
        <td style="max-width:240px">{e['question_text']}</td>
        <td style="color:var(--gold)">{'★'*e['score']}{'☆'*(5-e['score'])} <span class="text-muted">({e['score']}/5)</span></td>
        <td class="text-muted">{e['comment'] or '—'}</td>
        <td class="text-muted" style="font-family:'JetBrains Mono',monospace;font-size:0.75rem">{e['created_at'][:10]}</td>
        </tr>""" for e in evals)
    content = f"""
    <div class="page-header"><div class="page-title">My <span class="page-title-accent">Evaluations</span></div><div class="page-subtitle">Anonymous student feedback</div></div>
    <div class="table-wrap"><table>
      <thead><tr><th>#</th><th>Question</th><th>Score</th><th>Comment</th><th>Date</th></tr></thead>
      <tbody>{'<tr><td colspan="5"><div class="empty"><div class="empty-icon">◌</div><div class="empty-msg">No evaluations received yet</div></div></td></tr>' if not evals else rows}</tbody>
    </table></div>"""
    return page("My Evaluations", content, "teacher", session["name"], "evaluations")

@app.route("/teacher/results")
@login_required
@role_required("teacher")
def teacher_results():
    tid = session["user_id"]; conn = get_db()
    ts_raw = conn.execute("SELECT * FROM teacher_suggestion WHERE teacher_id=?", (tid,)).fetchone()
    ts = dict(ts_raw) if ts_raw else None
    by_q = conn.execute("""SELECT q.question_text,AVG(e.score) as avg_s,COUNT(*) as cnt
        FROM evaluation e JOIN question q ON e.question_id=q.id
        WHERE e.teacher_id=? GROUP BY e.question_id ORDER BY avg_s DESC""", (tid,)).fetchall()
    conn.close()

    def badge_class(pred):
        return {"Excellent":"excellent","Good":"good","Average":"average","Needs Improvement":"needs","Poor":"poor"}.get(pred,"average")

    if not ts:
        content = f"""
        <div class="page-header"><div class="page-title">My <span class="page-title-accent">Results</span></div></div>
        <div class="empty" style="padding:80px"><div class="empty-icon">◌</div><div class="empty-msg">No evaluations received yet. Results appear after students submit feedback.</div></div>"""
    else:
        confidence = ts.get("confidence", 0); ml_method = ts.get("ml_method","score_only")
        text_proba = json.loads(ts["text_proba"]) if ts.get("text_proba") else {}
        LABELS = ["Excellent","Good","Average","Needs Improvement","Poor"]
        COLORS = {"Excellent":"#22c55e","Good":"#f59e0b","Average":"#f97316","Needs Improvement":"#dc2626","Poor":"#7f1d1d"}
        proba_bars = "".join(f"""<div class="proba-row"><div class="proba-label">{lbl}</div><div class="proba-bar-wrap"><div class="proba-bar-fill" data-width="{text_proba.get(lbl,0)}" style="background:{COLORS[lbl]};width:0%"></div></div><div class="proba-pct">{text_proba.get(lbl,0)}%</div></div>""" for lbl in LABELS) if text_proba else ""
        q_bars = "".join(f"""<div style="margin-bottom:18px"><div class="flex-between" style="margin-bottom:7px"><span style="font-size:0.92rem;font-family:'Crimson Pro',serif">{q['question_text'][:65]}{'...' if len(q['question_text'])>65 else ''}</span><span style="font-size:0.88rem;color:var(--gold);font-weight:700;font-family:'Playfair Display',serif">{round(q['avg_s'],2)}/5</span></div><div class="score-bar" style="height:7px"><div class="score-fill" data-width="{round(q['avg_s']/5*100,1)}"></div></div><div style="font-family:'JetBrains Mono',monospace;font-size:0.6rem;color:var(--muted);margin-top:5px">{q['cnt']} response{'s' if q['cnt']!=1 else ''}</div></div>""" for q in by_q)
        no_proba_box_r = ('<div style="display:flex;align-items:flex-start;gap:12px;padding:16px 18px;background:rgba(245,158,11,0.07);border:1px solid rgba(245,158,11,0.22);border-radius:10px;"><span style="font-size:1.3rem;color:#f59e0b;">&#9888;</span><div><div style="font-family:\'JetBrains Mono\',monospace;font-size:0.7rem;color:#f59e0b;text-transform:uppercase;letter-spacing:0.12em;margin-bottom:5px;">No Comment Submitted Yet</div><div style="font-size:0.88rem;color:var(--muted);font-style:italic;line-height:1.6;">Probability analysis requires at least one comment.</div></div></div>') if not text_proba else ""
        content = f"""
        <div class="page-header"><div class="page-title">My <span class="page-title-accent">Results</span></div></div>
        <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:16px;margin-bottom:24px">
          <div class="stat-card gold"><div class="stat-icon">★</div><div class="stat-value">{ts.get('average_score',0)}</div><div class="stat-label">Average Score</div></div>
          <div class="stat-card crimson" style="display:flex;flex-direction:column;gap:10px;"><div class="stat-icon" style="position:relative;z-index:1;">⬡</div><div class="stat-value" style="position:relative;z-index:1;">{confidence}%</div><div class="stat-label" style="position:relative;z-index:1;">ML Confidence</div></div>
          <div class="card" style="display:flex;flex-direction:column;justify-content:center;gap:10px"><div style="font-family:'JetBrains Mono',monospace;font-size:0.6rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.15em">Prediction</div><span class="badge badge-{badge_class(ts.get('prediction',''))}" style="font-size:0.78rem;padding:7px 18px;align-self:flex-start">{ts.get('prediction','—')}</span><div class="ml-method-tag" style="align-self:flex-start">{ml_method.replace('_',' ')}</div></div>
        </div>
        <div class="card" style="margin-bottom:20px"><div class="section-title" style="margin-bottom:18px">Performance by Question</div>{q_bars or '<p class="text-muted">No question data.</p>'}</div>
        <div class="card" style="margin-bottom:20px"><div class="section-title">Feedback Analysis</div><p style="font-size:0.82rem;color:var(--muted);margin-bottom:16px;font-style:italic">Insights derived from evaluation responses</p>{proba_bars if proba_bars else no_proba_box_r}</div>
        <div class="result-card"><div class="section-title" style="margin-bottom:12px">AI Recommendation</div><p style="font-size:0.95rem;color:var(--text-dim);line-height:1.8;font-family:'Crimson Pro',serif;font-style:italic">{ts.get('suggestion_text','')}</p><div style="font-family:'JetBrains Mono',monospace;font-size:0.62rem;color:var(--muted);margin-top:14px">Last updated: {(ts.get('created_at') or '')[:16]}</div></div>"""
    return page("My Results", content, "teacher", session["name"], "results")

@app.route("/teacher/students")
@login_required
@role_required("teacher")
def teacher_students():
    tid = session["user_id"]; conn = get_db()
    rows = conn.execute("""
        SELECT u.id, u.name, u.email, u.department,
               st.block, st.semester, st.subject,
               (SELECT COUNT(*) FROM evaluation e WHERE e.student_id=u.id AND e.teacher_id=?) as eval_count,
               (SELECT AVG(e.score) FROM evaluation e WHERE e.student_id=u.id AND e.teacher_id=?) as avg_score
        FROM student_teacher st
        JOIN users u ON st.student_id = u.id
        WHERE st.teacher_id=? AND st.student_id != 0
        ORDER BY st.semester DESC, st.block, u.name
    """, (tid, tid, tid)).fetchall()
    conn.close()

    from collections import OrderedDict
    groups = OrderedDict()
    for r in rows:
        key = (r["semester"] or "Unknown Semester", r["block"] or "Unknown Block")
        if key not in groups:
            groups[key] = []
        groups[key].append(r)

    total_students = len(rows)
    total_blocks   = len(groups)

    if not groups:
        content = f"""
        <div class="page-header">
          <div class="page-title">My <span class="page-title-accent">Students</span></div>
          <div class="page-subtitle">Students assigned to you by block and semester</div>
        </div>
        <div class="empty" style="padding:80px">
          <div class="empty-icon">◌</div>
          <div class="empty-msg">No students assigned yet.</div>
        </div>"""
        return page("My Students", content, "teacher", session["name"], "students")

    group_html = ""
    for (semester, block), students in groups.items():
        evals_done   = sum(1 for s in students if s["eval_count"] and s["eval_count"] > 0)
        avg_of_avgs  = [s["avg_score"] for s in students if s["avg_score"]]
        block_avg    = round(sum(avg_of_avgs)/len(avg_of_avgs), 2) if avg_of_avgs else None
        group_id     = f"grp_{hash((semester, block)) & 0xFFFFFF}"

        student_rows = ""
        for s in students:
            avg_disp = f"{round(s['avg_score'],2)}/5" if s["avg_score"] else "—"
            eval_badge = (f'<span class="badge badge-student">{s["eval_count"]} eval{"s" if s["eval_count"]!=1 else ""}</span>'
                         if s["eval_count"] else '<span class="badge" style="background:rgba(122,80,85,0.15);color:var(--muted);border-color:rgba(122,80,85,0.2)">No evals</span>')
            student_rows += f"""
            <tr>
              <td>
                <div style="display:flex;align-items:center;gap:10px">
                  <div style="width:30px;height:30px;border-radius:50%;background:linear-gradient(135deg,var(--crimson-dk),var(--surface3));display:flex;align-items:center;justify-content:center;font-size:0.72rem;font-weight:700;color:var(--blush);flex-shrink:0;">
                    {"".join(w[0].upper() for w in s['name'].split()[:2])}
                  </div>
                  <div>
                    <div style="font-weight:600;font-size:0.92rem">{s['name']}</div>
                    <div style="font-family:'JetBrains Mono',monospace;font-size:0.62rem;color:var(--muted)">{s['email']}</div>
                  </div>
                </div>
              </td>
              <td class="text-muted" style="font-size:0.85rem">{s['department'] or '—'}</td>
            </tr>"""

        group_html += f"""
        <div class="block-group">
          <div class="block-group-header" onclick="toggleBlock('{group_id}')">
            <div class="block-title">
              <span style="color:var(--crimson)">⬡</span>
              {block}
              <span class="badge badge-block">{len(students)} student{'s' if len(students)!=1 else ''}</span>
              {f'<span class="badge badge-good" style="font-size:0.58rem">Avg {block_avg}/5</span>' if block_avg else ''}
            </div>
            <div style="display:flex;align-items:center;gap:12px">
              <div class="block-meta">{semester}</div>
              <div class="block-meta">{evals_done}/{len(students)} evaluated</div>
              <span id="arrow-{group_id}" style="color:var(--muted);font-size:0.8rem">▾</span>
            </div>
          </div>
          <div id="body-{group_id}" class="block-body" style="display:none">
            <div class="table-wrap" style="margin-top:12px;border:none;border-radius:8px">
              <table>
                <thead><tr>
                  <th>Student</th><th>Dept</th>
                </tr></thead>
                <tbody>{student_rows}</tbody>
              </table>
            </div>
          </div>
        </div>"""

    content = f"""
    <div class="page-header">
      <div class="page-title">My <span class="page-title-accent">Students</span></div>
      <div class="page-subtitle">Students assigned to you, grouped by block and semester</div>
    </div>
    <div class="cards-grid" style="margin-bottom:28px">
      <div class="stat-card crimson"><div class="stat-icon">◉</div><div class="stat-value">{total_students}</div><div class="stat-label">Total Students</div></div>
      <div class="stat-card gold"><div class="stat-icon">⬡</div><div class="stat-value">{total_blocks}</div><div class="stat-label">Block / Sections</div></div>
    </div>
    <div style="margin-bottom:12px;display:flex;gap:10px;flex-wrap:wrap">
      <button class="btn btn-secondary btn-sm" onclick="expandAll()">▾ Expand All</button>
      <button class="btn btn-secondary btn-sm" onclick="collapseAll()">▴ Collapse All</button>
    </div>
    {group_html}
    <script>
    const allGroups = {json.dumps([f"grp_{hash((sem, blk)) & 0xFFFFFF}" for (sem, blk) in groups.keys()])};
    function toggleBlock(id) {{
      const body  = document.getElementById('body-'+id);
      const arrow = document.getElementById('arrow-'+id);
      const open  = body.style.display !== 'none';
      body.style.display  = open ? 'none' : 'block';
      arrow.textContent   = open ? '▾' : '▴';
    }}
    function expandAll()  {{ allGroups.forEach(id => {{ document.getElementById('body-'+id).style.display='block';  document.getElementById('arrow-'+id).textContent='▴'; }}); }}
    function collapseAll(){{ allGroups.forEach(id => {{ document.getElementById('body-'+id).style.display='none';   document.getElementById('arrow-'+id).textContent='▾'; }}); }}
    </script>"""
    return page("My Students", content, "teacher", session["name"], "students")

# ════════════════════════════════════════════════════════════════
#  STUDENT ROUTES
# ════════════════════════════════════════════════════════════════
@app.route("/student/dashboard")
@login_required
@role_required("student")
def student_dashboard():
    sid = session["user_id"]; conn = get_db()
    my_evals = conn.execute("SELECT COUNT(DISTINCT teacher_id) FROM evaluation WHERE student_id=?", (sid,)).fetchone()[0]
    teachers_eval  = conn.execute("SELECT COUNT(DISTINCT teacher_id) FROM evaluation WHERE student_id=?", (sid,)).fetchone()[0]
    my_teachers    = conn.execute("SELECT COUNT(*) FROM student_teacher WHERE student_id=?", (sid,)).fetchone()[0]
    user_info      = conn.execute("SELECT block, semester FROM users WHERE id=?", (sid,)).fetchone()
    conn.close()
    block_info = ""
    if user_info and user_info["block"]:
        block_info = f"""
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:20px;padding:12px 16px;background:rgba(220,20,60,0.07);border:1px solid rgba(220,20,60,0.18);border-radius:10px;">
          <span style="color:var(--crimson);font-size:1.1rem">⬡</span>
          <div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.62rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.1em">My Block</div>
            <div style="font-weight:700;font-size:0.95rem">{user_info['block']} &nbsp;<span style="font-size:0.8rem;color:var(--muted)">·</span>&nbsp; <span style="font-size:0.85rem;color:var(--text-dim)">{user_info['semester'] or ''}</span></div>
          </div>
        </div>"""
    content = f"""
    <div class="page-header"><div class="page-title">Student <span class="page-title-accent">Portal</span></div><div class="page-subtitle">Evaluate your teachers and track your submissions</div></div>
    {block_info}
    <div class="cards-grid">
      <div class="stat-card crimson"><div class="stat-icon">✦</div><div class="stat-value">{my_evals}</div><div class="stat-label">Submissions</div></div>
      <div class="stat-card gold"><div class="stat-icon">◈</div><div class="stat-value">{teachers_eval}</div><div class="stat-label">Teachers Evaluated</div></div>
      <div class="stat-card rose"><div class="stat-icon">◉</div><div class="stat-value">{my_teachers}</div><div class="stat-label">My Teachers</div></div>
    </div>
    <div class="result-card">
      <div class="section-title" style="margin-bottom:12px">Ready to Evaluate?</div>
      <p style="color:var(--text-dim);font-size:0.95rem;margin-bottom:18px;line-height:1.7;font-family:'Crimson Pro',serif;font-style:italic">
        Your honest feedback helps improve teaching quality. All submissions remain completely anonymous.
      </p>
      <a href="/student/evaluate" class="btn btn-primary">＋ &nbsp;Start Evaluation</a>
    </div>"""
    return page("Dashboard", content, "student", session["name"], "dashboard")

@app.route("/student/evaluate", methods=["GET","POST"])
@login_required
@role_required("student")
def student_evaluate():
    sid = session["user_id"]; conn = get_db()

    if request.method == "POST":
        teacher_id = request.form.get("teacher_id","").strip()
        scores = request.form.getlist("scores[]"); qids = request.form.getlist("qids[]"); comments = request.form.getlist("comments[]")

        if not teacher_id:
            flash("Please select a teacher.", "danger")
        elif not scores or not qids:
            flash("Please rate all questions.", "danger")
        else:
            # ── ONE EVALUATION PER STUDENT PER TEACHER ──
            student_semester = conn.execute(
    "SELECT semester FROM users WHERE id=?", (sid,)
).fetchone()["semester"] or ""
            already = conn.execute("""
                SELECT COUNT(*) FROM evaluation e
    JOIN student_teacher st ON st.teacher_id = e.teacher_id AND st.student_id = e.student_id
    WHERE e.student_id=? AND e.teacher_id=? AND st.semester=?
""", (sid, int(teacher_id), student_semester)
            ).fetchone()[0]

            if already > 0:
                conn.close()
                flash("⚠ You have already submitted an evaluation for this teacher. Each teacher can only be evaluated once.", "warning")
                return redirect(url_for("student_evaluate"))

            try:
                for i, (qid, score) in enumerate(zip(qids, scores)):
                    comment = comments[i] if i < len(comments) else ""
                    conn.execute("INSERT INTO evaluation(teacher_id,student_id,question_id,score,comment)VALUES(?,?,?,?,?)",
                                 (int(teacher_id), sid, int(qid), int(score), comment))
                conn.commit(); conn.close()
                recalc_teacher(int(teacher_id))
                success = f"""<div style="display:flex;align-items:center;justify-content:center;min-height:60vh"><div style="text-align:center;max-width:440px;animation:fadeSlideUp 0.5s ease">
                  <div style="width:72px;height:72px;border-radius:50%;background:linear-gradient(135deg,var(--crimson-dk),var(--crimson));display:flex;align-items:center;justify-content:center;margin:0 auto 24px;font-size:2rem;box-shadow:0 0 32px rgba(220,20,60,0.4)">✓</div>
                  <h2 style="font-size:1.7rem;font-weight:900;font-family:'Playfair Display',serif;margin-bottom:12px">Evaluation Submitted</h2>
                  <p style="color:var(--text-dim);font-size:0.95rem;line-height:1.8;margin-bottom:28px;font-style:italic">Thank you! Your feedback has been processed by the ML engine. Your identity remains anonymous.</p>
                  <div class="flex gap-2" style="justify-content:center;flex-wrap:wrap">
                    <a href="/student/evaluate" class="btn btn-primary">Evaluate Another</a>
                    <a href="/student/history" class="btn btn-secondary">View My Submissions</a>
                  </div></div></div>"""
                return page("Submitted!", success, "student", session["name"], "evaluate")
            except Exception as e:
                conn.close(); flash(f"Error: {str(e)}", "danger")

    # ── GET: build form ──
    # Teachers already evaluated by this student
    student_semester = conn.execute(
    "SELECT semester FROM users WHERE id=?", (sid,)
).fetchone()["semester"] or ""
    
    evaluated_teacher_ids = set(
       r["teacher_id"] for r in conn.execute("""
        SELECT DISTINCT e.teacher_id FROM evaluation e
        JOIN student_teacher st ON st.teacher_id = e.teacher_id AND st.student_id = e.student_id
        WHERE e.student_id=? AND st.semester=?
    """, (sid, student_semester)).fetchall()
)
    my_teacher_ids = set(
        r["teacher_id"] for r in conn.execute(
            "SELECT teacher_id FROM student_teacher WHERE student_id=? AND student_id != 0", (sid,)
        ).fetchall()
    )
    all_teachers   = conn.execute("SELECT id,name,department FROM users WHERE role='teacher' ORDER BY name").fetchall()
    questions      = conn.execute("SELECT * FROM question ORDER BY id").fetchall()
    conn.close()

    my_teachers_list    = [t for t in all_teachers if t["id"] in my_teacher_ids]
    other_teachers_list = [t for t in all_teachers if t["id"] not in my_teacher_ids]

    def teacher_opt(t):
        already_done = t["id"] in evaluated_teacher_ids
        disabled = 'disabled' if already_done else ''
        style = 'style="color:var(--muted);font-style:italic"' if already_done else ''
        label = f'{t["name"]}{(" — "+t["department"]) if t["department"] else ""}{"  ✓ Already Evaluated" if already_done else ""}'
        return f'<option value="{t["id"]}" {disabled} {style}>{label}</option>'

    teacher_opts = ""
    if my_teachers_list:
        teacher_opts += '<optgroup label="— My Assigned Teachers —">'
        teacher_opts += "".join(teacher_opt(t) for t in my_teachers_list)
        teacher_opts += "</optgroup>"
    if other_teachers_list:
        teacher_opts += '<optgroup label="— Other Teachers —">'
        teacher_opts += "".join(teacher_opt(t) for t in other_teachers_list)
        teacher_opts += "</optgroup>"

    # Count how many available teachers left
    available_count = sum(1 for t in all_teachers if t["id"] not in evaluated_teacher_ids)
    done_count      = len(evaluated_teacher_ids)

    notice = ""
    if done_count > 0:
        notice = f"""
        <div style="display:flex;align-items:center;gap:10px;padding:12px 16px;background:rgba(245,158,11,0.07);border:1px solid rgba(245,158,11,0.20);border-radius:10px;margin-bottom:20px;font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:var(--amber);">
          ⚠ &nbsp; You have already evaluated <b>{done_count}</b> teacher{'s' if done_count!=1 else ''}. 
          Each teacher can only be evaluated <b>once</b>. 
          <span style="color:var(--muted);margin-left:6px">({available_count} remaining)</span>
        </div>"""

    qs_json = [{"id": q["id"], "text": q["question_text"]} for q in questions]
    content = f"""
    <div class="page-header"><div class="page-title">Evaluate a <span class="page-title-accent">Teacher</span></div>
      <div class="page-subtitle">Anonymous & confidential — one evaluation per teacher</div></div>
    {notice}
    <div class="card" style="max-width:700px">
      <form method="POST" id="evalForm">
        <div class="form-group"><label>Select Teacher</label>
          <select name="teacher_id" required id="teacherSelect">
            <option value="">— Choose a teacher —</option>{teacher_opts}
          </select>
        </div>
        <hr class="separator">
        <div id="questionsArea">
          <div class="empty" style="padding:32px"><div class="empty-icon" style="font-size:1.8rem">◇</div>
            <div class="empty-msg">Select a teacher above to load evaluation questions</div></div>
        </div>
        <div id="submitArea" style="display:none;margin-top:8px">
          <button type="submit" class="btn btn-primary" style="padding:11px 28px">Submit Evaluation &nbsp;→</button>
        </div>
      </form>
    </div>"""
    js_extra = f"""<script>
    const questions={json.dumps(qs_json)};
    const select=document.getElementById('teacherSelect');
    const area=document.getElementById('questionsArea');
    const submitArea=document.getElementById('submitArea');
    select.addEventListener('change',()=>{{
      if(!select.value||select.options[select.selectedIndex].disabled){{
        area.innerHTML='<div class="empty" style="padding:32px"><div class="empty-icon" style="font-size:1.8rem">◇</div><div class="empty-msg">Select an available teacher above</div></div>';
        submitArea.style.display='none';return;
      }}
      let html='';
      questions.forEach((q,i)=>{{
        html+=`<div style="margin-bottom:28px;padding-bottom:24px;border-bottom:1px solid var(--border)">
          <div style="font-family:'Crimson Pro',serif;font-size:1rem;color:var(--text);margin-bottom:14px;line-height:1.5">
            <span style="color:var(--crimson);font-family:'JetBrains Mono',monospace;font-size:0.75rem;margin-right:8px">${{String(i+1).padStart(2,'0')}}</span>${{q.text}}</div>
          <input type="hidden" name="qids[]" value="${{q.id}}">
          <input type="hidden" name="scores[]" id="score_${{q.id}}" value="0">
          <div class="stars" data-input="score_${{q.id}}">
            <span class="star">★</span><span class="star">★</span><span class="star">★</span><span class="star">★</span><span class="star">★</span>
          </div>
          <div style="margin-top:12px"><input type="text" name="comments[]" placeholder="Optional comment (helps ML accuracy)..." style="font-family:'Crimson Pro',serif"></div>
        </div>`;
      }});
      area.innerHTML=html;submitArea.style.display='block';
      area.querySelectorAll('.stars').forEach(starsEl=>{{
        const inp=document.getElementById(starsEl.dataset.input);
        const stars=starsEl.querySelectorAll('.star');
        stars.forEach((s,i)=>{{
          s.addEventListener('click',()=>{{if(inp)inp.value=i+1;stars.forEach((ss,j)=>ss.classList.toggle('filled',j<=i));}});
          s.addEventListener('mouseenter',()=>stars.forEach((ss,j)=>ss.classList.toggle('filled',j<=i)));
        }});
        starsEl.addEventListener('mouseleave',()=>{{const v=inp?parseInt(inp.value)||0:0;stars.forEach((ss,j)=>ss.classList.toggle('filled',j<v));}});
      }});
    }});
    document.getElementById('evalForm').addEventListener('submit',e=>{{
      const scores=document.querySelectorAll('[name="scores[]"]');
      for(let s of scores){{if(parseInt(s.value)<1){{e.preventDefault();alert('Please rate all questions.');return;}}}}
    }});
    </script>"""
    return page("Evaluate", content + js_extra, "student", session["name"], "evaluate")

@app.route("/student/history")
@login_required
@role_required("student")
def student_history():
    sid = session["user_id"]; conn = get_db()
    submissions = conn.execute("""
    SELECT e.id, u.name as teacher_name, u.id as teacher_id,
           q.question_text, e.score, e.comment, e.created_at,
           COALESCE(st.semester, '') as semester
    FROM evaluation e
    JOIN users u ON e.teacher_id=u.id
    JOIN question q ON e.question_id=q.id
    LEFT JOIN student_teacher st ON st.teacher_id=e.teacher_id AND st.student_id=e.student_id
    WHERE e.student_id=? ORDER BY st.semester DESC, u.name, e.created_at DESC
""", (sid,)).fetchall()
    
    conn.close()
    from collections import OrderedDict
    teachers = OrderedDict()
    for s in submissions:
        tid = s["teacher_id"]
        if tid not in teachers:
            teachers[tid] = {"name": s["teacher_name"], "evals": [], "total": 0, "score_sum": 0}
        teachers[tid]["evals"].append(s)
        teachers[tid]["total"] += 1
        teachers[tid]["score_sum"] += s["score"]
    summary_rows = ""; detail_sections = ""
    for tid, data in teachers.items():
        avg = round(data["score_sum"]/data["total"], 1)
        stars_avg = "★"*round(avg) + "☆"*(5-round(avg))
        latest_date = data["evals"][0]["created_at"][:10]
        semester_tag = data["evals"][0]["semester"]
        summary_rows += f"""
        <tr id="summary-{tid}">
          <td style="font-family:'JetBrains Mono',monospace;font-size:0.75rem;color:var(--muted)">◈</td>
          <td><b>{data['name']}</b></td>
          <td style="color:var(--gold)">{stars_avg} <span class="text-muted">({avg}/5)</span></td>
          <td style="font-family:'JetBrains Mono',monospace;font-size:0.78rem;color:var(--muted)">{data['total']} eval{'s' if data['total']!=1 else ''}</td>
          <td class="text-muted" style="font-family:'JetBrains Mono',monospace;font-size:0.75rem">{latest_date}</td>
          <td><span class="badge badge-block" style="font-size:0.58rem">{semester_tag or '—'}</span></td>
          <td><button class="btn btn-secondary btn-sm" onclick="toggleDetails({tid})">▾ &nbsp;View All</button></td>
        </tr>
        <tr id="details-{tid}" style="display:none"><td colspan="6" style="padding:0">
          <div style="background:var(--surface2);border-top:1px solid var(--border);padding:16px 20px">
            <table style="width:100%;border-collapse:collapse"><thead><tr style="border-bottom:1px solid var(--border)">
              <th style="padding:8px 12px;text-align:left;font-family:'JetBrains Mono',monospace;font-size:0.58rem;color:var(--muted);text-transform:uppercase">#</th>
              <th style="padding:8px 12px;text-align:left;font-family:'JetBrains Mono',monospace;font-size:0.58rem;color:var(--muted);text-transform:uppercase">Question</th>
              <th style="padding:8px 12px;text-align:left;font-family:'JetBrains Mono',monospace;font-size:0.58rem;color:var(--muted);text-transform:uppercase">Score</th>
              <th style="padding:8px 12px;text-align:left;font-family:'JetBrains Mono',monospace;font-size:0.58rem;color:var(--muted);text-transform:uppercase">Comment</th>
              <th style="padding:8px 12px;text-align:left;font-family:'JetBrains Mono',monospace;font-size:0.58rem;color:var(--muted);text-transform:uppercase">Date</th>
            </tr></thead><tbody>"""
        for e in data["evals"]:
            detail_sections += f"""<tr style="border-bottom:1px solid rgba(61,26,29,0.3)">
                <td style="padding:10px 12px;font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:var(--muted)">#{e['id']}</td>
                <td style="padding:10px 12px;max-width:260px;font-size:0.88rem;font-family:'Crimson Pro',serif">{e['question_text']}</td>
                <td style="padding:10px 12px;color:var(--gold)">{'★'*e['score']}{'☆'*(5-e['score'])} <span style="color:var(--muted);font-size:0.8rem">({e['score']}/5)</span></td>
                <td style="padding:10px 12px;color:var(--text-dim);font-size:0.88rem">{e['comment'] or '—'}</td>
                <td style="padding:10px 12px;font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:var(--muted)">{e['created_at'][:10]}</td>
                </tr>"""
        detail_sections += "</tbody></table></div></td></tr>"
    content = f"""
    <div class="page-header"><div class="page-title">My <span class="page-title-accent">Submissions</span></div><div class="page-subtitle">One row per teacher — click View All to expand</div></div>
    <div class="table-wrap"><table>
      <thead><tr><th></th><th>Teacher</th><th>Avg Score</th><th>Count</th><th>Latest</th><th>Semester</th><th>Action</th></tr></thead>
      <tbody>{'<tr><td colspan="7"><div class="empty"><div class="empty-icon">◌</div><div class="empty-msg">No submissions yet. <a href="/student/evaluate" style="color:var(--rose)">Start evaluating →</a></div></div></td></tr>' if not teachers else summary_rows + detail_sections}</tbody>
    </table></div>
    <script>function toggleDetails(tid){{const row=document.getElementById('details-'+tid);const btn=document.querySelector('#summary-'+tid+' button');const isOpen=row.style.display!=='none';row.style.display=isOpen?'none':'table-row';btn.innerHTML=isOpen?'▾ &nbsp;View All':'▴ &nbsp;Collapse';}}</script>"""
    return page("My Submissions", content, "student", session["name"], "history")

# ════════════════════════════════════════════════════════════════
#  Profiles
# ════════════════════════════════════════════════════════════════    
@app.route("/profile")
@login_required
def profile():
    conn = get_db()
    user = conn.execute("SELECT * FROM users WHERE id=?", (session["user_id"],)).fetchone()
    conn.close()
    
    block_row = f"""
    <tr><td style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:var(--muted);padding:10px 0">Block / Section</td>
    <td style="padding:10px 0;font-weight:600">{user['block'] or '—'}</td></tr>
    <tr><td style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:var(--muted);padding:10px 0">Semester</td>
    <td style="padding:10px 0">{user['semester'] or '—'}</td></tr>
    <tr><td style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:var(--muted);padding:10px 0">Status</td>
    <td style="padding:10px 0">
    <span class="badge" style="{'background:rgba(201,150,58,0.15);color:var(--gold-lt);border-color:rgba(201,150,58,0.2)' if (user['status'] or 'Regular')=='Regular' else 'background:rgba(251,146,60,0.12);color:#fb923c;border-color:rgba(251,146,60,0.2)'}">{user['status'] or 'Regular'}</span>
    </td></tr>""" if user['role'] == 'student' else ""

    dept_row = f"""
    <tr><td style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:var(--muted);padding:10px 0">Department</td>
    <td style="padding:10px 0">{user['department'] or '—'}</td></tr>""" if user['role'] in ('teacher','admin') else ""

    initials = "".join(w[0].upper() for w in user['name'].split()[:2])

    content = f"""
    <div class="page-header">
      <div class="page-title">My <span class="page-title-accent">Profile</span></div>
      <div class="page-subtitle">Your account information</div>
    </div>
    <div class="card" style="max-width:520px">
      <div style="display:flex;align-items:center;gap:18px;margin-bottom:28px;padding-bottom:24px;border-bottom:1px solid var(--border)">
        <div style="width:72px;height:72px;border-radius:50%;background:linear-gradient(135deg,var(--crimson-dk),var(--crimson));display:flex;align-items:center;justify-content:center;font-weight:900;font-size:1.5rem;color:#fff;font-family:'Playfair Display',serif;flex-shrink:0;box-shadow:0 0 24px rgba(220,20,60,0.35)">{initials}</div>
        <div>
          <div style="font-family:'Playfair Display',serif;font-size:1.4rem;font-weight:900;color:var(--text)">{user['name']}</div>
          <span class="badge badge-{user['role']}" style="margin-top:6px">{user['role'].upper()}</span>
        </div>
      </div>
      <table style="width:100%;border-collapse:collapse">
        <tr><td style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:var(--muted);padding:10px 0">Full Name</td>
          <td style="padding:10px 0;font-weight:600">{user['name']}</td></tr>
        <tr style="border-top:1px solid rgba(61,26,29,0.3)">
          <td style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:var(--muted);padding:10px 0">Email</td>
          <td style="padding:10px 0;color:var(--rose)">{user['email']}</td></tr>
        <tr style="border-top:1px solid rgba(61,26,29,0.3)">
          <td style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:var(--muted);padding:10px 0">Role</td>
          <td style="padding:10px 0"><span class="badge badge-{user['role']}">{user['role']}</span></td></tr>
        {'<tr style="border-top:1px solid rgba(61,26,29,0.3)">'+dept_row+'</tr>' if dept_row else ''}
        {'<tr style="border-top:1px solid rgba(61,26,29,0.3)">'+block_row+'</tr>' if block_row else ''}
        <tr style="border-top:1px solid rgba(61,26,29,0.3)">
          <td style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:var(--muted);padding:10px 0">Member Since</td>
          <td style="padding:10px 0;font-family:'JetBrains Mono',monospace;font-size:0.8rem;color:var(--text-dim)">{user['created_at'][:10]}</td></tr>
      </table>
    </div>"""
    return page("My Profile", content, user['role'], user['name'], "")

# ════════════════════════════════════════════════════════════════
#  ERROR HANDLERS
# ════════════════════════════════════════════════════════════════
@app.errorhandler(404)
def not_found(e):
    return f"""<!DOCTYPE html><html><head><style>{BASE_CSS}</style></head><body><div style="display:flex;align-items:center;justify-content:center;min-height:100vh;flex-direction:column;gap:16px;text-align:center"><h1 style="font-family:'Playfair Display',serif;color:var(--crimson)">404 — Not Found</h1><a href="/" class="btn btn-primary">Go Home</a></div></body></html>""", 404

@app.errorhandler(500)
def server_error(e):
    return f"""<!DOCTYPE html><html><head><style>{BASE_CSS}</style></head><body><div style="display:flex;align-items:center;justify-content:center;min-height:100vh;flex-direction:column;gap:16px;text-align:center"><h1 style="font-family:'Playfair Display',serif">500 — Server Error</h1><p style="color:var(--muted)">{str(e)}</p><a href="/" class="btn btn-primary">Go Home</a></div></body></html>""", 500

# ════════════════════════════════════════════════════════════════
#  RUN
# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Training ML model (scikit-learn TF-IDF + Logistic Regression)...")
    get_classifier()
    print("ML model ready.")
    init_db()
port = int(os.environ.get("PORT", 5000))
app.run(host="0.0.0.0", port=port)
#app.run(debug=True, port=5000, threaded=True, use_reloader=False)
