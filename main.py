#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bountybot (A-mode): "escrow-ish" / explicit pay only -> reproduce test -> fix -> PR
- Source: GitHub Search (prioritize Algora-like signals via keywords)
- Filtering: DeepSeek JSON classifier (fallback heuristic)
- Safety: strict thresholds, baseline tests must pass, PR per run capped

Commands:
  python main.py run      # collect -> rescore -> queue
  python main.py work     # process queued -> attempt PRs (strict)
  python main.py monitor  # monitor outcomes (learning)
  python main.py report   # dashboard
"""

from __future__ import annotations

import os
import re
import json
import time
import math
import random
import shutil
import sqlite3
import hashlib
import logging
import subprocess
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

import requests

# ---------------- Config ----------------
DB_PATH = os.getenv("DB_PATH", "bountybot.sqlite").strip()

GITHUB_TOKEN = os.getenv("BOUNTYBOT_GITHUB_TOKEN", "").strip() or os.getenv("GITHUB_TOKEN", "").strip()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat").strip() or "deepseek-chat"

LOOKBACK_DAYS = int(os.getenv("LOOKBACK_DAYS", "21"))
MAX_SEARCH_PAGES = int(os.getenv("MAX_SEARCH_PAGES", "3"))
PER_PAGE = int(os.getenv("PER_PAGE", "50"))
MAX_CANDIDATES_PER_RUN = int(os.getenv("MAX_CANDIDATES_PER_RUN", "250"))

PICK_TOP_N = int(os.getenv("PICK_TOP_N", "20"))
EPSILON = float(os.getenv("EPSILON", "0.10"))
MIN_PAYMENT_STRENGTH = float(os.getenv("MIN_PAYMENT_STRENGTH", "0.85"))  # A-mode: strict

# PR automation caps
BOT_MAX_PR_PER_RUN = int(os.getenv("BOT_MAX_PR_PER_RUN", "1"))  # keep low to protect reputation
WORK_TIMEOUT_SEC = int(os.getenv("WORK_TIMEOUT_SEC", "900"))
WORK_MAX_AUTOFIX = int(os.getenv("WORK_MAX_AUTOFIX", "2"))

# Monitoring / scoring
MONITOR_DAYS = int(os.getenv("MONITOR_DAYS", "45"))
STALE_DAYS_PENALTY = float(os.getenv("STALE_DAYS_PENALTY", "0.02"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("bountybot")


# ---------------- Utils ----------------
def utc_now() -> datetime:
    return datetime.now(timezone.utc)

def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def clamp(x: float, a: float, b: float) -> float:
    return max(a, min(b, x))

def run_cmd(cmd: List[str], cwd: Optional[str] = None, timeout: int = 600) -> Tuple[int, str]:
    p = subprocess.run(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    return p.returncode, (p.stdout or "").strip()


# ---------------- GitHub API ----------------
def gh_headers() -> Dict[str, str]:
    if not GITHUB_TOKEN:
        raise SystemExit("Missing token. Set BOUNTYBOT_GITHUB_TOKEN secret.")
    return {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "bountybot/1.0",
    }

def gh_get(url: str, params: Optional[Dict[str, Any]] = None) -> Any:
    r = requests.get(url, headers=gh_headers(), params=params, timeout=45)
    r.raise_for_status()
    return r.json()

def gh_post(url: str, payload: Dict[str, Any]) -> Any:
    r = requests.post(url, headers=gh_headers(), json=payload, timeout=45)
    r.raise_for_status()
    return r.json()


# ---------------- DB ----------------
SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS candidates (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  issue_url TEXT UNIQUE,
  repo_full_name TEXT,
  issue_number INTEGER,
  title TEXT,
  body_snip TEXT,
  labels_json TEXT,
  state TEXT,
  created_at TEXT,
  updated_at TEXT,
  comments_count INTEGER,
  payment_claim TEXT,         -- yes/no/ambiguous
  payment_strength REAL,      -- 0..1
  payment_amount_usd REAL,    -- guessed
  payout_hint TEXT,           -- "algora"/"gitcoin"/...
  language_hint TEXT,
  arm_key TEXT,
  base_score REAL,
  final_score REAL,
  last_scored_at TEXT
);

CREATE TABLE IF NOT EXISTS attempts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  candidate_id INTEGER,
  status TEXT,                -- queued/working/pr_opened/merged/rejected/ignored
  notes TEXT,
  pr_url TEXT,
  fork_full_name TEXT,
  branch_name TEXT,
  created_at TEXT,
  updated_at TEXT,
  FOREIGN KEY(candidate_id) REFERENCES candidates(id)
);

CREATE TABLE IF NOT EXISTS rewards (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  attempt_id INTEGER,
  reward_value REAL,
  reward_reason TEXT,
  created_at TEXT,
  FOREIGN KEY(attempt_id) REFERENCES attempts(id)
);

CREATE TABLE IF NOT EXISTS arms (
  arm_key TEXT PRIMARY KEY,
  n INTEGER NOT NULL,
  mean_reward REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_candidates_updated ON candidates(updated_at);
CREATE INDEX IF NOT EXISTS idx_attempts_status ON attempts(status);
"""

def db() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.executescript(SCHEMA)
    return con


# ---------------- DeepSeek ----------------
def deepseek_chat_json(obj: Dict[str, Any], temperature: float = 0.0, timeout: int = 60) -> Dict[str, Any]:
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("DEEPSEEK_API_KEY not set.")
    url = f"{DEEPSEEK_BASE_URL}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": "Return ONLY valid JSON. No markdown."},
            {"role": "user", "content": json.dumps(obj, ensure_ascii=False)},
        ],
        "temperature": temperature,
    }
    r = requests.post(url, headers=headers, json=payload, timeout=timeout)
    r.raise_for_status()
    content = r.json()["choices"][0]["message"]["content"].strip()
    return json.loads(content)


# ---------------- Payment detection ----------------
PAYMENT_KEYWORDS = [
    "bounty", "reward", "paid", "payout", "payment", "cash",
    "gitcoin", "opencollective", "bountysource", "issuehunt",
    "algora", "algora.io",
    "usdc", "usdt", "paypal", "stripe",
]
NEGATIVE_HINTS = [
    "no bounty", "not a bounty", "no reward", "unpaid",
    "volunteer", "donation", "tips welcome", "sponsor only",
]

MONEY_RE = re.compile(r"(?:(?:\$|usd\s*)\s*([0-9]+(?:\.[0-9]+)?))|(?:([0-9]+(?:\.[0-9]+)?)\s*usd)", re.I)
CRYPTO_RE = re.compile(r"\b([0-9]+(?:\.[0-9]+)?)\s*(usdc|usdt)\b", re.I)

def guess_amount_usd(text: str) -> Optional[float]:
    m = MONEY_RE.search(text)
    if m:
        v = m.group(1) or m.group(2)
        try:
            return float(v)
        except Exception:
            return None
    c = CRYPTO_RE.search(text)
    if c:
        try:
            return float(c.group(1))
        except Exception:
            return None
    return None

def guess_payout_hint(text: str) -> str:
    t = text.lower()
    if "algora" in t:
        return "algora"
    for k in ("gitcoin", "opencollective", "issuehunt", "bountysource", "paypal", "stripe", "usdc", "usdt"):
        if k in t:
            return k
    return ""

def heuristic_payment_classify(title: str, body: str, comments: str) -> Tuple[str, float, Optional[float], str]:
    text = f"{title}\n{body}\n{comments}".strip()
    tl = text.lower()

    for neg in NEGATIVE_HINTS:
        if neg in tl:
            return ("no", 1.0, None, guess_payout_hint(text))

    kw_hits = sum(1 for k in PAYMENT_KEYWORDS if k in tl)
    amt = guess_amount_usd(text)
    payout = guess_payout_hint(text)

    strength = 0.0
    if kw_hits >= 1:
        strength += 0.35
    if kw_hits >= 3:
        strength += 0.20
    if amt is not None:
        strength += 0.35
    if payout:
        strength += 0.20  # escrow-ish hint boosts

    strength = clamp(strength, 0.0, 1.0)
    if strength >= 0.75:
        return ("yes", strength, amt, payout)
    if strength <= 0.35:
        return ("no", strength, amt, payout)
    return ("ambiguous", strength, amt, payout)

def deepseek_payment_classify(title: str, body: str, comments: str) -> Tuple[str, float, Optional[float], str]:
    if not DEEPSEEK_API_KEY:
        return heuristic_payment_classify(title, body, comments)

    prompt = {
        "task": "Detect whether this issue has explicitly stated payment/bounty/reward with reliable terms.",
        "output_schema": {
            "payment_claim": "yes|no|ambiguous",
            "payment_strength": "0..1",
            "amount_usd_guess": "number|null",
            "payout_hint": "algora|gitcoin|opencollective|issuehunt|bountysource|paypal|stripe|usdc|usdt|''"
        },
        "rules": [
            "If it says tips/sponsor/donation without explicit reward amount+conditions => NO.",
            "If it says no bounty/no reward => NO (strength 1.0).",
            "If it includes explicit amount and condition like '$100 upon merge' => YES with high strength.",
            "If it references Algora bounties, set payout_hint='algora'.",
            "Return JSON only."
        ],
        "input": {
            "title": title[:300],
            "body": body[:2400],
            "comments": comments[:1800],
        }
    }
    try:
        d = deepseek_chat_json(prompt, temperature=0.0, timeout=60)
        claim = d.get("payment_claim", "ambiguous")
        strength = float(d.get("payment_strength", 0.0))
        amt = d.get("amount_usd_guess", None)
        payout = (d.get("payout_hint", "") or "").strip().lower()
        if amt is not None:
            try:
                amt = float(amt)
            except Exception:
                amt = None
        if claim not in ("yes", "no", "ambiguous"):
            claim = "ambiguous"
        return (claim, clamp(strength, 0.0, 1.0), amt, payout)
    except Exception as e:
        log.warning(f"DeepSeek classify failed -> heuristic fallback: {e}")
        return heuristic_payment_classify(title, body, comments)


# ---------------- Search / repo signals ----------------
def search_issues(query: str, page: int, per_page: int) -> Dict[str, Any]:
    url = "https://api.github.com/search/issues"
    try:
        return gh_get(url, params={"q": query, "page": page, "per_page": per_page, "sort": "updated", "order": "desc"})
    except requests.exceptions.HTTPError as e:
        # GitHub Searchはクエリが1文字でもダメだと422で落ちる → そのクエリは捨てて継続
        resp = getattr(e, "response", None)
        code = getattr(resp, "status_code", None)
        if code == 422:
            log.warning(f"Search query rejected (422). Skip this query. q={query!r}")
            return {"items": []}
        raise


def issue_comments(repo_full: str, number: int, per_page: int = 30) -> List[Dict[str, Any]]:
    url = f"https://api.github.com/repos/{repo_full}/issues/{number}/comments"
    return gh_get(url, params={"per_page": per_page, "page": 1})

def issue_labels(item: Dict[str, Any]) -> List[str]:
    out = []
    for l in item.get("labels", []) or []:
        if isinstance(l, dict) and l.get("name"):
            out.append(str(l["name"]))
        elif isinstance(l, str):
            out.append(l)
    return out

def guess_language(repo_full: str) -> str:
    try:
        d = gh_get(f"https://api.github.com/repos/{repo_full}")
        return (d.get("language") or "").lower()
    except Exception:
        return ""

# A-mode search: prioritize escrow-ish hints
SEARCH_QUERIES = [
    # 金額表現を "USD/USDC/USDT" に寄せる（$はAPIで422になりやすい）
    'is:issue is:open (bounty OR reward OR payout OR paid) (USD OR USDC OR USDT) in:title,body',
    # Algora hint
    'is:issue is:open (algora OR "algora.io") in:title,body,comments',
    # Known bounty label
    'is:issue is:open label:bounty',
    # Broad fallback (still filtered strictly later)
    'is:issue is:open (bounty OR reward OR payout OR paid) in:title,body',
]




# ---------------- Bandit ----------------
def arm_key(language_hint: str, payout_hint: str, has_amount: bool) -> str:
    lang = (language_hint or "unknown").lower()
    pay = (payout_hint or "unknown").lower()
    amt = "amt" if has_amount else "noamt"
    return f"{lang}:{pay}:{amt}"

def get_arm(con: sqlite3.Connection, key: str) -> Tuple[int, float]:
    row = con.execute("SELECT n, mean_reward FROM arms WHERE arm_key=?", (key,)).fetchone()
    if not row:
        con.execute("INSERT OR IGNORE INTO arms(arm_key,n,mean_reward) VALUES(?,?,?)", (key, 0, 0.0))
        con.commit()
        return (0, 0.0)
    return (int(row[0]), float(row[1]))

def total_arm_counts(con: sqlite3.Connection) -> int:
    row = con.execute("SELECT SUM(n) FROM arms").fetchone()
    return int(row[0] or 0)

def ucb_score(n: int, mean: float, total_n: int) -> float:
    if n <= 0:
        return 1.0
    return mean + math.sqrt(2.0 * math.log(max(2, total_n)) / n)

def update_arm(con: sqlite3.Connection, key: str, reward: float) -> None:
    n, mean = get_arm(con, key)
    n2 = n + 1
    mean2 = mean + (reward - mean) / n2
    con.execute("UPDATE arms SET n=?, mean_reward=? WHERE arm_key=?", (n2, mean2, key))
    con.commit()


# ---------------- Scoring ----------------
def compute_base_score(title: str, strength: float, amount_usd: Optional[float], comments_count: int, payout_hint: str) -> float:
    t = (title or "").lower()
    BAD_TITLE = ["implement", "design", "refactor", "rewrite", "migrate", "support", "scheduler"]
    if any(w in t for w in BAD_TITLE):
        return -999.0

    s = 0.0
    s += 2.8 * float(strength or 0.0)
    if amount_usd is not None:
        s += 0.6 * math.log(1.0 + float(amount_usd))
        # ---- amount shaping: favor $50-$200, penalize >$300 ----
        amt = float(amount_usd)
        if 50.0 <= amt <= 200.0:
            s += 2.0
        elif 200.0 < amt <= 300.0:
            s += 0.5
        elif amt > 300.0:
            s -= 1.5

    s += 0.06 * math.log(1.0 + max(0, int(comments_count or 0)))
    # escrow-ish bonus
    if (payout_hint or "").lower() in ("algora", "gitcoin", "opencollective", "issuehunt"):
        s += 0.25
    return s

def recency_penalty(updated_at: str) -> float:
    try:
        upd = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
        days = (utc_now() - upd).total_seconds() / 86400.0
        return -STALE_DAYS_PENALTY * max(0.0, days)
    except Exception:
        return 0.0


# ---------------- Pipeline: collect / queue ----------------
def collect(con: sqlite3.Connection) -> int:
    cutoff = (utc_now() - timedelta(days=LOOKBACK_DAYS)).date().isoformat()
    collected = 0

    for q in SEARCH_QUERIES:
        query = f"{q} updated:>={cutoff}"
        log.info(f"Search: {query}")

        for page in range(1, MAX_SEARCH_PAGES + 1):
            data = search_issues(query, page=page, per_page=PER_PAGE)
            items = data.get("items", []) or []
            if not items:
                break

            for it in items:
                if collected >= MAX_CANDIDATES_PER_RUN:
                    break

                issue_url = it.get("html_url", "")
                repo_full = it.get("repository_url", "").replace("https://api.github.com/repos/", "")
                number = int(it.get("number", 0) or 0)
                title = it.get("title", "") or ""
                body = it.get("body", "") or ""
                labels = issue_labels(it)
                state = (it.get("state") or "").lower()
                created_at = it.get("created_at", "") or ""
                updated_at = it.get("updated_at", "") or ""
                comments_count = int(it.get("comments", 0) or 0)

                if not issue_url or not repo_full or number <= 0:
                    continue

                comments_snip = ""
                try:
                    cmts = issue_comments(repo_full, number, per_page=30)
                    comments_snip = "\n".join([(c.get("body") or "")[:500] for c in cmts[:8]])
                except Exception:
                    pass

                lang = guess_language(repo_full)
                claim, strength, amt, payout = deepseek_payment_classify(title, body[:2400], comments_snip)

                akey = arm_key(lang, payout, has_amount=(amt is not None))
                base = compute_base_score(title, strength, amt, comments_count, payout)


                total_n = max(1, total_arm_counts(con))
                n, mean = get_arm(con, akey)
                bandit = ucb_score(n, mean, total_n)

                final = base + 0.35 * bandit + recency_penalty(updated_at)

                con.execute(
                    """
                    INSERT OR IGNORE INTO candidates(
                      issue_url, repo_full_name, issue_number, title, body_snip, labels_json,
                      state, created_at, updated_at, comments_count,
                      payment_claim, payment_strength, payment_amount_usd, payout_hint,
                      language_hint, arm_key, base_score, final_score, last_scored_at
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        issue_url, repo_full, number, title, body[:2400], json.dumps(labels, ensure_ascii=False),
                        state, created_at, updated_at, comments_count,
                        claim, float(strength), float(amt) if amt is not None else None, payout,
                        lang, akey, float(base), float(final), iso(utc_now()),
                    )
                )
                collected += 1

            con.commit()
            if collected >= MAX_CANDIDATES_PER_RUN:
                break

    log.info(f"Collected candidates: {collected}")
    return collected

def rescore_all(con: sqlite3.Connection) -> None:
    rows = con.execute(
        "SELECT id, updated_at, payment_strength, payment_amount_usd, comments_count, payout_hint, language_hint, arm_key FROM candidates"
    ).fetchall()

    total_n = max(1, total_arm_counts(con))
    now = iso(utc_now())

    for cid, updated_at, strength, amt, comments, payout, lang, akey in rows:
        base = compute_base_score(float(strength or 0.0), float(amt) if amt is not None else None, int(comments or 0), str(payout or ""))
        n, mean = get_arm(con, str(akey or "unknown:unknown:noamt"))
        bandit = ucb_score(n, mean, total_n)
        final = base + 0.35 * bandit + recency_penalty(str(updated_at or ""))
        con.execute(
            "UPDATE candidates SET base_score=?, final_score=?, last_scored_at=? WHERE id=?",
            (float(base), float(final), now, int(cid)),
        )
    con.commit()

def pick_queue(con: sqlite3.Connection) -> int:
    rows = con.execute(
        """
        SELECT id, final_score
        FROM candidates
        WHERE state='open' AND payment_claim='yes' AND payment_strength >= ?
        ORDER BY final_score DESC
        LIMIT 300
        """,
        (MIN_PAYMENT_STRENGTH,),
    ).fetchall()

    if not rows:
        log.info("No eligible candidates after strict filter.")
        return 0

    # epsilon-greedy over top pool
    pool = rows[:min(len(rows), 200)]
    chosen_ids: List[int] = []
    for _ in range(min(PICK_TOP_N, len(pool))):
        pick = random.choice(pool) if random.random() < EPSILON else pool[0]
        cid = int(pick[0])
        if cid in chosen_ids:
            continue
        # skip already attempted (except ignored)
        has = con.execute(
            "SELECT 1 FROM attempts WHERE candidate_id=? AND status NOT IN ('ignored') LIMIT 1", (cid,)
        ).fetchone()
        if has:
            continue
        chosen_ids.append(cid)

    for cid in chosen_ids:
        con.execute(
            "INSERT INTO attempts(candidate_id,status,notes,pr_url,fork_full_name,branch_name,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?)",
            (cid, "queued", "", "", "", "", iso(utc_now()), iso(utc_now())),
        )
    con.commit()

    log.info(f"Queued: {len(chosen_ids)}")
    return len(chosen_ids)


# ---------------- Monitor (learning signal) ----------------
def monitor(con: sqlite3.Connection) -> int:
    since = iso(utc_now() - timedelta(days=MONITOR_DAYS))
    rows = con.execute(
        """
        SELECT a.id, a.status, c.repo_full_name, c.issue_number, c.issue_url, c.arm_key
        FROM attempts a
        JOIN candidates c ON c.id=a.candidate_id
        WHERE a.updated_at >= ? AND a.status IN ('queued','working','pr_opened')
        """,
        (since,),
    ).fetchall()

    if not rows:
        return 0

    updated = 0
    for attempt_id, status, repo_full, issue_number, issue_url, akey in rows:
        try:
            issue = gh_get(f"https://api.github.com/repos/{repo_full}/issues/{int(issue_number)}")
        except Exception as e:
            log.warning(f"monitor failed {issue_url}: {e}")
            continue

        state = (issue.get("state") or "").lower()

        payout_found = False
        no_bounty_found = False
        try:
            cmts = issue_comments(repo_full, int(issue_number), per_page=40)
            text = "\n".join([(c.get("body") or "") for c in cmts[-12:]]).lower()
            if any(x in text for x in ("payout sent", "paid out", "payment sent", "reward sent", "bounty paid", "paid via")):
                payout_found = True
            if any(x in text for x in ("no bounty", "not a bounty", "no reward", "unpaid")):
                no_bounty_found = True
        except Exception:
            pass

        if no_bounty_found:
            reward, reason = -1.0, "explicit_no_bounty"
            con.execute("INSERT INTO rewards(attempt_id,reward_value,reward_reason,created_at) VALUES(?,?,?,?)",
                        (attempt_id, reward, reason, iso(utc_now())))
            update_arm(con, akey, reward)
            con.execute("UPDATE attempts SET status=?, notes=?, updated_at=? WHERE id=?",
                        ("rejected", reason, iso(utc_now()), attempt_id))
            updated += 1
            continue

        if payout_found:
            reward, reason = 1.0, "payout_comment_detected"
            con.execute("INSERT INTO rewards(attempt_id,reward_value,reward_reason,created_at) VALUES(?,?,?,?)",
                        (attempt_id, reward, reason, iso(utc_now())))
            update_arm(con, akey, reward)
            con.execute("UPDATE attempts SET status=?, notes=?, updated_at=? WHERE id=?",
                        ("merged", reason, iso(utc_now()), attempt_id))
            updated += 1
            continue

        if state == "closed":
            reward, reason = 0.2, "issue_closed_observed"
            con.execute("INSERT INTO rewards(attempt_id,reward_value,reward_reason,created_at) VALUES(?,?,?,?)",
                        (attempt_id, reward, reason, iso(utc_now())))
            update_arm(con, akey, reward)
            con.execute("UPDATE attempts SET status=?, notes=?, updated_at=? WHERE id=?",
                        ("ignored", reason, iso(utc_now()), attempt_id))
            updated += 1
            continue

        con.execute("UPDATE attempts SET updated_at=? WHERE id=?", (iso(utc_now()), attempt_id))

    con.commit()
    return updated


# ---------------- Worker: PR automation (strict) ----------------
def gh_me() -> Dict[str, Any]:
    return gh_get("https://api.github.com/user")

def default_branch(repo_full: str) -> str:
    d = gh_get(f"https://api.github.com/repos/{repo_full}")
    return d.get("default_branch") or "main"

def ensure_fork(upstream_full: str, my_login: str) -> str:
    _, repo = upstream_full.split("/", 1)
    fork_full = f"{my_login}/{repo}"
    try:
        gh_get(f"https://api.github.com/repos/{fork_full}")
        return fork_full
    except Exception:
        pass

    gh_post(f"https://api.github.com/repos/{upstream_full}/forks", {})
    for _ in range(30):
        time.sleep(3)
        try:
            gh_get(f"https://api.github.com/repos/{fork_full}")
            return fork_full
        except Exception:
            continue
    raise RuntimeError("Fork creation timeout.")

def open_pr(upstream_full: str, fork_full: str, branch: str, base: str, title: str, body: str) -> str:
    payload = {
        "title": title,
        "body": body,
        "head": f"{fork_full.split('/')[0]}:{branch}",
        "base": base,
        "maintainer_can_modify": True,
    }
    pr = gh_post(f"https://api.github.com/repos/{upstream_full}/pulls", payload)
    return pr.get("html_url", "")

def safe_mkdir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def write_file(path: str, content: str) -> None:
    safe_mkdir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def repo_tree_hint(repo_dir: str) -> str:
    lines = []
    for root, dirs, files in os.walk(repo_dir):
        rel = root.replace(repo_dir, "").lstrip(os.sep)
        if rel.startswith(".git") or rel.startswith(".venv") or rel.startswith("node_modules") or rel.startswith("dist") or rel.startswith("build"):
            dirs[:] = []
            continue
        if rel.count(os.sep) > 3:
            dirs[:] = []
            continue
        for fn in sorted(files)[:60]:
            if fn.endswith((".py", ".toml", ".cfg", ".ini", ".md")):
                lines.append(os.path.join(rel, fn) if rel else fn)
        if len(lines) > 400:
            break
    return "\n".join(lines[:400])

def prepare_python_env(repo_dir: str) -> None:
    run_cmd(["python", "-m", "pip", "install", "--upgrade", "pip"], timeout=600)
    # install deps if possible
    for name in ("requirements.txt", "requirements-dev.txt", "requirements_test.txt", "requirements-test.txt"):
        p = os.path.join(repo_dir, name)
        if os.path.exists(p):
            run_cmd(["python", "-m", "pip", "install", "-r", p], cwd=repo_dir, timeout=800)
            break
    if os.path.exists(os.path.join(repo_dir, "pyproject.toml")) or os.path.exists(os.path.join(repo_dir, "setup.py")):
        run_cmd(["python", "-m", "pip", "install", "-e", "."], cwd=repo_dir, timeout=900)
    run_cmd(["python", "-m", "pip", "install", "pytest"], timeout=600)

def run_pytest(repo_dir: str) -> Tuple[bool, str]:
    code, out = run_cmd(["python", "-m", "pytest", "-q"], cwd=repo_dir, timeout=WORK_TIMEOUT_SEC)
    return (code == 0, out)

def git_apply(repo_dir: str, diff_text: str) -> Tuple[bool, str]:
    p = os.path.join(repo_dir, ".bountybot.patch")
    with open(p, "w", encoding="utf-8") as f:
        f.write(diff_text)
    code, out = run_cmd(["git", "apply", "--whitespace=fix", p], cwd=repo_dir, timeout=120)
    return (code == 0, out)

def git_set_identity(repo_dir: str) -> None:
    run_cmd(["git", "config", "user.name", "bountybot"], cwd=repo_dir, timeout=60)
    run_cmd(["git", "config", "user.email", "bountybot@users.noreply.github.com"], cwd=repo_dir, timeout=60)

def git_checkout_new_branch(repo_dir: str, branch: str) -> None:
    run_cmd(["git", "checkout", "-b", branch], cwd=repo_dir, timeout=60)

def git_commit_all(repo_dir: str, msg: str) -> Tuple[bool, str]:
    run_cmd(["git", "add", "-A"], cwd=repo_dir, timeout=120)
    code, out = run_cmd(["git", "commit", "-m", msg], cwd=repo_dir, timeout=120)
    return (code == 0, out)

def git_push_to_fork(repo_dir: str, fork_full: str, branch: str) -> None:
    remote = "fork"
    run_cmd(["git", "remote", "remove", remote], cwd=repo_dir, timeout=60)
    run_cmd(["git", "remote", "add", remote, f"https://github.com/{fork_full}.git"], cwd=repo_dir, timeout=60)
    code, out = run_cmd(
        ["git", "-c", f"http.extraheader=AUTHORIZATION: bearer {GITHUB_TOKEN}", "push", remote, f"{branch}:{branch}", "--force"],
        cwd=repo_dir,
        timeout=300,
    )
    if code != 0:
        raise RuntimeError(f"git push failed: {out}")

def deepseek_repro_test(issue_title: str, issue_body: str, tree_hint: str) -> Dict[str, Any]:
    prompt = {
        "task": "Create a minimal pytest regression test that fails on current code and represents the bug described.",
        "output_schema": {"test_path": "string", "test_code": "string", "confidence": "0..1"},
        "rules": [
            "Return JSON only.",
            "No external network calls.",
            "If you cannot confidently reproduce, set confidence <= 0.3 and still return a safe test (maybe xfail).",
        ],
        "issue": {"title": issue_title[:300], "body": issue_body[:2000]},
        "repo_tree_hint": tree_hint[:4000],
    }
    return deepseek_chat_json(prompt, temperature=0.2, timeout=80)

def deepseek_fix_patch(issue_title: str, issue_body: str, failing_pytest_output: str, tree_hint: str) -> Dict[str, Any]:
    prompt = {
        "task": "Given failing pytest output, produce a minimal unified diff patch that makes tests pass while preserving intended behavior.",
        "output_schema": {"diff": "string", "confidence": "0..1"},
        "rules": [
            "Return JSON only.",
            "Diff must be valid unified diff (git apply).",
            "Keep changes minimal.",
        ],
        "issue": {"title": issue_title[:300], "body": issue_body[:2000]},
        "failing_pytest_output": failing_pytest_output[:6000],
        "repo_tree_hint": tree_hint[:4000],
    }
    return deepseek_chat_json(prompt, temperature=0.2, timeout=80)

def is_python(language_hint: str) -> bool:
    return (language_hint or "").lower() == "python"

def work(con: sqlite3.Connection) -> int:
    if not DEEPSEEK_API_KEY:
        log.info("DEEPSEEK_API_KEY missing -> PR automation disabled.")
        return 0

    rows = con.execute(
        """
        SELECT a.id, a.candidate_id, c.repo_full_name, c.issue_number, c.issue_url, c.title, c.body_snip, c.language_hint
        FROM attempts a
        JOIN candidates c ON c.id=a.candidate_id
        WHERE a.status='queued'
        ORDER BY c.final_score DESC
        LIMIT ?
        """,
        (BOT_MAX_PR_PER_RUN,),
    ).fetchall()
    # --- FORCE EXPLORE (bootstrap): widen fetch until we have at least one python row ---
    if rows:
        # If the top-N queued are all non-python, fetch more and filter to python so Work can actually try.
        if all((not is_python(r[-1])) for r in rows):
            more = con.execute(
                """
                SELECT a.id, a.candidate_id, c.repo_full_name, c.issue_number, c.issue_url, c.title, c.body_snip, c.language_hint
                FROM attempts a
                JOIN candidates c ON c.id=a.candidate_id
                WHERE a.status='queued'
                ORDER BY c.final_score DESC
                LIMIT ?
                """,
                (max(BOT_MAX_PR_PER_RUN * 20, 100),),
            ).fetchall()

            py_rows = [r for r in more if is_python(r[-1])]
            if py_rows:
                rows = py_rows[:BOT_MAX_PR_PER_RUN]

    if not rows:
        log.info("No queued attempts.")
        return 0

    created = 0
    for attempt_id, cand_id, upstream_full, issue_number, issue_url, title, body_snip, lang in rows:
        if not is_python(lang):
            con.execute("UPDATE attempts SET status=?, notes=?, updated_at=? WHERE id=?",
                        ("ignored", "non_python_repo", iso(utc_now()), attempt_id))
            con.commit()
            continue

        con.execute("UPDATE attempts SET status=?, notes=?, updated_at=? WHERE id=?",
                    ("working", "start", iso(utc_now()), attempt_id))
        con.commit()

        base = default_branch(upstream_full)
        workdir = os.path.join(os.getcwd(), "_work")
        os.makedirs(workdir, exist_ok=True)
        repo_dir = os.path.join(workdir, f"repo_{attempt_id}_{sha1(upstream_full)[:8]}")
        if os.path.exists(repo_dir):
            shutil.rmtree(repo_dir, ignore_errors=True)

        code, out = run_cmd(["git", "clone", "--depth", "1", "--branch", base, f"https://github.com/{upstream_full}.git", repo_dir], timeout=300)
        if code != 0:
            con.execute("UPDATE attempts SET status=?, notes=?, updated_at=? WHERE id=?",
                        ("rejected", f"clone_failed:{out[:200]}", iso(utc_now()), attempt_id))
            con.commit()
            continue

        git_set_identity(repo_dir)
        try:
            prepare_python_env(repo_dir)
        except Exception as e:
            con.execute("UPDATE attempts SET status=?, notes=?, updated_at=? WHERE id=?",
                        ("rejected", f"deps_failed:{str(e)[:200]}", iso(utc_now()), attempt_id))
            con.commit()
            continue

        ok, baseline = run_pytest(repo_dir)
        if not ok:
            con.execute("UPDATE attempts SET status=?, notes=?, updated_at=? WHERE id=?",
                        ("ignored", "baseline_tests_failing_skip", iso(utc_now()), attempt_id))
            con.commit()
            continue

        tree = repo_tree_hint(repo_dir)

        # Branch
        branch = f"bountybot-fix-{issue_number}-{sha1(issue_url)[:7]}"
        git_checkout_new_branch(repo_dir, branch)

        # Repro test
        try:
            t = deepseek_repro_test(title, body_snip, tree)
        except Exception as e:
            con.execute("UPDATE attempts SET status=?, notes=?, updated_at=? WHERE id=?",
                        ("rejected", f"repro_gen_failed:{str(e)[:200]}", iso(utc_now()), attempt_id))
            con.commit()
            continue

        conf = float(t.get("confidence", 0.0) or 0.0)
        test_path = t.get("test_path", "tests/test_regression_bountybot.py")
        test_code = (t.get("test_code") or "").strip()

        if conf < 0.60 or not test_code:
            con.execute("UPDATE attempts SET status=?, notes=?, updated_at=? WHERE id=?",
                        ("ignored", f"low_repro_conf:{conf:.2f}", iso(utc_now()), attempt_id))
            con.commit()
            continue

        write_file(os.path.join(repo_dir, test_path), test_code)

        ok, out_fail = run_pytest(repo_dir)
        if ok:
            con.execute("UPDATE attempts SET status=?, notes=?, updated_at=? WHERE id=?",
                        ("ignored", "repro_not_achieved", iso(utc_now()), attempt_id))
            con.commit()
            continue

        failing = out_fail

        # Autofix loop
        fixed = False
        for _ in range(WORK_MAX_AUTOFIX):
            try:
                fx = deepseek_fix_patch(title, body_snip, failing, tree)
            except Exception as e:
                con.execute("UPDATE attempts SET status=?, notes=?, updated_at=? WHERE id=?",
                            ("rejected", f"fix_gen_failed:{str(e)[:200]}", iso(utc_now()), attempt_id))
                con.commit()
                break

            fx_conf = float(fx.get("confidence", 0.0) or 0.0)
            diff = (fx.get("diff") or "").strip()
            if fx_conf < 0.60 or "diff --git" not in diff:
                con.execute("UPDATE attempts SET status=?, notes=?, updated_at=? WHERE id=?",
                            ("ignored", f"low_fix_conf:{fx_conf:.2f}", iso(utc_now()), attempt_id))
                con.commit()
                break

            applied, apply_out = git_apply(repo_dir, diff)
            if not applied:
                failing = f"git apply failed:\n{apply_out}\n\nprev pytest:\n{failing}"
                continue

            ok2, out2 = run_pytest(repo_dir)
            if ok2:
                fixed = True
                committed, cout = git_commit_all(repo_dir, f"Fix: {title[:60]}")
                if not committed:
                    con.execute("UPDATE attempts SET status=?, notes=?, updated_at=? WHERE id=?",
                                ("rejected", f"commit_failed:{cout[:200]}", iso(utc_now()), attempt_id))
                    con.commit()
                    fixed = False
                break
            failing = out2

        if not fixed:
            if con.execute("SELECT status FROM attempts WHERE id=?", (attempt_id,)).fetchone()[0] == "working":
                con.execute("UPDATE attempts SET status=?, notes=?, updated_at=? WHERE id=?",
                            ("rejected", "autofix_exhausted", iso(utc_now()), attempt_id))
                con.commit()
            continue

        # Fork / push / PR
        me = gh_me()
        my_login = me.get("login")
        if not my_login:
            con.execute("UPDATE attempts SET status=?, notes=?, updated_at=? WHERE id=?",
                        ("rejected", "cannot_get_login", iso(utc_now()), attempt_id))
            con.commit()
            continue

        try:
            fork_full = ensure_fork(upstream_full, my_login)
            git_push_to_fork(repo_dir, fork_full, branch)
            pr = open_pr(
                upstream_full=upstream_full,
                fork_full=fork_full,
                branch=branch,
                base=base,
                title=f"Fix: {title[:80]}",
                body=(
                    f"Fixes issue: {issue_url}\n\n"
                    f"- Adds regression test: `{test_path}`\n"
                    f"- Minimal fix to make tests pass\n\n"
                    f"(Generated by bountybot; conservative mode)\n"
                ),
            )
        except Exception as e:
            con.execute("UPDATE attempts SET status=?, notes=?, updated_at=? WHERE id=?",
                        ("rejected", f"pr_failed:{str(e)[:200]}", iso(utc_now()), attempt_id))
            con.commit()
            continue

        con.execute(
            "UPDATE attempts SET status=?, notes=?, pr_url=?, fork_full_name=?, branch_name=?, updated_at=? WHERE id=?",
            ("pr_opened", "opened_pr", pr, fork_full, branch, iso(utc_now()), attempt_id),
        )
        con.commit()
        created += 1

    return created


# ---------------- Report ----------------
def report(con: sqlite3.Connection) -> None:
    rows = con.execute(
        """
        SELECT a.id, a.status, a.pr_url, c.final_score, c.payment_amount_usd, c.payout_hint, c.issue_url, c.title
        FROM attempts a
        JOIN candidates c ON c.id=a.candidate_id
        ORDER BY a.created_at DESC
        LIMIT 30
        """
    ).fetchall()

    log.info("Attempts (latest 30):")
    for a_id, st, pr, score, amt, pay, url, title in rows:
        log.info(f"#{a_id} {st} score={float(score or 0):.2f} amt={amt} pay={pay} | {pr or '-'} | {url}")

    arms = con.execute("SELECT arm_key, n, mean_reward FROM arms ORDER BY mean_reward DESC, n DESC LIMIT 15").fetchall()
    log.info("Arms (top):")
    for k, n, m in arms:
        log.info(f"{k} | n={n} mean={float(m):.3f}")


# ---------------- Entrypoints ----------------
def cmd_run() -> None:
    con = db()
    collect(con)
    rescore_all(con)
    pick_queue(con)
    report(con)

def cmd_work() -> None:
    con = db()
    created = work(con)
    log.info(f"PRs created this run: {created}")
    report(con)

def cmd_monitor() -> None:
    con = db()
    updated = monitor(con)
    log.info(f"monitor updates: {updated}")
    report(con)

def cmd_report() -> None:
    con = db()
    report(con)

if __name__ == "__main__":
    import sys
    sub = (sys.argv[1] if len(sys.argv) > 1 else "run").lower()
    if sub == "run":
        cmd_run()
    elif sub == "work":
        cmd_work()
    elif sub == "monitor":
        cmd_monitor()
    elif sub == "report":
        cmd_report()
    else:
        raise SystemExit("Usage: python main.py [run|work|monitor|report]")
