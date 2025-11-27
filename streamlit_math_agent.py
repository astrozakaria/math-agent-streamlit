"""
Streamlit Math Agent - I/O & non-interactive safe version

This file is a robust starter that supports:
- Streamlit UI when `streamlit` is installed.
- CLI mode when Streamlit is absent, with a safe fallback for non-interactive/sandboxed environments
  where console `input()` or filesystem writes may fail.

Fixes in this version:
- Avoids unhandled OSError from file I/O by feature-detecting filesystem availability and
  falling back to in-memory storage when necessary.
- Avoids unhandled OSError from console input by detecting interactivity (tty) and providing
  a non-interactive demo runner when stdin isn't interactive.
- Wraps all `input()` calls in try/except catching `OSError`, `EOFError`, and `KeyboardInterrupt`.

How to run:
- Streamlit mode (if available): `streamlit run streamlit_math_agent.py`
- CLI mode (no streamlit): `python streamlit_math_agent.py`

If running in a CI or restricted sandbox where stdin/stdout and filesystem are not interactive/writable
this script will run a short non-interactive demo showing question generation and grading.
"""

import random
import json
import sys
import tempfile
from pathlib import Path

# Try importing streamlit; if unavailable, run CLI fallback
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except Exception:
    st = None
    STREAMLIT_AVAILABLE = False

from sympy import symbols, sympify, simplify
from sympy.parsing.sympy_parser import parse_expr

# ------------------ Safe I/O setup ------------------
DATA_FILE = Path("progress.json")
# Detect whether file I/O operations appear to be allowed in this environment.
FILE_IO_AVAILABLE = True
_in_memory_progress = {}

try:
    # Try a minimal write/read to a temporary file to check filesystem permissions.
    with tempfile.NamedTemporaryFile(mode='w+', delete=True) as tf:
        tf.write('ok')
        tf.flush()
        tf.seek(0)
        _ = tf.read()
except Exception:
    FILE_IO_AVAILABLE = False


def _log_warn(msg):
    if STREAMLIT_AVAILABLE:
        try:
            st.warning(msg)
        except Exception:
            print('WARNING:', msg, file=sys.stderr)
    else:
        print('WARNING:', msg, file=sys.stderr)


# ------------------ Utilities ------------------

def load_progress():
    """Load progress from DATA_FILE if file I/O is available. Otherwise return in-memory progress.

    This function will never raise OSError; it always returns a dict (possibly empty).
    """
    global _in_memory_progress, FILE_IO_AVAILABLE
    if not FILE_IO_AVAILABLE:
        _log_warn('File I/O not available in this environment; using in-memory progress (not persisted).')
        return dict(_in_memory_progress)

    try:
        if DATA_FILE.exists():
            try:
                text = DATA_FILE.read_text()
                return json.loads(text) if text else {}
            except Exception:
                return {}
        return {}
    except Exception:
        _log_warn('Could not access progress.json; falling back to in-memory progress.')
        try:
            FILE_IO_AVAILABLE = False
        except Exception:
            globals()['FILE_IO_AVAILABLE'] = False
        return dict(_in_memory_progress)


def save_progress(data):
    """Save progress to DATA_FILE if possible. If not, keep it in memory and return False.

    Returns True if persisted to disk, False otherwise.
    """
    global _in_memory_progress
    if not FILE_IO_AVAILABLE:
        _in_memory_progress = dict(data)
        _log_warn('File I/O not available; progress stored in memory only (will not persist).')
        return False

    try:
        DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        DATA_FILE.write_text(json.dumps(data, indent=2))
        return True
    except Exception as e:
        _in_memory_progress = dict(data)
        _log_warn(f"Failed to write progress.json ({e}); using in-memory progress.")
        try:
            globals()['FILE_IO_AVAILABLE'] = False
        except Exception:
            pass
        return False

# Symbol used for algebra
x = symbols('x')

# ------------------ Question Generators ------------------

def gen_linear_int_solution(min_a=1, max_a=9, min_b=-9, max_b=9, min_c=-20, max_c=20):
    a = random.randint(min_a, max_a)
    b = random.randint(min_b, max_b)
    c = random.randint(min_c, max_c)
    q = f"Solve for x: {a}*x + ({b}) = {c}"
    sol = (c - b) / a
    return {"text": q, "answer": str(sol), "type": "linear", "meta": {"a": a, "b": b, "c": c}}


def gen_quadratic_root_sum(min_a=1, max_a=3, min_r=-5, max_r=5):
    r1 = random.randint(min_r, max_r)
    r2 = random.randint(min_r, max_r)
    a = random.randint(min_a, max_a)
    expr = sympify(a) * (x - r1) * (x - r2)
    q = f"Given the quadratic equation {str(expr.expand())} = 0, what is the sum of the roots?"
    sol = r1 + r2
    return {"text": q, "answer": str(sol), "type": "quadratic", "meta": {"r1": r1, "r2": r2, "a": a}}


def gen_arithmetic_simple(min_n=2, max_n=4, min_val=1, max_val=20):
    n = random.randint(min_n, max_n)
    ops = random.choices(['+', '-', '*', '/'], k=n-1)
    nums = [str(random.randint(min_val, max_val)) for _ in range(n)]
    expr = " ".join(sum([[nums[i], ops[i]] for i in range(n-1)], []) + [nums[-1]])
    q = f"Evaluate: {expr}"
    ans = str(sympify(expr))
    return {"text": q, "answer": ans, "type": "arithmetic", "meta": {"expr": expr}}


def generate_question(topic="mixed", difficulty="easy"):
    if topic == "linear":
        return gen_linear_int_solution()
    if topic == "quadratic":
        return gen_quadratic_root_sum()
    if topic == "arithmetic":
        return gen_arithmetic_simple()
    choice = random.choice([gen_linear_int_solution, gen_quadratic_root_sum, gen_arithmetic_simple])
    return choice()

# ------------------ Grading ------------------

def parse_user_expression(user_input):
    try:
        return parse_expr(user_input, evaluate=True)
    except Exception:
        try:
            return sympify(user_input)
        except Exception:
            raise


def check_answer(user_input_raw, canonical_answer_raw):
    try:
        user_expr = parse_user_expression(user_input_raw)
    except Exception as e:
        return {"ok": False, "reason": f"Couldn't parse your answer: {e}"}

    try:
        canon_expr = parse_user_expression(str(canonical_answer_raw))
    except Exception as e:
        try:
            u_val = float(user_expr.evalf())
            c_val = float(canonical_answer_raw)
            same = abs(u_val - c_val) <= 1e-6 * max(1.0, abs(c_val))
            return {"ok": same, "reason": None if same else f"Numeric mismatch: {u_val} vs {c_val}"}
        except Exception as e2:
            return {"ok": False, "reason": f"Internal error parsing canonical answer: {e}; {e2}"}

    try:
        diff = simplify(user_expr - canon_expr)
        if diff == 0:
            return {"ok": True, "reason": None}
        try:
            u_val = float(user_expr.evalf())
            c_val = float(canon_expr.evalf())
            same = abs(u_val - c_val) <= 1e-6 * max(1.0, abs(c_val))
            if same:
                return {"ok": True, "reason": None}
            else:
                return {"ok": False, "reason": f"Numeric mismatch: {u_val} vs {c_val}"}
        except Exception:
            test_points = [random.randint(-10, 10) for _ in range(5)]
            all_close = True
            for t in test_points:
                ue = user_expr.subs(x, t).evalf()
                ce = canon_expr.subs(x, t).evalf()
                try:
                    ue_f = float(ue)
                    ce_f = float(ce)
                    if abs(ue_f - ce_f) > 1e-6 * max(1.0, abs(ce_f)):
                        all_close = False
                        break
                except Exception:
                    all_close = False
                    break
            if all_close:
                return {"ok": True, "reason": None}
            return {"ok": False, "reason": "Expressions do not appear equivalent."}
    except Exception as e:
        return {"ok": False, "reason": f"Error during comparison: {e}"}

# ------------------ App: Streamlit or CLI ------------------

def run_streamlit_app():
    def init_state():
        if 'progress' not in st.session_state:
            st.session_state.progress = load_progress()
        if 'current_q' not in st.session_state:
            st.session_state.current_q = None
        if 'history' not in st.session_state:
            st.session_state.history = []

    st.set_page_config(page_title="Math Practice Agent", layout="centered")
    init_state()

    st.title("🧠 Math Practice Agent")
    st.write("A small starter app that generates questions, checks answers with SymPy, and tracks progress.")

    st.sidebar.header("Settings")
    topic = st.sidebar.selectbox("Topic", options=["mixed", "linear", "quadratic", "arithmetic"], index=0)
    difficulty = st.sidebar.selectbox("Difficulty", options=["easy", "medium", "hard"], index=0)
    seed_btn = st.sidebar.button("New random seed")
    if seed_btn:
        random.seed()

    st.sidebar.markdown("---")
    if st.sidebar.button("Reset progress (local)"):
        st.session_state.progress = {}
        save_progress(st.session_state.progress)
        st.sidebar.success("Progress reset")

    if st.button("Generate question") or st.session_state.current_q is None:
        q = generate_question(topic=topic, difficulty=difficulty)
        st.session_state.current_q = q
        st.session_state.user_answer = ""

    q = st.session_state.current_q
    if q:
        st.subheader("Question")
        st.markdown(q['text'])
        st.text_input("Your answer (enter an expression or number):", key='user_answer')

        col1, col2 = st.columns([1,1])
        with col1:
            if st.button("Check answer"):
                user_raw = st.session_state.user_answer
                result = check_answer(user_raw, q['answer'])
                entry = {
                    'question': q['text'],
                    'canonical': q['answer'],
                    'answer': user_raw,
                    'ok': result['ok'],
                    'reason': result.get('reason')
                }
                st.session_state.history.append(entry)

                topic_key = q.get('type', 'mixed')
                prog = st.session_state.progress.get(topic_key, {'attempts': 0, 'correct': 0})
                prog['attempts'] = prog.get('attempts', 0) + 1
                if result['ok']:
                    prog['correct'] = prog.get('correct', 0) + 1
                    st.success("Correct! ✅")
                else:
                    st.error("Incorrect. ❌")
                    if result.get('reason'):
                        st.info(result['reason'])
                st.session_state.progress[topic_key] = prog
                save_progress(st.session_state.progress)

        with col2:
            if st.button("Show solution"):
                st.info(f"Canonical answer: {q['answer']}")

    st.sidebar.header("Progress")
    if st.session_state.progress:
        for k, v in st.session_state.progress.items():
            attempts = v.get('attempts', 0)
            correct = v.get('correct', 0)
            pct = (correct / attempts * 100) if attempts > 0 else 0
            st.sidebar.write(f"{k.title()}: {correct}/{attempts} correct ({pct:.1f}%)")
    else:
        st.sidebar.write("No progress yet. Generate and answer questions to track progress.")

    st.header("Session History")
    if st.session_state.history:
        for i, h in enumerate(reversed(st.session_state.history[-10:]), 1):
            st.write(f"**Q:** {h['question']}")
            st.write(f"Your answer: `{h['answer']}` — {'✅' if h['ok'] else '❌'}")
            if h.get('reason'):
                st.write(f"_Note_: {h['reason']}")
            st.markdown("---")
    else:
        st.write("No attempts yet — answer a question to see history.")

    st.markdown("---")
    if st.button("Export recent attempts as JSON"):
        export = json.dumps(st.session_state.history[-50:], indent=2)
        st.download_button("Download JSON", export, file_name="attempts.json", mime="application/json")

    st.write("Made with ❤️ — extend the generators, rubrics, and UI as you like!")


def _noninteractive_demo(num_questions=5):
    """Run a short non-interactive demo: generate questions and auto-check using canonical answers.

    Useful when stdin is not interactive (e.g. sandboxed environment). This avoids calling input().
    """
    print("Running non-interactive demo — stdin is not interactive.\n")
    progress = load_progress()
    for _ in range(num_questions):
        q = generate_question()
        print(f"Question: {q['text']}")
        # Use canonical answer as the simulated user answer
        user_ans = q['answer']
        print(f"Simulated answer: {user_ans}")
        res = check_answer(user_ans, q['answer'])
        print("Result:", "Correct ✅" if res['ok'] else f"Incorrect ❌ ({res.get('reason')})")
        topic_key = q.get('type', 'mixed')
        prog = progress.get(topic_key, {'attempts': 0, 'correct': 0})
        prog['attempts'] = prog.get('attempts', 0) + 1
        if res['ok']:
            prog['correct'] = prog.get('correct', 0) + 1
        progress[topic_key] = prog
    save_progress(progress)
    print('\nDemo finished. Progress saved if possible.')


def run_cli():
    """Simple command-line fallback for environments without Streamlit.
    - If stdin is interactive, run an interactive prompt loop using input().
    - If stdin is NOT interactive, run a non-interactive demo to avoid calling input().
    """
    print("Math Practice Agent (CLI mode)")
    if not FILE_IO_AVAILABLE:
        print("Note: File I/O is not available; progress will NOT be persisted to disk.")

    # Detect whether stdin supports interactive input
    interactive_stdin = sys.stdin.isatty()
    if not interactive_stdin:
        _noninteractive_demo(num_questions=5)
        return

    progress = load_progress()

    while True:
        print('\n---')
        try:
            topic = input("Choose topic (mixed/linear/quadratic/arithmetic) [mixed]: ").strip() or 'mixed'
        except (EOFError, KeyboardInterrupt, OSError):
            print("\nInput closed or unavailable. Exiting.")
            break
        q = generate_question(topic=topic)
        print(f"\nQuestion: {q['text']}")
        try:
            ans = input("Your answer (or type 'show' to reveal solution, 'exit' to quit): ").strip()
        except (EOFError, KeyboardInterrupt, OSError):
            print("\nInput closed or unavailable. Exiting.")
            break
        if ans.lower() == 'exit':
            break
        if ans.lower() == 'show':
            print(f"Canonical answer: {q['answer']}")
            continue
        result = check_answer(ans, q['answer'])
        entry = {
            'question': q['text'],
            'canonical': q['answer'],
            'answer': ans,
            'ok': result['ok'],
            'reason': result.get('reason')
        }
        topic_key = q.get('type', 'mixed')
        prog = progress.get(topic_key, {'attempts': 0, 'correct': 0})
        prog['attempts'] = prog.get('attempts', 0) + 1
        if result['ok']:
            prog['correct'] = prog.get('correct', 0) + 1
            print("Correct! ✅")
        else:
            print("Incorrect. ❌")
            if result.get('reason'):
                print(f"Note: {result.get('reason')}")
        progress[topic_key] = prog
        save_progress(progress)

    print("Goodbye — progress saved (if possible).")


# ------------------ Small self-tests (run in CLI mode) ------------------

def _run_tests():
    tests = []

    # Simple numeric equality
    tests.append(("3", "3", True))
    tests.append(("6/2", "3", True))
    tests.append(("1/3", "0.3333333333333333", True))

    # Symbolic equivalence
    tests.append(("(x+1)**2", "x**2 + 2*x + 1", True))
    tests.append(("2*x", "x + x", True))
    tests.append(("x+1", "x-1", False))

    # Quadratic root sum check (numerical)
    tests.append(("5", "5", True))

    # Additional tests for edge cases
    tests.append(("-2", "-2", True))
    tests.append((("(x-2)*(x+3)"), "x**2 + x -6", True))

    # Extra test to increase coverage
    tests.append(("(x+2)*(x-3)", "x**2 - x -6", True))

    ok = True
    for i, (u, c, expected) in enumerate(tests, 1):
        r = check_answer(u, c)
        passed = r['ok'] == expected
        ok = ok and passed
        print(f"Test {i}: user='{u}' canon='{c}' -> ok={r['ok']} (expected {expected}) {'PASS' if passed else 'FAIL'}")
        if not passed and r.get('reason'):
            print("  reason:", r['reason'])

    if ok:
        print("All tests passed.")
    else:
        print("Some tests failed — inspect outputs above.")


if __name__ == '__main__':
    if STREAMLIT_AVAILABLE:
        print("Streamlit is available. Run this app with: streamlit run streamlit_math_agent.py")
        try:
            resp = 'n'
            # Only prompt if stdin is interactive
            if sys.stdin.isatty():
                try:
                    resp = input("Start CLI mode anyway? (y/N): ").strip().lower()
                except (EOFError, KeyboardInterrupt, OSError):
                    resp = 'n'
        except Exception:
            resp = 'n'
        if resp == 'y':
            _run_tests()
            run_cli()
        else:
            print("Exiting. Launch using streamlit for full UI.")
            sys.exit(0)
    else:
        # If streamlit is not available, run tests then CLI (or demo if non-interactive)
        _run_tests()
        run_cli()
