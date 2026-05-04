import streamlit as st
import joblib
import json
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix

# ── Chargement des artefacts ──────────────────────────────────────
model  = joblib.load('model.pkl')
tfidf  = joblib.load('tfidf.pkl')
scaler = joblib.load('scaler.pkl')
le     = joblib.load('label_encoder.pkl')

# ── Logique de complexité ─────────────────────────────────────────
WEIGHTS = {
    'algo_array':1,'algo_string':1,'algo_hash':1,
    'algo_two_pointers':1,'algo_sliding_window':2,'algo_stack':2,
    'algo_heap':3,'algo_tree':3,'algo_graph':4,
    'algo_dp':5,'algo_backtracking':4,'algo_bit_manipulation':3
}

ALGO_KEYWORDS = {
    'algo_tree':            ['tree','binary tree','bst','traversal','root','leaf','node'],
    'algo_graph':           ['graph','edge','vertex','path','cycle','dijkstra','bfs','dfs'],
    'algo_dp':              ['dynamic','dp','optimal','subproblem','memoization','tabulation'],
    'algo_backtracking':    ['backtrack','permutation','combination','subset'],
    'algo_sliding_window':  ['sliding window','window','subarray'],
    'algo_two_pointers':    ['two pointer','two sum','left right'],
    'algo_heap':            ['heap','priority queue','min heap','max heap'],
    'algo_stack':           ['stack','bracket','parenthes'],
    'algo_hash':            ['hash','hashmap','dictionary','lookup'],
    'algo_bit_manipulation':['xor','bit','&','|','<<','>>'],
    'algo_string':          ['string','substring','palindrome','anagram'],
    'algo_array':           ['array','subarray','prefix sum','cumulative']
}

def detect_algos(text):
    text_lower = text.lower()
    return {algo: sum(1 for kw in kws if kw in text_lower)
            for algo, kws in ALGO_KEYWORDS.items()}

def compute_complexity_score(algos, difficulty):
    score = sum(WEIGHTS.get(algo, 1) for algo, count in algos.items() if count > 0)
    if difficulty == 'Medium': score += 2
    elif difficulty == 'Hard': score += 4
    return score

def estimate_big_o(score):
    if score <= 3:   return 'O(n)'
    elif score <= 6: return 'O(n log n)'
    elif score <= 9: return 'O(n²)'
    else:            return 'O(2ⁿ)'

def choose_language_by_complexity(score):
    if score <= 4:   return 'Python'
    elif score <= 8: return 'Java'
    else:            return 'C++'

# ── Prédiction ML ────────────────────────────────────────────────
def predict_ml(title, description, constraints=[]):
    text = title + ' ' + title + ' ' + description + ' ' + json.dumps(constraints)
    text_lower = text.lower()

    algo_keywords_train = {
        'tree':             ['tree','binary tree','bst','traversal','root','leaf','node'],
        'graph':            ['graph','edge','vertex','path','cycle','dijkstra','bfs','dfs'],
        'dp':               ['dynamic','dp','optimal','subproblem','memoization','tabulation'],
        'greedy':           ['greedy','interval','activity','schedule'],
        'bit_manipulation': ['xor','bit','&','|','<<','>>'],
        'string':           ['string','substring','palindrome','anagram'],
        'array':            ['array','subarray','prefix sum','cumulative']
    }

    hard_kw = ['tree','graph','dp','dynamic','xor','minimum','maximum',
                'optimal','cycle','mst','dijkstra','backtrack','recursion']
    easy_kw = ['sort','search','array','check','count','unique','string','simple']

    hard_count = sum(1 for kw in hard_kw if kw in text_lower)
    easy_count = sum(1 for kw in easy_kw if kw in text_lower)
    difficulty_ratio = hard_count / (easy_count + 1)

    algo_features = [
        sum(1 for kw in kws if kw in text_lower)
        for kws in algo_keywords_train.values()
    ]

    weights = {'array':1,'string':1,'greedy':2,'tree':3,'graph':4,'dp':5,'bit_manipulation':3}
    complexity_score = sum(
        algo_features[i] * weights.get(a, 1)
        for i, a in enumerate(algo_keywords_train)
    )

    num_feat = np.array([[
        len(title),
        len(description),
        len(description.split()),
        len(constraints),
        hard_count,
        easy_count,
        difficulty_ratio,
        complexity_score
    ] + algo_features])

    X_t = tfidf.transform([text])
    X_n = scaler.transform(num_feat)
    X   = hstack([X_t, csr_matrix(X_n)])

    proba = model.predict_proba(X)[0]
    top2  = np.argsort(proba)[-2:][::-1]
    return [(le.classes_[i], proba[i]) for i in top2]

# ── Dataset ───────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv('questions_dataset.csv')

df = load_data()

# ── Interface ─────────────────────────────────────────────────────
st.title("Prédicteur de langage optimal")
st.caption("Prédit le meilleur langage selon la complexité algorithmique du problème.")

if st.button("🎲 Exemple aléatoire du dataset"):
    row = df.sample(1).iloc[0]
    st.session_state['title']       = str(row['title'])
    st.session_state['description'] = str(row['description'])
    try:
        c = json.loads(row['constraints'])
        st.session_state['constraints'] = ', '.join(c) if isinstance(c, list) else str(row['constraints'])
    except:
        st.session_state['constraints'] = str(row.get('constraints', ''))
    st.session_state['difficulty'] = str(row.get('difficulty_level', 'Medium'))

st.divider()

title           = st.text_input("Titre du problème", value=st.session_state.get('title', 'Two Sum'))
description     = st.text_area("Description",        value=st.session_state.get('description', 'Given an array of integers, return indices of two numbers that add up to target.'))
constraints_raw = st.text_input("Contraintes",       value=st.session_state.get('constraints', '2 <= n <= 10^4'))
difficulty      = st.selectbox("Niveau de difficulté", ['Easy','Medium','Hard'],
                                index=['Easy','Medium','Hard'].index(st.session_state.get('difficulty','Medium')))

if st.button("Prédire"):
    constraints = [c.strip() for c in constraints_raw.split(',') if c.strip()]
    text = title + ' ' + description

    algos           = detect_algos(text)
    score           = compute_complexity_score(algos, difficulty)
    big_o           = estimate_big_o(score)
    lang_complexity = choose_language_by_complexity(score)
    ml_results      = predict_ml(title, description, constraints)

    st.subheader("Résultats")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Analyse de complexité**")
        st.success(f"Langage recommandé : **{lang_complexity}**")
        st.info(f"Complexité estimée : **{big_o}**")
        st.write(f"Score : `{score}`")
        algos_actifs = [a.replace('algo_','') for a, v in algos.items() if v > 0]
        if algos_actifs:
            st.write("Algorithmes détectés : " + ", ".join(algos_actifs))

    with col2:
        st.markdown("**Prédiction ML**")
        for i, (lang, prob) in enumerate(ml_results):
            label = "1er choix" if i == 0 else "2ème choix"
            st.success(f"{label} : **{lang}** — {prob:.1%}")
