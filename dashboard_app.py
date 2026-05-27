from __future__ import annotations

import threading
import webbrowser
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template_string, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split


BASE_PATH = Path(__file__).resolve().parent
DATASET_PATH = BASE_PATH / 'depression_dataset_reddit_cleaned_es.csv'


HIGH_RISK_PATTERNS = [
    'suicid',
    'quitarme la vida',
    'acabar con todo',
    'me voy a matar',
    'quiero morir',
    'no quiero vivir',
    'terminar con mi vida',
]


EXAMPLE_TEXTS = [
    'Hoy tuve un buen día y avance en mis tareas',
    'Ya no quiero hablar con nadie, me siento vacío',
    'Que buen dia para salir',
    'Ya no aguanto mas voy a acabar con todo',
]


@dataclass
class DashboardModel:
    df: pd.DataFrame
    vectorizer: TfidfVectorizer
    model: LogisticRegression
    summary: dict



def load_and_train() -> DashboardModel:
    df = pd.read_csv(DATASET_PATH)
    df = df[['clean_text', 'is_depression']].copy()
    df['clean_text'] = (
        df['clean_text']
        .fillna('')
        .astype(str)
        .str.lower()
        .str.replace(r'\s+', ' ', regex=True)
        .str.strip()
    )
    df = df[df['clean_text'].str.len() > 0].copy()

    X = df['clean_text']
    y = df['is_depression']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    vectorizer = TfidfVectorizer(
        max_features=15000,
        ngram_range=(1, 2),
        min_df=2,
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)
    summary = {
        'rows': int(df.shape[0]),
        'columns': int(df.shape[1]),
        'positive_rate': float(df['is_depression'].mean()),
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred)),
        'recall': float(recall_score(y_test, y_pred)),
        'f1': float(f1_score(y_test, y_pred)),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
    }

    return DashboardModel(df=df, vectorizer=vectorizer, model=model, summary=summary)


MODEL = load_and_train()
APP = Flask(__name__)



def classify_risk(text: str) -> dict:
    normalized_text = ' '.join(str(text).lower().split()).strip()
    if not normalized_text:
        return {
            'risk_score': 0.0,
            'risk_level': 'bajo',
            'probability': 0.0,
            'rule_triggered': False,
            'matched_pattern': None,
        }

    matched_pattern = next((pattern for pattern in HIGH_RISK_PATTERNS if pattern in normalized_text), None)
    if matched_pattern:
        return {
            'risk_score': 95.0,
            'risk_level': 'alto',
            'probability': 0.95,
            'rule_triggered': True,
            'matched_pattern': matched_pattern,
        }

    text_vector = MODEL.vectorizer.transform([normalized_text])
    probability = float(MODEL.model.predict_proba(text_vector)[0, 1])
    risk_score = round(probability * 100, 2)

    if probability < 0.33:
        risk_level = 'bajo'
    elif probability < 0.66:
        risk_level = 'medio'
    else:
        risk_level = 'alto'

    return {
        'risk_score': risk_score,
        'risk_level': risk_level,
        'probability': probability,
        'rule_triggered': False,
        'matched_pattern': None,
    }


def render_result_html(result: dict) -> str:
    if not result:
        return ''

    level = result.get('risk_level', 'bajo')
    score = float(result.get('risk_score', 0.0))
    probability = float(result.get('probability', 0.0))
    rule_triggered = result.get('rule_triggered', False)
    matched_pattern = result.get('matched_pattern') or 'ninguna'

    if level == 'alto':
        result_class = 'high'
        badge_class = 'high'
    elif level == 'medio':
        result_class = 'medium'
        badge_class = 'medium'
    else:
        result_class = 'low'
        badge_class = 'low'

    recommendation_items = {
        'alto': [
            'Escalar de inmediato a supervision humana',
            'Revisar el contexto completo del mensaje',
        ],
        'medio': [
            'Monitorear y revisar manualmente',
        ],
        'bajo': [
            'Seguimiento estandar',
        ],
    }

    recommendations_html = ''.join(f'<li>{item}</li>' for item in recommendation_items.get(level, recommendation_items['bajo']))

    return f'''
    <div class="result {result_class}">
      <div class="result-head">
        <div>
          <h2 style="margin:0 0 6px;">Nivel: {level.upper()}</h2>
          <div style="color: var(--muted);">Probabilidad estimada: {probability:.2f}</div>
        </div>
        <div class="badge {badge_class}">Score {score:.1f}</div>
      </div>
      <div class="bar"><div class="fill" style="width: {score}%;"></div></div>
      <div class="grid-2">
        <div class="mini">
          <h3>Detalles</h3>
          <ul>
            <li>Patron de alerta: {'si' if rule_triggered else 'no'}</li>
            <li>Palabra clave: {matched_pattern}</li>
          </ul>
        </div>
        <div class="mini">
          <h3>Recomendacion</h3>
          <ul>
            {recommendations_html}
          </ul>
        </div>
      </div>
    </div>
    '''


DASHBOARD_TEMPLATE = """
<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Panel de riesgo emocional</title>
  <style>
    :root {
      --bg: #0b1020;
      --panel: rgba(14, 23, 45, 0.86);
      --panel-2: rgba(21, 34, 64, 0.9);
      --text: #eef2ff;
      --muted: #a8b3cf;
      --accent: #7dd3fc;
      --accent-2: #c084fc;
      --good: #4ade80;
      --warn: #fbbf24;
      --bad: #fb7185;
      --line: rgba(148, 163, 184, 0.18);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      background:
        radial-gradient(circle at top left, rgba(125, 211, 252, 0.16), transparent 28%),
        radial-gradient(circle at bottom right, rgba(192, 132, 252, 0.16), transparent 24%),
        linear-gradient(180deg, #07111f 0%, #0b1020 100%);
      color: var(--text);
      min-height: 100vh;
    }
    .wrap { max-width: 1280px; margin: 0 auto; padding: 28px; }
    .hero {
      display: grid;
      grid-template-columns: 1fr;
      gap: 20px;
      align-items: stretch;
      margin-bottom: 18px;
    }
    .card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 24px;
      box-shadow: 0 18px 50px rgba(0,0,0,0.35);
      backdrop-filter: blur(14px);
    }
    .hero-main { padding: 28px; }
    .eyebrow {
      display: inline-flex; align-items: center; gap: 8px;
      padding: 8px 12px; border-radius: 999px;
      background: rgba(125, 211, 252, 0.12); color: var(--accent);
      font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase; font-size: 12px;
    }
    h1 { margin: 16px 0 12px; font-size: clamp(32px, 5vw, 54px); line-height: 0.95; }
    .sub {
      color: var(--muted);
      font-size: 16px;
      line-height: 1.7;
      width: 100%;
      max-width: none;
      margin-top: 6px;
    }
    .metric-grid {
      display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-top: 22px;
    }
    .metric {
      background: var(--panel-2);
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 18px;
      min-height: 110px;
    }
    .metric .label { color: var(--muted); font-size: 13px; text-transform: uppercase; letter-spacing: 0.08em; }
    .metric .value { font-size: 34px; font-weight: 800; margin-top: 8px; }
    .metric .hint { color: var(--muted); font-size: 13px; margin-top: 8px; }
    .side h2, .panel h2 { margin: 0 0 14px; font-size: 22px; }
    .chip-row { display: flex; flex-wrap: wrap; gap: 8px; }
    .chip {
      padding: 8px 12px; border-radius: 999px; background: rgba(148, 163, 184, 0.12);
      color: var(--text); border: 1px solid var(--line); font-size: 13px;
    }
    .layout { display: grid; grid-template-columns: 1.05fr 0.95fr; gap: 18px; margin-top: 18px; }
    .panel { padding: 24px; }
    label { display: block; margin-bottom: 8px; color: var(--muted); font-size: 14px; }
    textarea {
      width: 100%; min-height: 210px; resize: vertical; border-radius: 18px;
      border: 1px solid var(--line); background: rgba(7, 12, 24, 0.9); color: var(--text);
      padding: 16px; font-size: 15px; line-height: 1.6; outline: none;
    }
    textarea:focus { border-color: rgba(125, 211, 252, 0.7); box-shadow: 0 0 0 4px rgba(125, 211, 252, 0.08); }
    .actions { display: flex; gap: 12px; align-items: center; margin-top: 14px; }
    .btn {
      display: inline-flex; align-items: center; justify-content: center; gap: 10px;
      background: linear-gradient(135deg, var(--accent), var(--accent-2)); color: #08111f;
      border: none; border-radius: 14px; padding: 13px 18px; font-weight: 800; cursor: pointer;
    }
    .btn.secondary { background: transparent; color: var(--text); border: 1px solid var(--line); }
    .result {
      border-radius: 20px; padding: 18px; margin-top: 18px; border: 1px solid var(--line);
      background: rgba(7, 12, 24, 0.7);
    }
    .result.high { border-color: rgba(251, 113, 133, 0.4); box-shadow: inset 0 0 0 1px rgba(251, 113, 133, 0.08); }
    .result.medium { border-color: rgba(251, 191, 36, 0.4); }
    .result.low { border-color: rgba(74, 222, 128, 0.3); }
    .result-head { display: flex; justify-content: space-between; gap: 12px; align-items: center; flex-wrap: wrap; }
    .badge { padding: 8px 12px; border-radius: 999px; font-weight: 800; font-size: 13px; }
    .badge.low { background: rgba(74, 222, 128, 0.14); color: #86efac; }
    .badge.medium { background: rgba(251, 191, 36, 0.14); color: #fde68a; }
    .badge.high { background: rgba(251, 113, 133, 0.14); color: #fecdd3; }
    .bar {
      margin-top: 16px; height: 18px; border-radius: 999px; background: rgba(148, 163, 184, 0.12); overflow: hidden;
    }
    .fill { height: 100%; border-radius: inherit; background: linear-gradient(90deg, var(--good), var(--warn), var(--bad)); }
    .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-top: 18px; }
    .mini {
      background: rgba(148, 163, 184, 0.08); border: 1px solid var(--line); border-radius: 16px; padding: 14px;
    }
    .mini h3 { margin: 0 0 10px; font-size: 15px; }
    ul { margin: 0; padding-left: 18px; color: var(--muted); }
    .footer { color: var(--muted); font-size: 13px; margin-top: 18px; }
    .table { width: 100%; border-collapse: collapse; margin-top: 12px; }
    .table th, .table td { text-align: left; padding: 10px 8px; border-bottom: 1px solid var(--line); }
    .table th { color: var(--muted); font-weight: 700; font-size: 13px; }
    .table td { font-size: 14px; }
    @media (max-width: 1060px) {
      .hero, .layout, .metric-grid, .grid-2 { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="hero">
      <div class="card hero-main">
        <div class="eyebrow">Panel de monitoreo emocional</div>
        <h1>Dashboard de detección temprana de riesgo textual</h1>
        <div class="sub">
          Nuestro modelo analiza mensajes escritos en español con un enfoque de clasificación textual para estimar riesgo emocional,
          identificar señales de alerta y apoyar la priorización de revisión humana desde una interfaz web clara y centralizada.
        </div>
        <div class="metric-grid">
          <div class="metric">
            <div class="label">Registros</div>
            <div class="value">{{ summary.rows }}</div>
            <div class="hint">Filas limpias usadas para entrenar el prototipo</div>
          </div>
          <div class="metric">
            <div class="label">Positivos</div>
            <div class="value">{{ '{:.1f}%'.format(summary.positive_rate * 100) }}</div>
            <div class="hint">Proporción de ejemplos marcados como depresión</div>
          </div>
          <div class="metric">
            <div class="label">Accuracy</div>
            <div class="value">{{ '{:.3f}'.format(summary.accuracy) }}</div>
            <div class="hint">Rendimiento general en el conjunto de prueba</div>
          </div>
          <div class="metric">
            <div class="label">F1</div>
            <div class="value">{{ '{:.3f}'.format(summary.f1) }}</div>
            <div class="hint">Balance entre precision y recall</div>
          </div>
        </div>
      </div>
    </div>

    <div class="layout">
      <div class="card panel">
        <h2>Analizar texto</h2>
        <form id="risk-form" method="post">
          <label for="text_input">Escribe un mensaje para evaluar el riesgo:</label>
          <textarea id="text_input" name="text_input" placeholder="Escribe aquí el texto a analizar...">{{ input_text }}</textarea>
          <div class="actions">
            <button class="btn" type="submit" name="action" value="analyze">Evaluar riesgo</button>
            <button class="btn secondary" type="button" id="clear-btn">Limpiar</button>
          </div>
        </form>

        <div id="result-container">
          {% if result %}
          {{ result_html | safe }}
          {% endif %}
        </div>
      </div>

      <div class="card panel">
        <h2>Ejemplos rapidos</h2>
        <div class="mini" style="margin-bottom:14px;">
          <h3>Textos de prueba</h3>
          <ul>
            {% for item in example_rows %}
            <li>{{ item }}</li>
            {% endfor %}
          </ul>
        </div>
      </div>
    </div>
  </div>
  <script>
    const form = document.getElementById('risk-form');
    const textInput = document.getElementById('text_input');
    const resultContainer = document.getElementById('result-container');
    const clearBtn = document.getElementById('clear-btn');

    function renderResult(result) {
      const level = result.risk_level || 'bajo';
      const score = Number(result.risk_score || 0);
      const probability = Number(result.probability || 0);
      const ruleTriggered = Boolean(result.rule_triggered);
      const matchedPattern = result.matched_pattern || 'ninguna';
      const recommendations = level === 'alto'
        ? ['Escalar de inmediato a supervision humana', 'Revisar el contexto completo del mensaje']
        : level === 'medio'
          ? ['Monitorear y revisar manualmente']
          : ['Seguimiento estandar'];

      resultContainer.innerHTML = `
        <div class="result ${level}">
          <div class="result-head">
            <div>
              <h2 style="margin:0 0 6px;">Nivel: ${level.toUpperCase()}</h2>
              <div style="color: var(--muted);">Probabilidad estimada: ${probability.toFixed(2)}</div>
            </div>
            <div class="badge ${level}">Score ${score.toFixed(1)}</div>
          </div>
          <div class="bar"><div class="fill" style="width: ${score}%;"></div></div>
          <div class="grid-2">
            <div class="mini">
              <h3>Detalles</h3>
              <ul>
                <li>Patron de alerta: ${ruleTriggered ? 'si' : 'no'}</li>
                <li>Palabra clave: ${matchedPattern}</li>
              </ul>
            </div>
            <div class="mini">
              <h3>Recomendacion</h3>
              <ul>
                ${recommendations.map((item) => `<li>${item}</li>`).join('')}
              </ul>
            </div>
          </div>
        </div>
      `;
    }

    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      const body = new URLSearchParams(new FormData(form));
      body.set('action', 'analyze');

      const response = await fetch('/', {
        method: 'POST',
        headers: {
          'X-Requested-With': 'XMLHttpRequest',
        },
        body,
      });

      const data = await response.json();
      renderResult(data.result);
      textInput.value = data.input_text || textInput.value;
    });

    clearBtn.addEventListener('click', () => {
      textInput.value = '';
      resultContainer.innerHTML = '';
      textInput.focus();
    });
  </script>
</body>
</html>
"""


@APP.route('/', methods=['GET', 'POST'])
def index():
    input_text = ''
    result = None
    ajax_request = request.headers.get('X-Requested-With') == 'XMLHttpRequest'
    if request.method == 'POST':
        action = request.form.get('action', 'analyze')
        if action == 'clear':
            input_text = ''
            result = None
        else:
            input_text = request.form.get('text_input', '').strip()
            if input_text:
                result = classify_risk(input_text)

    if ajax_request:
        return jsonify({
            'input_text': input_text,
            'result': result,
            'result_html': render_result_html(result),
        })

    return render_template_string(
        DASHBOARD_TEMPLATE,
        summary=MODEL.summary,
        result=result,
        input_text=input_text,
        example_rows=EXAMPLE_TEXTS,
        result_html=render_result_html(result),
    )



def open_browser_later(url: str) -> None:
    threading.Timer(1.2, lambda: webbrowser.open_new_tab(url)).start()


if __name__ == '__main__':
    url = 'http://127.0.0.1:5000'
    open_browser_later(url)
    APP.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)
