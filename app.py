from flask import Flask, request, jsonify, send_from_directory
import numpy as np
import joblib
import os
import math
import json

app = Flask(__name__, static_folder='static', template_folder='templates')


MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'asd_model.pkl')
model_data = joblib.load(MODEL_PATH)

model        = model_data['model']
feature_cols = model_data['feature_cols']

METRICS_PATH = os.path.join(os.path.dirname(__file__), 'model', 'metrics.json')

with open(METRICS_PATH, 'r') as f:
    metrics = json.load(f)


FULL_FEATURE_MAP = {
    # Raw Q-CHAT items
    'A1': None, 'A2': None, 'A3': None, 'A4': None, 'A5': None,
    'A6': None, 'A7': None, 'A8': None, 'A9': None, 'A10': None,

    # Demographics
    'Age_Years': None,
    'Sex_M':     None,

    # Ethnicity flags
    'Ethnicity_WE': None,
    'Ethnicity_SA': None,
    'Ethnicity_ME': None,

    # Binary clinical flags
    'Jaundice':          None,
    'Family_ASD':        None,
    'Speech_Delay':      None,
    'Learning_Disorder': None,
    'Genetic':           None,
    'Depression':        None,
    'Global_Dev_Delay':  None,
    'Social_Issues':     None,
    'Anxiety':           None,

    # Who completed the screening
    'Completed_HCP': None,

    # Domain sub-scores (from engineer_features)
    'A_comm':        None,   # A1 + A2 + A9
    'A_social':      None,   # A3 + A4 + A7
    'A_imaginative': None,   # A5 + A6
    'A_sensory':     None,   # A8
    'A10_blank':     None,   # A10

    # Interaction terms
    'Qchat_x_Social':        None,   # A10_blank * Social_Issues
    'Qchat_x_Family':        None,   # A10_blank * Family_ASD
    'Speech_x_Learning':     None,
    'Genetic_x_GDD':         None,
    'Anxiety_x_Dep':         None,
    'SocialDef_x_SpeechDelay':  None,
    'SocialDef_x_SocialIssues': None,
    'A10_x_SocialIssues':    None,

    # Age group flags
    'Is_Toddler':    None,
    'Is_Child':      None,
    'Is_Adolescent': None,
    'Age_Group_Ord': None,

    # Aggregate clinical score
    'Clinical_Flag_Count': None,
}


def build_feature_vector(data: dict) -> np.ndarray:
    """
    Convert JSON form input → 1-D feature array aligned with feature_cols
    saved by train_model.py.

    Frontend conventions (same as training data):
      A1–A9  : 1 = behaviour ABSENT (atypical/risk)  |  0 = behaviour present (normal)
      A10    : 1 = blank staring PRESENT (risk item, reversed polarity)
    """

    # ── Q-CHAT items ────────────────────────────────────────────────────────
    a_items = [int(data.get(f'A{i}', 0)) for i in range(1, 11)]
    A1, A2, A3, A4, A5, A6, A7, A8, A9, A10 = a_items

    # ── Demographics ────────────────────────────────────────────────────────
    age   = float(data.get('age', 8))
    sex_m = 1 if str(data.get('sex', 'M')).upper() == 'M' else 0

    # ── Ethnicity ───────────────────────────────────────────────────────────
    eth    = str(data.get('ethnicity', 'other')).lower().strip()
    eth_we = 1 if eth in ('white_european', 'white european', 'white-european') else 0
    eth_sa = 1 if eth in ('south_asian', 'south asian', 'south-asian', 'asian')  else 0
    eth_me = 1 if eth in ('middle_eastern', 'middle eastern', 'middle-eastern')  else 0

    # ── Binary clinical flags ────────────────────────────────────────────────
    jaundice          = int(data.get('jaundice',          0))
    family_asd        = int(data.get('family_asd',        0))
    speech_delay      = int(data.get('speech_delay',      0))
    learning_disorder = int(data.get('learning_disorder', 0))
    genetic           = int(data.get('genetic',           0))
    depression        = int(data.get('depression',        0))
    global_dev_delay  = int(data.get('global_dev_delay',  0))
    social_issues     = int(data.get('social_issues',     0))
    anxiety           = int(data.get('anxiety',           0))

    # ── Who completed the screening ─────────────────────────────────────────
    completed_hcp = 1 if str(data.get('completed_by', 'parent')).lower() in (
        'hcp', 'health', 'doctor', 'clinician', 'professional'
    ) else 0

    # ── Domain sub-scores (mirror engineer_features exactly) ────────────────
    a_comm        = A1 + A2 + A9
    a_social      = A3 + A4 + A7
    a_imaginative = A5 + A6
    a_sensory     = A8
    a10_blank     = A10

    # ── Interaction terms (mirror engineer_features exactly) ─────────────────
    # train uses:  Qchat_x_Social = A10_blank * Social_Issues
    #              Qchat_x_Family = A10_blank * Family_ASD
    qchat_x_social        = a10_blank * social_issues
    qchat_x_family        = a10_blank * family_asd
    speech_x_learning     = speech_delay      * learning_disorder
    genetic_x_gdd         = genetic           * global_dev_delay
    anxiety_x_dep         = anxiety           * depression

    a_social_deficit          = A1 + A2 + A3 + A4 + A5 + A6 + A7 + A8 + A9
    social_def_x_speech_delay  = a_social_deficit * speech_delay
    social_def_x_social_issues = a_social_deficit * social_issues
    a10_x_social_issues        = a10_blank         * social_issues

    # ── Age group flags ─────────────────────────────────────────────────────
    is_toddler    = 1 if age <= 3       else 0
    is_child      = 1 if 3 < age <= 11 else 0
    is_adolescent = 1 if age > 11      else 0
    age_group_ord = is_toddler * 0 + is_child * 1 + is_adolescent * 2

    # ── Aggregate clinical flag count ────────────────────────────────────────
    clinical_flag_count = (
        jaundice + family_asd + speech_delay + learning_disorder
        + genetic + social_issues + global_dev_delay + anxiety + depression
    )

    # ── Build lookup dict (all possible engineered features) ────────────────
    all_features = {
        'A1': A1, 'A2': A2, 'A3': A3, 'A4': A4, 'A5': A5,
        'A6': A6, 'A7': A7, 'A8': A8, 'A9': A9, 'A10': A10,
        'Age_Years': age,
        'Sex_M':     sex_m,
        'Ethnicity_WE': eth_we,
        'Ethnicity_SA': eth_sa,
        'Ethnicity_ME': eth_me,
        'Jaundice':          jaundice,
        'Family_ASD':        family_asd,
        'Speech_Delay':      speech_delay,
        'Learning_Disorder': learning_disorder,
        'Genetic':           genetic,
        'Depression':        depression,
        'Global_Dev_Delay':  global_dev_delay,
        'Social_Issues':     social_issues,
        'Anxiety':           anxiety,
        'Completed_HCP':     completed_hcp,
        'A_comm':        a_comm,
        'A_social':      a_social,
        'A_imaginative': a_imaginative,
        'A_sensory':     a_sensory,
        'A10_blank':     a10_blank,
        'Qchat_x_Social':           qchat_x_social,
        'Qchat_x_Family':           qchat_x_family,
        'Speech_x_Learning':        speech_x_learning,
        'Genetic_x_GDD':            genetic_x_gdd,
        'Anxiety_x_Dep':            anxiety_x_dep,
        'SocialDef_x_SpeechDelay':  social_def_x_speech_delay,
        'SocialDef_x_SocialIssues': social_def_x_social_issues,
        'A10_x_SocialIssues':       a10_x_social_issues,
        'Is_Toddler':    is_toddler,
        'Is_Child':      is_child,
        'Is_Adolescent': is_adolescent,
        'Age_Group_Ord': age_group_ord,
        'Clinical_Flag_Count': clinical_flag_count,
    }

    # Only keep columns the saved model actually expects (in the correct order)
    row = [all_features.get(c, 0) for c in feature_cols]
    return np.array(row, dtype=float).reshape(1, -1)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    return send_from_directory('templates', 'index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        X = build_feature_vector(data)


        prob_raw = float(model.predict_proba(X)[0][1])


        def calibrate(p, T=1.3):
            p = min(max(p, 1e-6), 1 - 1e-6)
            logit = math.log(p / (1 - p))
            return 1 / (1 + math.exp(-logit / T))

        prob         = calibrate(prob_raw)
        risk_percent = round(prob * 100, 1)

        pred = 1 if prob >= 0.6 else 0

        if risk_percent < 25:
            risk = 'Low'
        elif risk_percent < 50:
            risk = 'Moderate'
        elif risk_percent < 75:
            risk = 'High'
        else:
            risk = 'Very High'

        # ── Key contributing factors ─────────────────────────────────────────
        qchat            = sum(int(data.get(f'A{i}', 0)) for i in range(1, 11))
        a_social_deficit = sum(int(data.get(f'A{i}', 0)) for i in range(1, 10))
        a10_blank        = int(data.get('A10', 0))

        factors = []
        if qchat >= 6:
            factors.append({'factor': 'Q-CHAT / AQ Score',
                            'value': f'{qchat}/10', 'weight': 'High'})
        if a_social_deficit >= 5:
            factors.append({'factor': 'Social Communication Deficit (A1–A9)',
                            'value': f'{a_social_deficit}/9', 'weight': 'High'})
        if a10_blank:
            factors.append({'factor': 'Atypical Blank Staring (A10)',
                            'value': 'Frequently Present', 'weight': 'Moderate'})
        if int(data.get('social_issues', 0)):
            factors.append({'factor': 'Social/Behavioral Issues',
                            'value': 'Present', 'weight': 'High'})
        if int(data.get('speech_delay', 0)):
            factors.append({'factor': 'Speech / Language Delay',
                            'value': 'Present', 'weight': 'Moderate'})
        if int(data.get('family_asd', 0)):
            factors.append({'factor': 'Family History of ASD',
                            'value': 'Present', 'weight': 'Moderate'})
        if int(data.get('learning_disorder', 0)):
            factors.append({'factor': 'Learning Disorder',
                            'value': 'Present', 'weight': 'Moderate'})
        if int(data.get('genetic', 0)):
            factors.append({'factor': 'Genetic Disorder',
                            'value': 'Present', 'weight': 'Moderate'})
        if int(data.get('global_dev_delay', 0)):
            factors.append({'factor': 'Global Developmental Delay',
                            'value': 'Present', 'weight': 'Moderate'})
        if int(data.get('anxiety', 0)):
            factors.append({'factor': 'Anxiety Disorder',
                            'value': 'Present', 'weight': 'Low'})
        if int(data.get('depression', 0)):
            factors.append({'factor': 'Depression',
                            'value': 'Present', 'weight': 'Low'})

        # ── Age-appropriate intervention suggestions ──────────────────────────
        age = float(data.get('age', 8))
        if pred == 1:
            if age <= 3:
                interventions = [
                    'Early intensive behavioral therapy (EIBI)',
                    'Speech-language therapy',
                    'Parent-mediated intervention (e.g. PACT)',
                    'Occupational therapy for sensory integration',
                ]
            elif age <= 11:
                interventions = [
                    'Applied Behavior Analysis (ABA)',
                    'Social skills training',
                    'Speech and communication therapy',
                    'Emotional regulation support',
                    'School-based support plan (IEP/504)',
                ]
            else:
                interventions = [
                    'Cognitive Behavioral Therapy (CBT)',
                    'Social communication training',
                    'Vocational / life-skills coaching',
                    'Family and peer counseling',
                ]
        else:
            interventions = [
                'Regular developmental monitoring',
                'Routine pediatric check-ups',
                'Reassess if new concerns arise',
            ]

        return jsonify({
            'prediction':       pred,
            'probability':      risk_percent,
            'risk_level':       risk,
            'key_factors':      factors,
            'interventions':    interventions,
            'qchat_score':      qchat,
            'a_social_deficit': a_social_deficit,
            'a10_blank':        a10_blank,
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/metrics')
def get_metrics():
    return jsonify({
        'accuracy':  round(metrics['accuracy'], 2),
        'precision': round(metrics['precision'], 2),
        'recall':    round(metrics['recall'], 2),
        'f1':        round(metrics['f1'], 2),
        'auc':       round(metrics['auc'], 2),
        'cv_mean':   round(metrics['cv_mean'], 2),
        'cv_std':    round(metrics['cv_std'], 2),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
