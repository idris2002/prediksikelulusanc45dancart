import os
import joblib
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)

# ==============================
# LOAD MODEL
# ==============================
def load_artifacts():
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))

        model_c45_path = os.path.join(base_dir, 'model_c45.pkl')
        model_cart_path = os.path.join(base_dir, 'model_cart.pkl')

        if not os.path.exists(model_c45_path) or not os.path.exists(model_cart_path):
            return None, None, 'Model tidak ditemukan di server!'

        model_c45 = joblib.load(model_c45_path)
        model_cart = joblib.load(model_cart_path)

        return model_c45, model_cart, None
    except Exception as e:
        return None, None, str(e)

model_c45, model_cart, load_error = load_artifacts()

# ==============================
# ROUTE
# ==============================
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_text = ''
    is_error = False

    if load_error:
        prediction_text = load_error
        is_error = True
        return render_template('index.html', prediction_text=prediction_text, is_error=is_error)

    if request.method == 'POST':
        try:
            attendance = float(request.form['attendance'])
            test1 = float(request.form['test1'])
            test2 = float(request.form['test2'])
            assignment = float(request.form['assignment'])
            study_hours = float(request.form['study_hours'])

            final_input = np.array([[attendance, test1, test2, assignment, study_hours]])

            pred_c45 = model_c45.predict(final_input)[0]
            prob_c45 = model_c45.predict_proba(final_input)[0][1] * 100

            pred_cart = model_cart.predict(final_input)[0]
            prob_cart = model_cart.predict_proba(final_input)[0][1] * 100

            result_c45 = 'LULUS' if pred_c45 == 1 else 'TIDAK LULUS'
            result_cart = 'LULUS' if pred_cart == 1 else 'TIDAK LULUS'

            prediction_text = f'''
            C4.5 → {result_c45} (Peluang: {prob_c45:.2f}%)<br>
            CART → {result_cart} (Peluang: {prob_cart:.2f}%)
            '''

        except Exception as e:
            prediction_text = f'Error: {e}'
            is_error = True

    return render_template('index.html', prediction_text=prediction_text, is_error=is_error)

if __name__ == '__main__':
    app.run()