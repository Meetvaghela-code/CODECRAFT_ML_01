from flask import Flask, render_template, request, redirect, url_for, session
import pickle
import numpy as np

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Required for session

# Load model
with open('LinearModel.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/', methods=['GET'])
def home():
    # Get prediction from session if it exists
    prediction = session.pop('prediction', None)
    return render_template('index.html', prediction=prediction)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        total_sqft = float(request.form['total_sqft'])
        bhk = int(request.form['bhk'])
        bath = int(request.form['bath'])

        total_per_bhk = total_sqft / bhk
        bath_per_bhk = bath / bhk

        features = np.array([[total_sqft, bath, bhk, total_per_bhk, bath_per_bhk]])
        predicted_price = model.predict(features)[0]
        predicted_price = round(predicted_price, 2)

        # Store in session temporarily
        session['prediction'] = f"Estimated Price: ₹ {predicted_price} Lakhs"

        # Redirect to home
        return redirect(url_for('home'))

    except Exception as e:
        session['prediction'] = f"Error: {str(e)}"
        return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)
