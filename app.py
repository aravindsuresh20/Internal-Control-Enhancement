from flask import Flask, render_template, request, send_file # Import send_file
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load your pre-trained model
try:
    model = joblib.load("risk_model.pkl")
except FileNotFoundError:
    print("Error: 'risk_model.pkl' not found. Please ensure the model file is in the same directory.")
    model = None
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Define the outputs directory and the Excel file path
OUTPUTS_DIR = "outputs"
EXCEL_FILE = os.path.join(OUTPUTS_DIR, "predictions_log.xlsx")

# Function to run basic EDA and save a report
def run_basic_eda():
    try:
        df = pd.read_csv("audit_data.csv")
        report = []
        report.append("🟢 DATA OVERVIEW\n")
        report.append(str(df.head(10)))
        report.append("\n\n🔢 DATA TYPES\n")
        report.append(str(df.dtypes))
        report.append("\n\n📊 NULL VALUES\n")
        report.append(str(df.isnull().sum()))
        report.append("\n\n📈 DESCRIPTIVE STATS\n")
        report.append(str(df.describe()))
        full_report = "\n".join(report)
        
        # Ensure the outputs directory exists for EDA report as well
        os.makedirs(OUTPUTS_DIR, exist_ok=True)
        eda_report_path = os.path.join(OUTPUTS_DIR, "basic_eda_report.txt")
        with open(eda_report_path, "w", encoding="utf-8") as f:
            f.write(full_report)
        print(f"\n✅ EDA saved to {eda_report_path}")
    except FileNotFoundError:
        print("Warning: 'audit_data.csv' not found. Skipping EDA.")
    except Exception as e:
        print(f"Error running EDA: {e}")

# Route for the landing page
@app.route('/')
def landing_page():
    return render_template("landing.html")

# Route for the input form page
@app.route('/input')
def input_page():
    run_basic_eda()
    return render_template("input.html")

# Route for handling predictions from the form
@app.route('/predict', methods=["POST"])
def predict():
    inputs = {}
    if model is None:
        return render_template("result.html", prediction="Error: Model not loaded.", **inputs)

    try:
        # Retrieve form data
        inputs = {
            "Audit_Risk": float(request.form["Audit_Risk"]),
            "Inherent_Risk": float(request.form["Inherent_Risk"]),
            "Score": float(request.form["Score"]),
            "TOTAL": float(request.form["TOTAL"]),
            "Money_Value": float(request.form["Money_Value"])
        }
        # Prepare input for the model
        input_list = list(inputs.values())
        # Make a prediction
        prediction_value = model.predict([input_list])[0]
        # Determine the risk result based on prediction
        result = "High Risk" if prediction_value == 1 else "Low Risk"

        # --- Save to Excel ---
        new_entry = inputs.copy()
        new_entry['Predicted_Risk'] = result
        
        # Create a DataFrame from the new entry
        new_df = pd.DataFrame([new_entry])

        # Ensure the outputs directory exists before writing the file
        os.makedirs(OUTPUTS_DIR, exist_ok=True)

        # Check if the Excel file exists within the outputs directory
        if os.path.exists(EXCEL_FILE):
            existing_df = pd.read_excel(EXCEL_FILE)
            updated_df = pd.concat([existing_df, new_df], ignore_index=True)
            updated_df.to_excel(EXCEL_FILE, index=False)
        else:
            new_df.to_excel(EXCEL_FILE, index=False)
        print(f"Prediction and inputs saved to {EXCEL_FILE}")
        # --- End Save to Excel ---

    except ValueError:
        result = "Invalid input: Please ensure all fields are numbers."
        inputs = {}
    except Exception as e:
        result = f"An unexpected error occurred during prediction: {e}"
        inputs = {}
    
    # Render the result page with the prediction and all input values
    return render_template("result.html", prediction=result, **inputs)

# New route for downloading the Excel file
@app.route('/download_predictions_log')
def download_predictions_log():
    if os.path.exists(EXCEL_FILE):
        return send_file(EXCEL_FILE, as_attachment=True, download_name="predictions_log.xlsx")
    else:
        return "File not found.", 404 # Or render an error page
        
# Run the Flask application
if __name__ == "__main__":
    app.run(debug=True)