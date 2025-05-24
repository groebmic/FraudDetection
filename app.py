import gradio as gr
import joblib
import numpy as np

# Modell laden
model = joblib.load("best_model.joblib")

# Vorhersagefunktion
def predict_fraud(amt, is_holiday):
    input_data = np.array([[is_holiday, amt]])
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        return "⚠️ Betrug erkannt"
    else:
        return "✅ Keine Betrugsabsicht"

# Gradio-Interface definieren
interface = gr.Interface(
    fn=predict_fraud,
    inputs=[
        gr.Number(label="Transaktionsbetrag (amt)"),
        gr.Radio([0, 1], label="Feiertag? (0 = Nein, 1 = Ja)")
    ],
    outputs=gr.Text(label="Vorhersage"),
    title="💳 Kreditkarten-Betrugserkennung",
    description="Gib den Betrag und an, ob die Transaktion an einem Feiertag stattfindet, um vorherzusagen, ob es sich um Betrug handelt.",
)

# Start Gradio-App
if __name__ == "__main__":
    interface.launch()
