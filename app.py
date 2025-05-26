from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import PredictionPipeline

app = Flask(__name__)

# Load the summarization pipeline
summarizer_pipeline = PredictionPipeline()

@app.route("/", methods=["GET", "POST"])
def home():
    summary = ""
    if request.method == "POST":
        input_text = request.form["text"]
        if input_text.strip():
            summary = summarizer_pipeline.predict(input_text)
    return render_template("index.html", summary=summary)

if __name__ == "__main__":
    print("Visit: http://localhost:8081")
    app.run(host="0.0.0.0", port=8081, debug=True)


