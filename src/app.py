from flask import Flask, render_template, request, jsonify, abort
from pathlib import Path
from config import hg_token
from llm import LLM

app = Flask(__name__)

LIST_OF_MAPS = {
    "disposable_income": "visualizations/disposable_income.html",
    "GDP_per_cap": "visualizations/GDP_per_cap.html",
    "unemployed": "visualizations/unemployed.html",
    "RnD": "visualizations/RnD.html",
}

@app.route("/")
def index():
    current_map = request.args.get("map", "GDP_per_cap")  # Default to disposable_income
    return render_template("index.html", current_map=current_map)

@app.route("/map")
def get_map():
    map_type = request.args.get("type")
    if map_type not in LIST_OF_MAPS:
        abort(404, description="Map not found")  # Returns 404 if invalid type
    map_path = LIST_OF_MAPS[map_type]
    try:
        html_content = Path(map_path).read_text(encoding="utf-8")
        return html_content  # Return raw HTML for iframe
    except FileNotFoundError:
        abort(404, description="Map file not found")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "")

    try:
        llm = LLM()
        response = llm.analyze_scenario(scenario = user_message, region_specific=False)
    except Exception as e:
        response = f"Error: {e}"

    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)