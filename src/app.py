from flask import Flask, render_template, request, jsonify, abort
from openai import OpenAI
from pathlib import Path
from config import hg_token
from map_update import map_script
app = Flask(__name__)

# --- Hugging Face OpenAI-compatible API ---
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=hg_token,
)

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
    current_map_key = data.get("current_map", "GDP per capita (euro)")
    try:
        # completion = client.chat.completions.create(
        #     model="swiss-ai/Apertus-8B-Instruct-2509:publicai",
        #     messages=[{"role": "user", "content": user_message}],
        # )
        # response = completion.choices[0].message["content"]
        response = ["Gross value added (millions of euro), H Transportation and storage", "-35%"]
        
        # Create new map using map_script
        industry, change = response
        change_value = float(change.strip('%')) / 100  # Convert percentage to decimal
        
        # Initialize map_script with region (you might want to make this dynamic)
        map_handler = map_script(region_id=1, value_being_shown=current_map_key)
        
        # Update the map
        map_handler.update_map(industry, change_value)
        # Add the new map to available maps
        LIST_OF_MAPS["updated"] = "visualizations/updated_map.html"
    except Exception as e:
        response = f"Error: {e}"

    return jsonify({
        "response": response,
        "map_type": "updated"
        })

if __name__ == "__main__":
    app.run(debug=True)