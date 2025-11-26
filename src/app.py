from flask import Flask, render_template, request, jsonify, abort
from pathlib import Path
from config import hg_token
from map_update import map_script
from llm import LLM
#from llm import LLM

app = Flask(__name__)

MAP_DEFINITIONS = {
    "disposable_income": {"path": "visualizations/disposable_income.html", "name": "Disposable income, net"},
    "GDP_per_cap": {"path": "visualizations/GDP_per_cap.html", "name": "GDP per capita (euro at current prices)"},
    "business_vitality_index": {"path": "visualizations/business_vitality_index.html", "name": "Business Vitality"},
    "demographic_sustainability_index": {"path": "visualizations/demographic_sustainability_index.html", "name": "Demographic Sustainability"},
    "economic_prosperity_index": {"path": "visualizations/economic_prosperity_index.html", "name": "Economic Prosperity"},
    "no_slider": {"path": "visualizations/no_slider.html", "name": "More info"},
    "updated": {"path": "visualizations/updated_map.html", "name": "Updated Scenario Map (Simulated)"},
}

# The list of maps initially available in the dropdown
INITIAL_MAP_KEYS = ["disposable_income", "GDP_per_cap", "business_vitality_index", "demographic_sustainability_index", "economic_prosperity_index","no_slider"]

# Global variable (or temporary store) to manage the current list of available map keys
# NOTE: In a real app, this should be managed per-session.
CURRENT_AVAILABLE_MAP_KEYS = INITIAL_MAP_KEYS.copy()

@app.route("/")
def index():
    global CURRENT_AVAILABLE_MAP_KEYS
    # Check if 'updated' map should be in the list
    if "updated" in request.args:
        CURRENT_AVAILABLE_MAP_KEYS = INITIAL_MAP_KEYS + ["updated"]
    else:
        # If no "updated" flag, reset to initial maps
        CURRENT_AVAILABLE_MAP_KEYS = INITIAL_MAP_KEYS
        
    current_map_key = request.args.get("map", "GDP_per_cap")
    
    # Check if the requested map key is valid for the current list
    if current_map_key not in CURRENT_AVAILABLE_MAP_KEYS:
        current_map_key = "GDP_per_cap" # Fallback

    # Pass the list of available maps and their definitions to the template
    available_maps = {key: MAP_DEFINITIONS[key] for key in CURRENT_AVAILABLE_MAP_KEYS}

    return render_template("index.html", current_map_key=current_map_key, available_maps=available_maps)

@app.route("/map")
def get_map():
    map_type = request.args.get("type")
    if map_type not in MAP_DEFINITIONS:
        abort(404, description="Map not found")
    map_path = MAP_DEFINITIONS[map_type]["path"]
    try:
        html_content = Path(map_path).read_text(encoding="utf-8")
        return html_content
    except FileNotFoundError:
        abort(404, description="Map file not found")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "")
    current_map_key = data.get("current_map", "GDP per capita (euro at current prices)")
    try:
        llm = LLM()
        response = llm.analyze_scenario(scenario = user_message) # needs to be of form tuple[industry, change]           
        # response = ("Gross value added (millions of euro), G Wholesale and retail trade; repair of motor vehicles and motorcycles", "35%")
        # Create new map using map_script
        industry, change, summary = response
        change_value = float(change.strip('%')) / 100  # Convert percentage to decimal
        
        # Initialize map_script with region (you might want to make this dynamic)
        map_handler = map_script(region_id=1, value_being_shown=current_map_key)
        
        # Update the map
        map_handler.update_map(industry, change_value)
        # Add the new map to available maps
        map_type = "updated"
    except Exception as e:
        response = f"Error: {e}"
        map_type = None

    return jsonify({
        "response": response,
        "map_type": map_type
        })

if __name__ == "__main__":
    app.run(debug=True)