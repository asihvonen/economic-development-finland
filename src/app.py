import streamlit as st
import pandas as pd
import numpy as np
import prophet
from prophet.plot import plot_plotly, plot_components_plotly
import plotly.io as pio
from pathlib import Path
import streamlit.components.v1 as components
from openai import OpenAI
from config import hg_token
# CONFIG
st.set_page_config(
    page_title="GDP Dashboard",
    layout="wide",         
    initial_sidebar_state="expanded"
)
# '==========================================================================='
# 'TITLE'
# '==========================================================================='
st.markdown("""
<h1 style='
    color:#FF6A10; 
    font-size:50px; 
    font-weight:bold; 
    text-align:center;
'>
    Analysis of Finland Economic Development
</h1>
""", unsafe_allow_html=True)
DATA_URL = "data/finland_gdp_by_quarter_(large).csv"

# ============================================================================
# FUNCTIONS
# ============================================================================
@st.cache_data
def load_data():
    data = pd.read_csv(DATA_URL)
    data = data.rename(columns={'observation_date':'ds','CLVMNACSCAB1GQFI':'y'})
    return data
def profet(df):
    p = prophet.Prophet()
    p.fit(df)
    future = p.make_future_dataframe(periods=80, freq='Q')
    forecast = p.predict(future)
    return (p,forecast)
@st.cache_data
def load_html_map(path):
    return Path(path).read_text(encoding="utf-8")
@st.cache_data
def load_html_plot(path):
    return Path(path).read_text(encoding="utf-8")
LIST_OF_MAPS = [
    "visualizations/disposable_income.html",
    "visualizations/GDP_per_cap.html",
    "visualizations/unemployed.html",
    "visualizations/RnD.html",
]
MAP_NAMES = ["Disposable Income", "GDP per capita", "Unemployement", "RnD"]

# '=========================================================================='
# LLM MODEL
# '=========================================================================='
# --- HUGGING FACE API SETUP ---
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=hg_token,
)

def get_llm_response(user_message):
    """Call the HF OpenAI-compatible API and return the assistant's message."""
    completion = client.chat.completions.create(
        model="swiss-ai/Apertus-8B-Instruct-2509:publicai",
        messages=[
            {"role": "user", "content": user_message}
        ],
    )
    return completion.choices[0].message["content"]



# '=========================================================================='
# 'START OF SECTION 1'
# '=========================================================================='
st.markdown(
        f"<h2 style='font-size:45px; text-align:center; color:#FFFFFF; font-weight:bold'>Models fit to data</h2>", 
        unsafe_allow_html=True
    )

plots = {
    "ARIMA": "visualizations/arima_forecast.html",
    "Prophet": "visualizations/prophet_forecast.html",
    "ASTAR (autoregressive MARS)" :  "visualizations/mars_forecast.html",
}
# Dropdown menu with different plots
choice = st.selectbox("Select a model:", list(plots.keys()))
html_file = load_html_plot(plots[choice])
# Sketchy solution for resizing
responsive_html = f"""
<div id="plot-container" style="width:100%; height:90vh; margin:auto;">
    {html_file}
</div>

<script>
    function resizePlot() {{
        const container = document.getElementById('plot-container');
        const plot = container.querySelector('.plotly-graph-div');
        if (!plot) return;

        // Read actual container dimensions
        const w = container.clientWidth;
        const h = container.clientHeight;

        // Resize Plotly plot to match container
        if (window.Plotly) {{
            window.Plotly.relayout(plot, {{
                width: w,
                height: h
            }});
        }}
    }}

    // Run on load and whenever window resizes
    window.addEventListener('load', resizePlot);
    window.addEventListener('resize', resizePlot);
</script>
"""

components.html(responsive_html, height=900, scrolling=False)



# '============================================================================'
# 'START OF SECTION 2'
# '============================================================================'
st.markdown(
        f"<h2 style='font-size:45px; text-align:center; color:#FF6A10; font-weight:bold'>Interactive map</h2>", 
        unsafe_allow_html=True
    )
if "selected_map_index" not in st.session_state:
    st.session_state.selected_map_index = 0
col1, col2 = st.columns(2) 

# 'LEFT COLUMN - MAP SELECTION AND DISPLAY'

with col1:
    button_cols = st.columns(len(MAP_NAMES))
    
    for i, b_col in enumerate(button_cols):
        with b_col:
            if st.button(MAP_NAMES[i], key=f"map_btn_{i}", use_container_width=True):
                st.session_state.selected_map_index = i
    
    selected_map = LIST_OF_MAPS[st.session_state.selected_map_index]
    selected_map_name = MAP_NAMES[st.session_state.selected_map_index]
    st.markdown(
        f"<p style='font-size:35px; text-align:center; color:#FF6A10; font-weight:bold'>{selected_map_name}</p>", 
        unsafe_allow_html=True
    )

    
    html_fig = load_html_map(selected_map)
    components.html(html_fig, height=600)

# 'RIGHT COLUMN - LLM CHATBOT'

with col2:
    st.markdown(
        f"<h3 style='font-size:35px; text-align:center; color:#FFFFFF; font-weight:bold'>Your personal assistant</h3>", 
        unsafe_allow_html=True
    )

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # --- CHAT INPUT AREA ---
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input("Type your question...", key="user_message")
        submitted = st.form_submit_button("Send")

    # --- HANDLE INPUT ---
    if submitted and user_input:
        # Get response from HF API
        response = get_llm_response(user_input)

        # Add to chat history
        st.session_state.chat_history.append({"role": "user", "message": user_input})
        st.session_state.chat_history.append({"role": "llm", "message": response})
    # --- DISPLAY CHAT HISTORY ---
    chat_html = """
    <div style='height:500px; overflow-y:auto; padding:10px; border:1px solid #ddd; 
                border-radius:5px; background-color:white;'>
    """
    for chat in st.session_state.chat_history:
        color = "blue" if chat["role"] == "user" else "green"
        chat_html += f"<p style='color:{color}; margin:2px 0'><b>{chat['role'].capitalize()}:</b> {chat['message']}</p>"
    chat_html += "</div>"

    st.markdown(chat_html, unsafe_allow_html=True)

    # --- CLEAR CHAT BUTTON ---
    if st.button("🗑️ Clear Conversation"):
        st.session_state.chat_history = []
        st.rerun()
