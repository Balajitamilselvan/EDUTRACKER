import streamlit as st
import google.generativeai as genai

# -----------------------------
# MANUAL API KEY CONFIGURATION
# -----------------------------
API_KEY = "AIzaSyDdsOHvj5IL3YHU69-Pxq6muKGdcqzWYZY"  # Replace with your key
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash-exp')

# -----------------------------
# SESSION HISTORY
# -----------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def get_response(prompt, difficulty="intermediate"):
    difficulty_prompts = {
        "beginner": "Explain this in simple terms for a beginner: ",
        "intermediate": "Provide a detailed explanation of: ",
        "advanced": "Give an in-depth technical analysis of: "
    }
    full_prompt = f"{difficulty_prompts[difficulty]}{prompt}"
    try:
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

def save_to_history(question, answer):
    st.session_state.history.append({"question": question, "answer": answer})

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="AI Learning Buddy", page_icon="🤖", layout="wide")

st.markdown(
"""
<style>
h1 {color: #4a4a4a; text-align: center;}
textarea, input, select {width: 100%; padding: 0.5rem; margin: 0.5rem 0; border-radius: 5px; border: 1px solid #ccc;}
button {padding: 0.5rem 1rem; border-radius: 5px; background-color: #3B82F6; color: white; border: none;}
button:hover {background-color: #2563eb;}

/* ✅ Dark theme box for responses */
.output-dark {
    background: #1e1e1e;
    color: #f1f1f1;
    padding: 1rem;
    border-radius: 5px;
    border: 1px solid #444;
    white-space: pre-wrap;
    font-family: Arial, monospace;
}
</style>
""", unsafe_allow_html=True
)

tab1, tab2, tab3 = st.tabs(["📚 Learn", "🧩 Quiz", "📈 History"])

with tab1:
    st.header("Learn Something New")
    user_prompt = st.text_area("What would you like to learn?", height=100)
    difficulty = st.select_slider("Difficulty", ["beginner", "intermediate", "advanced"], value="intermediate")

    if st.button("Get Answer", key="learn_button"):
        if user_prompt:
            with st.spinner("Generating response..."):
                response = get_response(user_prompt, difficulty)
                st.success("Here's your explanation:")
                st.markdown(f"<div class='output-dark'>{response}</div>", unsafe_allow_html=True)
                save_to_history(user_prompt, response)
        else:
            st.warning("Please enter a topic.")

with tab2:
    st.header("Generate Quiz")
    quiz_topic = st.text_input("Enter a topic for a quick quiz:")
    difficulty_quiz = st.select_slider("Difficulty", ["beginner", "intermediate", "advanced"], value="intermediate", key="quiz_diff")

    if st.button("Generate Quiz", key="quiz_button"):
        if quiz_topic:
            with st.spinner("Generating quiz..."):
                quiz_prompt = f"Create a 3-question quiz about {quiz_topic} suitable for {difficulty_quiz} level"
                quiz = get_response(quiz_prompt, difficulty_quiz)
                st.success("Here's your quiz:")
                st.markdown(f"<div class='output-dark'>{quiz}</div>", unsafe_allow_html=True)
        else:
            st.warning("Please enter a topic for the quiz.")

with tab3:
    st.header("Learning History")
    if not st.session_state.history:
        st.info("No history available yet.")
    else:
        for i, item in enumerate(st.session_state.history):
            with st.expander(f"Topic {i+1}", expanded=False):
                st.markdown(f"<div class='output-dark'><b>Question:</b> {item['question']}<br><br><b>Answer:</b> {item['answer']}</div>", unsafe_allow_html=True)

        if st.button("Clear History", key="clear_history"):
            st.session_state.history = []
            st.success("History cleared successfully!")
