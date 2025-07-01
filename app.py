import streamlit as st
import pandas as pd
from smart_agent import VectorScheduleAgent


st.set_page_config(
    page_title="Smart Schedule Agent",
    layout="wide"
)
def load_agent():
    csv_file_path = "./resource/shift_schedule.csv" 
    examples_json_path = "examples.json" 
    try:
        agent = VectorScheduleAgent(csv_file_path, examples_json_path)
        return agent
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None

agent = load_agent()

st.title("Smart Schedule Agent")

if agent:
    # Input field for user query
    user_query = st.text_input("Enter your query:")
    if st.button("Submit Query", key="process_query_button"): 
        if user_query: # Check if the query is not empty
            with st.spinner("Processing your query..."):
                try:
                    result = agent.process_user_query(user_query)
                    if isinstance(result, pd.DataFrame):
                        if not result.empty:
                            st.dataframe(result)
                        else:
                            st.info("No matching results found in the schedule.")
                    elif isinstance(result, str):
                        st.markdown(result)
                    else:
                        st.write(result)
                except Exception as e:
                    st.error(f"An error occurred while processing your query: {e}")
        else: # If the query is empty when the button is pressed
            st.warning("Please enter a query.")
