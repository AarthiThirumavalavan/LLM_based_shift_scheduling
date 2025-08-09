# LLM_based_shift_scheduling
A simple automation tool that can parse and interpret shift scheduling data using LLM. It uses a combination of vector similarity search for intent recognition and a Large Language Model (LLM) for fallback and potentially more complex understanding.

<img width="1600" height="900" alt="image" src="https://github.com/user-attachments/assets/e153574e-2ca8-4284-aaf5-0711cb9fe31c" />

## File Structure

* smart_agent.py                ---> Main application logic for the agent
* examples.json                 ---> Example queries for intent matching and vector DB creation
* resource/shift_schedule.csv   ---> CSV file containing the shift schedule data
* lookup_functions.py           ---> Contains functions for querying schedule data
* shift_functions.py            ---> Contains functions for modifying schedule data (add, update, remove)
* csv_parser.py                 ---> Contains functions for loading and cleaning the CSV data
* requirements.txt              ---> Python package dependencies
* .env                          ---> For storing API key

## How to run

* To run via streamlit ---> streamlit run app.py
* To run via CLI ---> python smart_agent.py


