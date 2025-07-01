import json
import os
os.environ['MKL_VERBOSE'] = '0' 
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import re
import pandas as pd
import numpy as np
from datetime import date, datetime
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from typing import Dict, List, Tuple
from langchain_groq import ChatGroq
from csv_parser import clean_schedule_df
from lookup_functions import *
from shift_functions import *
from dotenv import load_dotenv

# === Initialize Groq client ===
load_dotenv()
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"), 
    model="llama-3.3-70b-versatile", 
    temperature=0.2
)

class VectorScheduleAgent:
    def __init__(self, csv_file_path, examples_json_path, vector_db_path = "schedule_vector_db"):
        # Load and clean CSV
        self.df = clean_schedule_df(csv_file_path)
        self.employee_name_list = list(self.df['Employee Name'].dropna().unique()) # Ensure NaN values are dropped
        print("DataFrame loaded and cleaned")
        
        # Initialize vector database
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.examples = []
        self.index = None
        self.embeddings = None
        
        # Load vector database if exists, else create it.
        if os.path.exists(f"{vector_db_path}_data.pkl"):
            self.load_vector_db(vector_db_path)
        else:
            self.create_vector_db(examples_json_path, vector_db_path)

        self.llm = llm
        
        # Function mapping based on functions in shift_functions.py and lookup_functions.py
        self.function_map = {
            "get_shifts_by_manager_and_date": get_shifts_by_manager_and_date,
            "get_shifts_by_role_and_date": get_shifts_by_role_and_date,
            "add_shift": add_shift,
            "update_shift": update_shift,
            "get_employee_schedule": get_employee_schedule,
            "get_daily_schedule": get_daily_schedule,
            "check_max_hours": check_max_hours,
            "check_rest_period": check_rest_period,
            "get_total_hours_by_employee": get_total_hours_by_employee,
            "get_employees_by_role": get_employees_by_role,
            "get_shifts_by_date_range": get_shifts_by_date_range,
            "get_shifts_by_date": get_shifts_by_date,
            "get_shifts_by_employee": get_shifts_by_employee,
            "get_shifts_by_type": get_shifts_by_type,
            "get_shifts_by_location": get_shifts_by_location,
            "get_schedule_this_week": get_schedule_this_week,
            "get_shifts_by_manager": get_shifts_by_manager,
            "get_shifts_by_role": get_shifts_by_role,
            "swap_shifts": swap_shifts,
            "reassign_shift": reassign_shift,
            "remove_shift": remove_shift
        }

    def create_vector_db(self, examples_json_path, save_path): 
        ## CREATE VECTOR DATABASE USING EXAMPLES WRITTEN IN JSON FILE
        print("Creating vector database with examples from json file...")

        # Load examples
        with open(examples_json_path, 'r') as f:
            self.examples = json.load(f)

        # Create embeddings
        queries = [example['user_query'] for example in self.examples]
        self.embeddings = self.model.encode(queries)

        # Build FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)

        # Normalize for cosine similarity
        normalized_embeddings = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.index.add(normalized_embeddings.astype('float32'))

        # Save database
        self.save_vector_db(save_path)

    def save_vector_db(self, save_path):
        ## SAVE VECTOR DATABASE
        db_data = {
            'examples': self.examples,
            'embeddings': self.embeddings
        }

        # Save FAISS index
        faiss.write_index(self.index, f"{save_path}_index.faiss")

        # Save other data
        with open(f"{save_path}_data.pkl", 'wb') as f:
            pickle.dump(db_data, f)
        print("Vector database created and saved successfully.")

    def load_vector_db(self, load_path):
        ## LOAAD CREATED VECTOR DATABASE
        print("Loading existing vector database...")

        # Load FAISS index
        self.index = faiss.read_index(f"{load_path}_index.faiss")

        # Load other data
        with open(f"{load_path}_data.pkl", 'rb') as f:
            db_data = pickle.load(f)

        self.examples = db_data['examples']
        self.embeddings = db_data['embeddings']
        print(f"Vector database loaded with {len(self.examples)} examples")
    
    def find_similar_intent(self, user_query, top_k = 3):
        ## SEARCH FOR SIMILAR INTENT USING VECTOR DATABASE

        # Encode user query
        query_embedding = self.model.encode([user_query])
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        
        # Search for similar examples
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        # print("scores:", scores)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.examples):
                results.append((self.examples[idx], float(score)))
        # print("results:", results)
        return results # returns list of tuples (example, score)
    
    def extract_parameters_from_query(self, user_query, template_params):
        ## EXTRACT PARAMETERS FROM USER QUERY BASED ON TEMPLATE PARAMETERS
        
        params = template_params.copy()
        query_lower = user_query.lower()
        
        # Extract employee names (capitalize first letter)
        employee_pattern = r'\b([A-Z][a-z]+)\b'
        employees = re.findall(employee_pattern, user_query)
        # print("Extracted employees:", employees)
        # Map employee names to parameters if employee name is present in employee_name_list
        if employees:
            # print("Extracted employees:", employees)
            param_keys = ['employee_name', 'emp', 'from_emp', 'to_emp', 'emp1', 'emp2']
            matched_employees = [name for name in self.employee_name_list if any(emp.lower() in name.lower() for emp in employees)]
            # print("Matched employees:", matched_employees)
            # create a mapping of employee names to parameters
            for i, emp in enumerate(matched_employees[:len(param_keys)]):
                for key in param_keys:
                    if key in params:
                        params[key] = emp
                        # print("params[key]:", params[key])
                        # print("-------")
                        break

        # Extract dates YYYY-MM-DD format
        date_pattern = r'\b(\d{4}-\d{2}-\d{2})\b'
        dates = re.findall(date_pattern, user_query)
        if dates:
            date_keys = ['date', 'shift_date', 'week_start', 'start_date', 'end_date']
            for i, date_val in enumerate(dates[:len(date_keys)]):
                for key in date_keys:
                    if key in params:
                        params[key] = date_val
                        break
        
        # Extract times HH:MM format
        time_pattern = r'\b(\d{1,2}:\d{2})\b'
        times = re.findall(time_pattern, user_query)
        if len(times) >= 1 and 'start_time' in params:
            params['start_time'] = times[0]
        if len(times) >= 2 and 'end_time' in params:
            params['end_time'] = times[1]
        
        # Extract roles
        self.roles_list = ['Manager', 'Cashier', 'Stock', 'Security']
        for role in self.roles_list:
            if role.lower() in query_lower and 'role' in params:
                params['role'] = role
                print("params['role']:", params['role'] )
                break
        
        # Extract locations
        self.location_mapping_list = ['Warehouse', 'Store A', 'Store B']
        for location in self.location_mapping_list:
            if location.lower() in query_lower and 'location' in params:
                params['location'] = location
                print("params['location']:", params['location'] )
                break        
        
        # Extract shift types
        self.shift_types_list = ['Morning', 'Afternoon', 'Night']
        for shift_type in self.shift_types_list:
            if shift_type.lower() in query_lower and 'shift_type' in params:
                params['shift_type'] = shift_type
                break

        # Extract hours
        hours_pattern = r'\b(\d+)\s*hours?\b'
        hours_match = re.search(hours_pattern, query_lower)
        if hours_match and 'hours' in params:
            params['hours'] = hours_match.group(1)
        
        # Extract manager names
        manager_pattern = r'manager\s+([A-Z][a-z]+)|([A-Z][a-z]+)\s+(?:is\s+)?managing'
        manager_match = re.search(manager_pattern, user_query)
        if manager_match and 'manager' in params:
            params['manager'] = manager_match.group(1) or manager_match.group(2)
        
        return params
    
    def process_user_query(self, user_query, similarity_threshold = 0.5): #Vector based similarity threshold set to 0.5
        ## USING VECTOR BASED SIMILARITY TO PROCESS USER QUERY AND FIND INTENT OF BEST MATCHING EXAMPLE.
        print(f"Processing query: {user_query}")
        
        # Find similar examples
        similar_examples = self.find_similar_intent(user_query, top_k=3)
        
        if not similar_examples:
            return "No similar examples found in the database."
        
        best_match, confidence = similar_examples[0]
        print(f"Best match: '{best_match['user_query']}' (confidence: {confidence:.3f})")
        
        # If confidence is too low, try LLM fallback
        if confidence < similarity_threshold:
            if self.llm:
                return self.llm_fallback(user_query)
            else:
                return f"Low confidence match ({confidence:.3f}). Please be more specific or rephrase your query."
        
        # Extract intent and parameters
        intent = best_match['intent']
        template_params = best_match['parameters']
        
        # Extract actual parameters from user query
        extracted_params = self.extract_parameters_from_query(user_query, template_params)
        
        # print(f"Intent: {intent}")
        # print(f"Parameters: {extracted_params}")
        
        # Execute function
        if intent not in self.function_map:
            return f"Function '{intent}' not implemented."
        
        try:
            result = self.function_map[intent](self.df, **extracted_params)
            
            if isinstance(result, pd.DataFrame):
                if not result.empty:
                    return result
                else:
                    return "No matching results found."
            else:
                return str(result)
        
        except Exception as e:
            return f"Error executing '{intent}': {str(e)}"
    
    def llm_fallback(self, user_query):
        ##FALLBACK TO LLM IF VECTOR BASED SIMILARITY IS LOW OR NO MATCH FOUND
        if not self.llm:
            return "LLM fallback not available."
        
        function_list = ", ".join(self.function_map.keys())
        system_prompt = f"""
        You are a scheduling assistant. Based on the user's query, extract the intent and parameters.
        Available functions: {function_list}. Use the available functions to answer the query. 
        If you cannot find a match, provide a fallback response. Your response should be in the following 
        JSON format: 
        ``` Sorry, I was not able to find a match for your query. Please try rephrasing it. ```
        
        Output JSON with:
        - "intent": function name
        - "parameters": dictionary of parameters
        """
        
        try:
            ai_response = self.llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ])
            
            content = ai_response.content
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', content, re.DOTALL)
            
            if json_match:
                parsed_json = json.loads(json_match.group(1))
                intent = parsed_json.get("intent")
                params = parsed_json.get("parameters", {})
                
                if intent in self.function_map:
                    result = self.function_map[intent](self.df, **params)
                    if isinstance(result, pd.DataFrame) and not result.empty:
                        return result
                    else:
                        str(result)
            return "Could not parse LLM response."
        
        except Exception as e:
            return f"LLM fallback error: {str(e)}"


# Example usage and testing
def main():
    # File paths
    csv_file_path = "./resource/shift_schedule.csv"
    examples_json_path = "examples.json"
    
    # Initialize enhanced agent
    agent = VectorScheduleAgent(csv_file_path, examples_json_path)
    
    # Test queries
    test_queries = [
        "Show me all Security shifts on 2025-04-01",
        "What is Charlie's schedule on 2025-04-01?",
        "List all Stock shifts?",
        "Who is working at Warehouse on 2025-04-01?",
        "Add a shift for Alice on 2025-04-02 from 09:00 to 17:00", #Not working correctly
        "Update Bob's shift on 2025-04-01 to start at 15:00" #Not working correctly
    ]
    
    print("=== Test queries execution ===")
    for query in test_queries:
        print(f"\nQuery: {query}")
        result = agent.process_user_query(query)
        if isinstance(result, pd.DataFrame):
            print(f"Result: DataFrame with {len(result)} rows")
            print(result.head())
            print("*"*50)
        else:
            print(f"Result: {result}")


if __name__ == "__main__":
    main()    
