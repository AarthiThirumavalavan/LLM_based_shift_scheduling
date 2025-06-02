import pandas as pd
from datetime import datetime, date, timedelta
date_today = datetime.today().date()

def get_employee_schedule(df, employee_name, date):
    # Get the schedule for a specific employee on a given date.
    if date is None:
        target_date_obj = date_today
    try:
        # Convert the input string date to a datetime.date object
        target_date_obj = pd.to_datetime(date).date()
    except ValueError:
        return f"Error: Invalid date format provided: '{date}'. Please use YYYY-MM-DD."
    
    output = (df["Employee Name"].str.lower() == employee_name.lower()) & (df["Date"].dt.date == target_date_obj)
    return df[output].sort_values("Date")

def get_daily_schedule(df, date):
    # Get the daily schedule for all employees on a given date.
    if date is not None:
        try:
            date_obj = pd.to_datetime(date).date()
        except ValueError:
            return f"Error: Invalid date format provided: '{date}'. Please use YYYY-MM-DD."
    else:
        date_obj = date_today
    return df[df["Date"].dt.date == date_obj].sort_values("Start Time")

def get_employees_by_role(df, role):
    # Get a list of unique employees by their role.
    return df[df["Role"].str.lower() == role.lower()]["Employee Name"].dropna().unique().tolist()

def get_total_hours_by_employee(df, week_start_date):
    # Get total hours worked by each employee in a week starting from week_start_date.
    try:
        # Convert the week_start_date string to a datetime.date object
        # YYYY-MM-DD format
        start_date_obj = pd.to_datetime(week_start_date, format='%Y-%m-%d').date()
    except ValueError:
        return f"Error: Invalid date format for week_start_date: '{week_start_date}'. Please use YYYY-MM-DD format."

    week_end_obj = start_date_obj + timedelta(days=6)
    weekly_df = df[(df["Date"].dt.date >= start_date_obj) & (df["Date"].dt.date <= week_end_obj)]
    if weekly_df.empty:
        return pd.DataFrame(columns=["Employee Name", "Hours"]) # Return empty DataFrame with expected columns
    return weekly_df.groupby("Employee Name")["Hours"].sum().reset_index()

def check_max_hours(df , employee_name, week_start, max_hours=48):
    # Check if employee is within max working hours for a week.
    try:
        week_start_obj = pd.to_datetime(week_start).date()
        max_hours_float = float(max_hours) # Ensure max_hours is numeric
    except ValueError:
        return f"Error: Invalid format for week_start date ('{week_start}') or max_hours ('{max_hours}'). Please use YYYY-MM-DD for dates."
    except Exception as e: # Catch any other parsing errors
        return f"Error processing parameters: {e}"

    week_end_obj = week_start_obj + timedelta(days=6)
    output = (df["Employee Name"].str.lower() == employee_name.lower()) & \
             (df["Date"].dt.date >= week_start_obj) & \
             (df["Date"].dt.date <= week_end_obj)
    total_hours = df.loc[output, "Hours"].sum()
    if total_hours <= max_hours_float:
        return f"{employee_name} has worked within max working hours limit for the week."
    else:
        return f"{employee_name} has exceeded the max working hours limit for the week."

def check_rest_period(df, employee_name, min_rest_hours):
    # Check if employee has sufficient rest period between shifts.
    try:
        min_rest_hours_float = float(min_rest_hours) # Ensure min_rest_hours is numeric
    except ValueError:
        return "Error: Invalid format for min_rest_hours. Must be a number."
    except Exception as e:
        return f"Error processing min_rest_hours: {e}"
    emp_df = df[df["Employee Name"].str.lower() == employee_name.lower()].sort_values("Date")
    violations = []

    for i in range(1, len(emp_df)):
        prev_end = datetime.combine(emp_df.iloc[i - 1]["Date"], emp_df.iloc[i - 1]["End Time"])
        curr_start = datetime.combine(emp_df.iloc[i]["Date"], emp_df.iloc[i]["Start Time"])
        rest_hours = (curr_start - prev_end).total_seconds() / 3600 
        #print(delta)  # ➝ 12:30:00
        if rest_hours < min_rest_hours_float:
            violations.append((emp_df.iloc[i - 1]["Date"].date(), emp_df.iloc[i]["Date"].date()))

    if violations:
        violations_df = pd.DataFrame(violations, columns=["Previous Shift Date", "Current Shift Date"])
        return violations_df
    else:
        # Return an empty DataFrame with expected columns if no violations
        return pd.DataFrame(columns=["Previous Shift Date", "Current Shift Date"])


def get_shifts_by_date_range(df, start_date, end_date):
    # Get shifts within a specific date range.
    output = (df["Date"] >= start_date) & (df["Date"] <= end_date)
    return df[output]

def get_shifts_by_date(df, date):
    # Get shifts for a specific date.
    if date is None:
        date = date_today
    return df[df["Date"] == date]

def get_shifts_by_employee(df, employee_name, date):
    # Get shifts for a specific employee on a given date.
    df_filtered = df[df["Employee Name"].str.lower() == employee_name.lower()]   
    if date:
        df_filtered = df_filtered[df_filtered["Date"] == date]
    return df_filtered

def get_shifts_by_type(df, shift_type, date):
    # Get shifts of a specific type on a given date.
    output = df["Shift Type"].str.lower() == shift_type.lower()
    if date is None:
        date = date_today
    output &= df["Date"] == date
    return df[output]

def get_shifts_by_location(df, location, date):
    # Get shifts at a specific location on a given date.
    output = df[df["Location"].str.lower() == location.lower()]
    if date is None:
        date = date_today
    output = output[output["Date"] == date]
    return output

def get_schedule_this_week(df, employee_name):
    # Get the schedule for an employee for the current week.
    date_today = datetime.today().date()
    end_date = date_today + timedelta(days=7)
    output = (df["Employee Name"].str.lower() == employee_name.lower()) & \
           (df["Date"].dt.date >= date_today) & (df["Date"].dt.date <= end_date)
    return df[output]

def get_shifts_by_manager(df, manager_name):
    # Get shifts managed by a specific manager.
    return df[df["Manager"].str.lower() == manager_name.lower()]

def get_shifts_by_role(df, role):
    # Get shifts for a specific role.
    return df[df["Role"].str.lower() == role.lower()]

def get_shifts_by_manager_and_date(df, manager, date):
    # Get shifts managed by a specific manager on a given date.
    if date is None:
        date = date_today
    return df[(df["Manager"].str.lower() == manager.lower()) & (df["Date"] == date)]

def get_shifts_by_role_and_date(df, role, date):
    # Get shifts for a specific role on a given date.
    if date is None:
        date = date_today
    return df[(df["Role"].str.lower() == role.lower()) & (df["Date"] == date)]


