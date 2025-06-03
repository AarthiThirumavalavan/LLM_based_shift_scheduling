import pandas as pd
from datetime import datetime, date, timedelta

def swap_shifts(df, emp1, emp2, shift_date):
    # Swap shifts between two employees on the same dates
    ent1 = (df["Employee Name"].str.lower() == emp1.lower()) & (df["Date"].dt.date == shift_date)
    ent2 = (df["Employee Name"].str.lower() == emp2.lower()) & (df["Date"].dt.date == shift_date)

    if not ent1.any() or not ent2.any():
        print("One or both employees have no shift on the given date.")
        return df

    temp = df.loc[ent1].copy()
    df.loc[ent1, ["Start Time", "End Time", "Shift Type", "Hours", "Location"]] = df.loc[ent2, ["Start Time", "End Time", "Shift Type", "Hours", "Location"]].values
    df.loc[ent2, ["Start Time", "End Time", "Shift Type", "Hours", "Location"]] = temp[["Start Time", "End Time", "Shift Type", "Hours", "Location"]].values
    return df


def reassign_shift(df , from_emp, to_emp, shift_date):
    # Reassign shift from one employee to another on a given date.
    emp_shift = (df["Employee Name"].str.lower() == from_emp.lower()) & (df["Date"].dt.date == shift_date)

    if not emp_shift.any():
        print("The mentioned employee has no shift on the given date.")
        return df

    df.loc[emp_shift, "Employee Name"] = to_emp
    return df


def remove_shift(df, emp, shift_date ):
    # Remove an employee's shift for a given date.
    entry_to_remove = (df["Employee Name"].str.lower() == emp.lower()) & (df["Date"].dt.date == shift_date)
    return df[~entry_to_remove]

def add_shift(df, employee_name, shift_date, start_time, end_time, shift_type, hours, location):
    # Add a new shift for an employee on a given date.
    new_shift = pd.DataFrame([{
        "Employee Name": employee_name,
        "Date": pd.to_datetime(shift_date),
        "Start Time": pd.to_datetime(start_time).time(),
        "End Time": pd.to_datetime(end_time).time(),
        "Shift Type": shift_type,
        "Hours": float(hours),
        "Location": location
    }])

    return pd.concat([df, new_shift], ignore_index=True)

def update_shift(df, employee_name, date, start_time, end_time, shift_type, hours, location, manager_name, **kwargs):
    # Update an existing shift for an employee on a given date.
    try:
        target_date_obj = pd.to_datetime(date).date()
    except Exception as e: 
        return f"Error processing date '{date}': {e}"

    employee_shift_to_upd = (df["Employee Name"].str.lower() == employee_name.lower()) & (df["Date"].dt.date == target_date_obj)

    if not employee_shift_to_upd.any():
        return f"Info: No shift found for employee '{employee_name}' on {target_date_obj.strftime('%Y-%m-%d')} to update."

    if employee_shift_to_upd.any():
        current_start_time_val = df.loc[employee_shift_to_upd, "Start Time"].iloc[0]
        current_end_time_val = df.loc[employee_shift_to_upd, "End Time"].iloc[0]  
    else:
        current_start_time_val = None
        current_end_time_val = None
        
    new_start_time = None
    if start_time is not None:
        try:
            new_start_time = pd.to_datetime(start_time).time()
        except ValueError:
            return f"Error: Invalid format for start_time: '{start_time}'. Please use HH:MM."
    else:
        new_start_time = current_start_time_val

    new_end_time = None
    if end_time is not None:
        try:
            new_end_time = pd.to_datetime(end_time).time()
        except ValueError:
            return f"Error: Invalid format for end_time: '{end_time}'. Please use HH:MM."
    else:
        new_end_time = current_end_time_val

    if new_start_time is not None and new_end_time is not None:
        if new_start_time >= new_end_time:
            return f"Validation Error: Start time ({new_start_time}) must be before end time ({new_end_time})."

    # Apply updates
    if start_time is not None:
        df.loc[employee_shift_to_upd, "Start Time"] = new_start_time
    if end_time is not None:
        df.loc[employee_shift_to_upd, "End Time"] = new_end_time
    if shift_type is not None:
        df.loc[employee_shift_to_upd, "Shift Type"] = shift_type
    if hours is not None:
        try:
            df.loc[employee_shift_to_upd, "Hours"] = float(hours)
        except ValueError:
            return f"Error: Invalid value for hours: '{hours}'. Must be a number."
    if location is not None:
        df.loc[employee_shift_to_upd, "Location"] = location
    if manager_name is not None:
        df.loc[employee_shift_to_upd, "Manager"] = manager_name

    return df
