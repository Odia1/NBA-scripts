import streamlit as st
import pandas as pd
import random
import os
import io

st.title("Assignment of Marks from a total with CO score computed")

st.markdown("""
- The first row specifies a number of sections as (NumQuesInSection,RequiredToAnswer) pairs.
- The 4th header row (`Mark Per Que` in row 3) is per-question maxima.
- For each section, only `Required` number of questions get marks; others get zero.
- Marks are integers if possible (for integer maxima), else half marks allowed. Choose if half-marks allowed. Partial Marks allowed for a question.
- The student's total marks column is the last (5th after question columns end) column.
-Row5 maps COs to questions. The values in Blooms Taxonomy is ignored.
- Student records start at row 6.    --Prof. Priyadrsan Patra
""")

# User choice widget is now always visible
mark_granularity = st.radio(
    "Allow partial marks (multiples of 0.5)?",
    ("No", "Yes"),
    horizontal=True,
    key="granularity_choice" # Added a key here for good practice
)


# File uploader with a unique key to prevent errors
uploaded_file = st.file_uploader("Upload the input CSV", type=["csv"], key="file_uploader")

def partition_with_steps(total, per_q_max, choose, step_size):
    """
    Partitions 'total' among 'choose' questions, with marks as multiples of step_size (1 or 0.5).
    """
    steps = int(round(1 / step_size))
    n = len(per_q_max)
    int_total = int(round(total * steps))
    int_max_bounds = [int(round(m * steps)) for m in per_q_max]

    req = int(choose)
    if req == 0:
        return [0.0] * n if abs(total) < 1e-9 else None

    if req >= n:
        indices = list(range(n))
    else:
        indices = sorted(random.sample(range(n), req))

    max_selected = [int_max_bounds[i] for i in indices]
    if int_total > sum(max_selected):
        return None

    parts = []
    remaining = int_total
    for j in range(req):
        max_sum_rest = sum(max_selected[j+1:])
        lo = max(0, remaining - max_sum_rest)
        hi = min(max_selected[j], remaining)

        if lo > hi: return None

        if j < req - 1:
            val = random.randint(lo, hi)
        else:
            val = remaining
        parts.append(val)
        remaining -= val

    if remaining != 0: return None

    result = [0.0] * n
    for idx, p in zip(indices, parts):
        result[idx] = p / steps
    return result

if uploaded_file:
    step_size = 0.5 if mark_granularity == "Yes" else 1.0

    df_raw = pd.read_csv(uploaded_file, header=None, dtype=str)
    n_rows, n_cols = df_raw.shape

    # --- Parse Template Structure ---
    header_row = df_raw.iloc[1].str.strip().tolist()
    
    # Find start and end of question columns dynamically
    question_start_col = 2  # Assuming 'Name' and 'Roll No' are first two
    try:
        first_co_col_index = header_row.index('CO1')
        question_end_col = first_co_col_index
    except ValueError:
        st.error("Could not find 'CO1' in the header row (Row 2). Please check the template.")
        st.stop()
        
    num_questions = question_end_col - question_start_col

    # --- NEW: Parse CO Mapping and Find CO Column Indices ---
    try:
        co_mapping = df_raw.iloc[4, question_start_col:question_end_col].str.strip().tolist()
        
        co_labels = sorted(list(set(co_mapping))) # Find unique COs like ['CO1', 'CO2', ...]
        co_column_indices = {label: header_row.index(label) for label in co_labels if label in header_row}
    except Exception as e:
        st.error(f"Error parsing CO mapping from Row 5 or finding CO columns. Details: {e}")
        st.stop()

    # --- Parse Sections ---
    section_row = df_raw.iloc[0].fillna('')
    sec_info = section_row.tolist()[1:] # Start from second element
    sections = []
    i = 0
    while i < len(sec_info):
        if sec_info[i] == '':
            i += 1; continue
        try:
            num_q = int(sec_info[i])
            num_choose = int(sec_info[i+1])
            sections.append({"count": num_q, "choose": num_choose})
            i += 2
        except (ValueError, IndexError):
            break

    # --- Parse Max Marks per Question (Row 4) ---
    try:
        max_marks_all = pd.to_numeric(
            df_raw.iloc[3, question_start_col:question_end_col].tolist(), errors='raise'
        ).tolist()
    except Exception as e:
        st.error(f"Error parsing max marks in Row 4 (index 3). Details: {e}")
        st.stop()
    
    section_maxima = [max_marks_all] # Simpler for this template
    
    # --- Process Student Records ---
    last_col = n_cols - 1
    record_start_row = 6
    df_out = df_raw.copy()

    for i in range(record_start_row, n_rows):
        row = df_out.iloc[i]
        try:
            name = str(row[0]).strip()
            if not name or name.lower() == 'nan': continue
            total_marks = float(str(row[last_col]).strip())
            if pd.isna(total_marks): continue
        except (ValueError, IndexError):
            continue

        marks_for_student = []
        possible = True
        
        # This template seems to have only one section
        sec = sections[0]
        per_q_max = section_maxima[0]
        
        section_marks = partition_with_steps(total_marks, per_q_max, sec['choose'], step_size)
        
        if section_marks is None:
            st.error(f"Could not generate marks for student in row {i+1} ({name}) with total {total_marks}. Skipped.")
            continue
        
        marks_for_student.extend(section_marks)
        
        # Write question marks to DataFrame
        for k, m in enumerate(marks_for_student):
            col_idx = question_start_col + k
            df_out.iat[i, col_idx] = str(int(m)) if m == int(m) else str(m)
            
        # --- NEW: Calculate and Fill CO Totals ---
        co_totals = {label: 0.0 for label in co_column_indices.keys()}
        for k, mark in enumerate(marks_for_student):
            co_label = co_mapping[k]
            if co_label in co_totals:
                co_totals[co_label] += mark

        for label, total in co_totals.items():
            col_idx = co_column_indices[label]
            df_out.iat[i, col_idx] = str(int(total)) if total == int(total) else f"{total:.2f}"
        
        # Verify and write final sum
        final_sum = sum(marks_for_student)
        df_out.iat[i, last_col] = str(int(final_sum)) if final_sum == int(final_sum) else str(final_sum)

    st.success("Marks and CO totals have been successfully generated.")
    
    input_filename = uploaded_file.name
    base, ext = os.path.splitext(input_filename)
    output_filename = f"{base}_filled{ext}"

    num_students = n_rows - record_start_row
    if num_students > 0 and num_students <= 10:
        st.subheader("Output Preview")
        st.dataframe(df_out.iloc[record_start_row:, :])
    elif num_students > 10:
        st.info(f"Output for {num_students} students generated. Preview is hidden for large files.")

    out_buf = io.StringIO()
    df_out.to_csv(out_buf, index=False, header=False)
    st.download_button(
        label="Download Filled CSV",
        data=out_buf.getvalue(),
        file_name=output_filename,
        mime="text/csv"
    )
