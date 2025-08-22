import streamlit as st
import pandas as pd
import random
import os
import io

st.title("Assignment of Marks from a total, with CO scores computed")

st.markdown("""
- The first row specifies a number of sections as (NumQuesInSection,RequiredToAnswer) pairs.
- 3rd row of bloom's taxonomy ignored.
- 4th row (`Mark Per Que` in row 3) is per-question maxima.
- 5th row maps COs to questions. 
- For each section, only `Choose` number of questions get marks; others get zero.
- Marks are integers if possible (for integer maxima), else half marks allowed. Choose if half-marks allowed. Partial Marks allowed for a question.
- The student's total marks column is the last (5th after question columns end) column.
- Student records start at row 6. The last column (after COs) contains the total.
""")
#The script now **supports pre-filled marks**. Leave a question's cell empty to have it auto-filled.
# --- Template Download Section ---
st.subheader("Download Template if desired")
template_csv_string = """SectionMax,Choose,10,10,12,8,3,2,,,,,,,,,,,,,,,,,,,,,,,,
Question Number,,Q1A,Q1B,Q1C,Q1D,Q1E,Q1F,Q1G,Q1H,Q1I,Q1J,Q2A,Q2B,Q2C,Q2D,Q2E,Q2F,Q2G,Q2H,Q2I,Q2J,Q2K,Q2L,Q3A,Q4A,Q5A,CO1,CO2,CO3,CO4,Total
Question Level(Blooms Taxonomy),,1,2,2,4,3,2,2,3,3,1,2,2,2,2,3,3,2,2,2,3,2,2,2,3,2,,,,,
Mark Per Que,,2,2,2,2,2,2,2,2,2,2,6,6,6,6,6,6,6,6,6,6,6,6,16,16,16,,,,,
CO,,CO1,CO1,CO1,CO2,CO2,CO3,CO3,CO3,CO4,CO4,CO1,CO1,CO1,CO2,CO2,CO2,CO3,CO3,CO3,CO4,CO4,CO4,CO1,CO2,CO3,,,,,
Name of the student,Roll Number,Marks per Question,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
Ms. First,202181001,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,81
H.  Man,202160002,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,55
"""

st.download_button(
   label="Download Input Template (.csv)",
   data=template_csv_string,
   file_name="multi_section_template.csv",
   mime="text/csv",
)
st.write("_--Authored by Prof. Priyadarsan Patra_")
st.divider()

# --- Main Application ---
mark_granularity = st.radio(
    "Allow partial marks (multiples of 0.5)?", ("No", "Yes"), horizontal=True, key="granularity_choice"
)


uploaded_file = st.file_uploader("Upload your input CSV", type=["csv"], key="file_uploader")

def partition_with_steps(total, per_q_max, choose, step_size):
    steps = int(round(1 / step_size))
    n = len(per_q_max)
    int_total = int(round(total * steps))
    int_max_bounds = [int(round(m * steps)) for m in per_q_max]
    req = int(choose)
    if req < 0: return None
    if req == 0: return [0.0] * n if abs(total) < 1e-9 else None
    if req > n: return None
    if req >= n: indices = list(range(n))
    else: indices = sorted(random.sample(range(n), req))
    max_selected = [int_max_bounds[i] for i in indices]
    if int_total > sum(max_selected) or int_total < 0: return None
    parts, remaining = [], int_total
    for j in range(req):
        max_sum_rest = sum(max_selected[j+1:])
        lo = max(0, remaining - max_sum_rest)
        hi = min(max_selected[j], remaining)
        if lo > hi: return None
        val = random.randint(lo, hi) if j < req - 1 else remaining
        parts.append(val)
        remaining -= val
    if remaining != 0: return None
    final_parts = [0.0] * n
    for idx, p in zip(indices, parts): final_parts[idx] = p / steps
    return final_parts

if uploaded_file:
    step_size = 0.5 if mark_granularity == "Yes" else 1.0
    df_raw = pd.read_csv(uploaded_file, header=None, dtype=str).fillna('')
    n_rows, n_cols = df_raw.shape

    # --- Dynamic Template Parsing ---
    header_row = df_raw.iloc[1].str.strip().tolist()
    question_start_col = 2
    try:
        first_co_col_index = header_row.index('CO1')
        question_end_col = first_co_col_index
    except ValueError:
        st.error("Could not find 'CO1' in the header row (Row 2). Please check the template.")
        st.stop()
        
    co_mapping = df_raw.iloc[4, question_start_col:question_end_col].str.strip().tolist()
    co_labels = sorted(list(set(co_mapping)))
    co_column_indices = {label: header_row.index(label) for label in co_labels if label in header_row}
    
    # --- Multi-Section Parsing ---
    section_row = df_raw.iloc[0].fillna('')
    sec_info = section_row.tolist()[2:]
    sections = []
    i = 0
    while i < len(sec_info) - 1 and sec_info[i] and sec_info[i+1]:
        try:
            num_q = int(sec_info[i])
            num_choose = int(sec_info[i+1])
            sections.append({"count": num_q, "choose": num_choose})
            i += 2
        except (ValueError, IndexError): break

    max_marks_all = pd.to_numeric(df_raw.iloc[3, question_start_col:question_end_col].tolist(), errors='raise').tolist()
    
    cursor = 0
    for sec in sections:
        sec['start_idx'] = cursor
        sec['end_idx'] = cursor + sec['count']
        sec['maxima'] = max_marks_all[sec['start_idx']:sec['end_idx']]
        cursor += sec['count']

    last_col = n_cols - 1
    record_start_row = 6
    df_out = df_raw.copy()

    for i in range(record_start_row, n_rows):
        row = df_raw.iloc[i]
        try:
            name = str(row[0]).strip()
            if not name: continue
            total_marks = float(str(row[last_col]).strip())
        except (ValueError, IndexError): continue
        
        # ####################################################################
        # --- START: CORRECTED STUDENT PROCESSING LOGIC ---
        # ####################################################################

        # 1. Handle pre-filled marks across all sections at once
        final_marks = [None] * len(max_marks_all)
        sum_of_prefilled = 0.0
        for k in range(len(max_marks_all)):
            cell_val = str(row[question_start_col + k]).strip()
            if cell_val:
                try:
                    mark = float(cell_val)
                    if mark > max_marks_all[k] + 1e-6: raise ValueError("Exceeds max")
                    final_marks[k] = mark
                    sum_of_prefilled += mark
                except (ValueError, TypeError): # Ignore non-numeric, treat as empty
                    pass
        
        remaining_total = total_marks - sum_of_prefilled
        if remaining_total < -1e-6: # Allow for small float inaccuracies
             st.error(f"Error for {name}: Sum of pre-filled marks ({sum_of_prefilled}) exceeds total ({total_marks}). Skipped.")
             continue
        
        # 2. Greedy allocation across sections for the remaining marks
        possible = True
        for sec in sections:
            sec_prefilled_marks = [m for m in final_marks[sec['start_idx']:sec['end_idx']] if m is not None]
            num_sec_prefilled = len(sec_prefilled_marks)
            
            empty_slots_indices = [k for k, m in enumerate(final_marks[sec['start_idx']:sec['end_idx']]) if m is None]
            empty_slots_maxima = [sec['maxima'][k] for k in empty_slots_indices]

            num_questions_to_fill = sec['choose'] - num_sec_prefilled
            if num_questions_to_fill < 0:
                st.error(f"Error for {name}: Too many questions pre-filled for a section. Skipped.")
                possible = False; break

            max_for_section_empty = sum(sorted(empty_slots_maxima, reverse=True)[:num_questions_to_fill])
            
            marks_to_allocate = min(remaining_total, max_for_section_empty)
            marks_to_allocate = round(marks_to_allocate / step_size) * step_size
            
            distributed_marks = partition_with_steps(marks_to_allocate, empty_slots_maxima, num_questions_to_fill, step_size)
            
            if distributed_marks is None:
                possible = False; break
            
            for local_idx, mark in enumerate(distributed_marks):
                if mark > 0:
                    global_idx = sec['start_idx'] + empty_slots_indices[local_idx]
                    final_marks[global_idx] = mark
            
            remaining_total -= sum(distributed_marks)

        if not possible:
            st.error(f"Could not find a valid distribution for {name}. Skipped.")
            continue
        
        # 3. Finalize and write to DataFrame
        final_marks = [m if m is not None else 0.0 for m in final_marks]
        for k, m in enumerate(final_marks):
            df_out.iat[i, question_start_col + k] = str(int(m)) if m == int(m) else str(m)

        co_totals = {label: 0.0 for label in co_column_indices.keys()}
        for k, mark in enumerate(final_marks):
            if co_mapping[k] in co_totals: co_totals[co_mapping[k]] += mark
        for label, total in co_totals.items():
            df_out.iat[i, co_column_indices[label]] = str(int(total)) if total == int(total) else f"{total:.2f}"
        
        final_sum = sum(final_marks)
        df_out.iat[i, last_col] = str(int(final_sum)) if final_sum == int(final_sum) else f"{final_sum:.2f}"

        # ####################################################################
        # --- END: CORRECTED STUDENT PROCESSING LOGIC ---
        # ####################################################################

    st.success("Marks and CO totals have been successfully generated.")
    
    output_filename = f"{os.path.splitext(uploaded_file.name)[0]}_filled.csv"
    num_students = n_rows - record_start_row
    if num_students > 0 and num_students <= 5:
        st.subheader("Output Preview")
        st.dataframe(df_out.iloc[record_start_row:, :])
    elif num_students > 5:
        st.info(f"Output for {num_students} students generated. Preview is hidden.")

    out_buf = io.StringIO()
    df_out.to_csv(out_buf, index=False, header=False)
    st.download_button("Download Filled CSV", out_buf.getvalue(), output_filename, "text/csv")
