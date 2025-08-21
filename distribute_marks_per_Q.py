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
- For each section, only `Required` number of questions get marks; others get zero.
- Marks are integers if possible (for integer maxima), else half marks allowed. Choose if half-marks allowed. Partial Marks allowed for a question.
- The student's total marks column is the last (5th after question columns end) column.
- Student records start at row 6.    --Prof. Priyadrsan Patra
""")
#The script now **supports pre-filled marks**. Leave a question's cell empty to have it auto-filled.
# --- Template Download Section ---
st.subheader("Download Template if desired")
template_csv_string = """SectionMax,Choose,10,10,12,8,3,2,,,,,,,,,,,,,,,,,,,,,,,,
Question Number,,Q1A,Q1B,Q1C,Q1D,Q1E,Q1F,Q1G,Q1H,Q1I,Q1J,Q2A,Q2B,Q2C,Q2D,Q2E,Q2F,Q2G,Q2H,Q2I,Q2J,Q2K,Q2L,Q3A,Q4A,Q5A,CO1,CO2,CO3,CO4,Total
Question Level(Blooms Taxonomy),,1,2,2,4,3,2,2,3,3,1,2,2,2,2,3,3,2,2,2,3,2,2,2,3,2,,,,,
Mark Per Que,,2,2,2,2,2,2,2,2,2,2,6,6,6,6,6,6,6,6,6,6,6,6,16,16,16,,,,,140
CO,,CO1,CO1,CO1,CO2,CO2,CO3,CO3,CO3,CO4,CO4,CO1,CO1,CO1,CO2,CO2,CO2,CO3,CO3,CO3,CO4,CO4,CO4,CO1,CO2,CO3,,,,,
Name of the student,Roll Number,Marks per Question,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
Ms. First,202181001,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,81
H.  Man,202160002,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,55
"""
st.download_button(
   label="Download Template (.csv)",
   data=template_csv_string,
   file_name="input_file_template.csv",
   mime="text/csv",
)
st.markdown(""" Authored by Prof. Priyadarsan Patra""")
st.divider()

# User choice widget is now always visible
mark_granularity = st.radio(
    "Allow partial marks (multiples of 0.5)?",
    ("No", "Yes"),
    horizontal=True,
    key="granularity_choice" # Added a key here for good practice
)

uploaded_file = st.file_uploader("Upload the input CSV", type=["csv"], key="file_uploader")

def partition_with_steps(total, per_q_max, choose, step_size):
    steps = int(round(1 / step_size))
    n = len(per_q_max)
    int_total = int(round(total * steps))
    int_max_bounds = [int(round(m * steps)) for m in per_q_max]

    req = int(choose)
    if req < 0: return None # Cannot choose a negative number of questions
    if req == 0:
        return [0.0] * n if abs(total) < 1e-9 else None

    if req > n: # If we need to choose more questions than are available
         return None

    if req >= n: # If we must fill all available questions
        indices = list(range(n))
    else:
        indices = sorted(random.sample(range(n), req))

    max_selected = [int_max_bounds[i] for i in indices]
    if int_total > sum(max_selected) or int_total < 0:
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
    # Place distributed marks into the correct positions of the available slots
    final_parts = [0.0] * n
    for idx, p in zip(indices, parts):
        final_parts[idx] = p / steps
    return final_parts


if uploaded_file:
    step_size = 0.5 if mark_granularity == "Yes" else 1.0

    df_raw = pd.read_csv(uploaded_file, header=None, dtype=str).fillna('')
    n_rows, n_cols = df_raw.shape

    header_row = df_raw.iloc[1].str.strip().tolist()
    question_start_col = 2
    try:
        first_co_col_index = header_row.index('CO1')
        question_end_col = first_co_col_index
    except ValueError:
        st.error("Could not find 'CO1' in the header row (Row 2). Please check the template.")
        st.stop()
        
    try:
        co_mapping = df_raw.iloc[4, question_start_col:question_end_col].str.strip().tolist()
        co_labels = sorted(list(set(co_mapping)))
        co_column_indices = {label: header_row.index(label) for label in co_labels if label in header_row}
    except Exception as e:
        st.error(f"Error parsing CO mapping from Row 5 or finding CO columns. Details: {e}")
        st.stop()

    section_row = df_raw.iloc[0].fillna('')
    sec_info = section_row.tolist()[1:]
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

    try:
        max_marks_all = pd.to_numeric(
            df_raw.iloc[3, question_start_col:question_end_col].tolist(), errors='raise'
        ).tolist()
    except Exception as e:
        st.error(f"Error parsing max marks in Row 4 (index 3). Details: {e}")
        st.stop()
    
    last_col = n_cols - 1
    record_start_row = 6
    df_out = df_raw.copy()

    for i in range(record_start_row, n_rows):
        row = df_raw.iloc[i]
        try:
            name = str(row[0]).strip()
            if not name or name.lower() == 'nan': continue
            total_marks = float(str(row[last_col]).strip())
            if pd.isna(total_marks): continue
        except (ValueError, IndexError):
            continue

        # --- NEW: Logic for handling pre-filled marks ---
        prefilled_marks = {}
        sum_of_prefilled = 0.0
        indices_to_fill = []
        maxima_for_empty_qs = []
        
        valid = True
        for k in range(question_start_col, question_end_col):
            q_index_local = k - question_start_col # 0-based index for the question
            q_max = max_marks_all[q_index_local]
            cell_val = str(row[k]).strip()

            if cell_val: # Cell is not empty
                try:
                    mark = float(cell_val)
                    if mark > q_max + 1e-6:
                        st.error(f"Error for {name}: Pre-filled mark {mark} for Q{q_index_local+1} exceeds its max of {q_max}. Skipping student.")
                        valid = False; break
                    prefilled_marks[k] = mark
                    sum_of_prefilled += mark
                except ValueError:
                    # Not a number, so treat as empty
                    indices_to_fill.append(k)
                    maxima_for_empty_qs.append(q_max)
            else: # Cell is empty
                indices_to_fill.append(k)
                maxima_for_empty_qs.append(q_max)
        
        if not valid: continue

        sec = sections[0]
        total_questions_to_choose = sec['choose']
        num_prefilled = len(prefilled_marks)

        if num_prefilled > total_questions_to_choose:
            st.error(f"Error for {name}: {num_prefilled} questions were pre-filled, but section requires choosing only {total_questions_to_choose}. Skipping.")
            continue
        
        remaining_total = total_marks - sum_of_prefilled
        num_questions_to_fill = total_questions_to_choose - num_prefilled

        if remaining_total < 0:
            st.error(f"Error for {name}: Sum of pre-filled marks ({sum_of_prefilled}) is greater than total marks ({total_marks}). Skipping.")
            continue

        # Distribute the remaining marks among the empty slots
        distributed_marks = partition_with_steps(remaining_total, maxima_for_empty_qs, num_questions_to_fill, step_size)

        if distributed_marks is None:
            st.error(f"Could not find a valid distribution for {name} with the given pre-filled marks and total. Skipping.")
            continue
        
        # --- Combine pre-filled and newly generated marks ---
        final_marks = [0.0] * len(max_marks_all)
        # Place pre-filled marks
        for col_idx, mark in prefilled_marks.items():
            final_marks[col_idx - question_start_col] = mark
        # Place distributed marks
        for list_idx, col_idx in enumerate(indices_to_fill):
            mark = distributed_marks[list_idx]
            if mark > 0: # Only place marks for chosen questions
                final_marks[col_idx - question_start_col] = mark

        # Write question marks to DataFrame
        for k, m in enumerate(final_marks):
            col_idx = question_start_col + k
            df_out.iat[i, col_idx] = str(int(m)) if m == int(m) else str(m)
            
        # Calculate and Fill CO Totals
        co_totals = {label: 0.0 for label in co_column_indices.keys()}
        for k, mark in enumerate(final_marks):
            co_label = co_mapping[k]
            if co_label in co_totals:
                co_totals[co_label] += mark

        for label, total in co_totals.items():
            col_idx = co_column_indices[label]
            df_out.iat[i, col_idx] = str(int(total)) if total == int(total) else f"{total:.2f}"
        
        # Verify and write final sum
        final_sum = sum(final_marks)
        df_out.iat[i, last_col] = str(int(final_sum)) if final_sum == int(final_sum) else f"{final_sum:.2f}"

    st.success("Marks and CO totals have been successfully generated.")
    
    input_filename = uploaded_file.name
    base, ext = os.path.splitext(input_filename)
    output_filename = f"{base}_filled{ext}"

    num_students = n_rows - record_start_row
    if num_students > 0 and num_students <= 5:
        st.subheader("Output Preview")
        st.dataframe(df_out.iloc[record_start_row:, :])
    elif num_students > 5:
        st.info(f"Output for {num_students} students generated. Preview is hidden for files with more than 5 records.")

    out_buf = io.StringIO()
    df_out.to_csv(out_buf, index=False, header=False)
    st.download_button(
        label="Download Filled CSV",
        data=out_buf.getvalue(),
        file_name=output_filename,
        mime="text/csv"
    )
