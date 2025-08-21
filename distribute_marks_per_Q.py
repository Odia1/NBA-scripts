import streamlit as st
import pandas as pd
import random
import os
import io

st.title("Assignment of Marks from a total")

st.markdown("""
- The first row specifies a number of sections as (NumQuesInSection,RequiredToAnswer) pairs.
- The 4th header row (`Mark Per Que` in row 3) is per-question maxima.
- For each section, only `Required` number of questions get marks; others get zero.
- Marks are integers if possible (for integer maxima), else half marks allowed. Choose if half-marks allowed. Partial Marks allowed for a question.
- The student's total marks column is the last (5th after question columns end) column.
-The values in Blooms Taxonomy and CO rows and columns are ignored.
- Student records start at row 6.    --Prof. Priyadrsan Patra
""")


# User choice widget is now always visible
mark_granularity = st.radio(
    "Allow partial marks (multiples of 0.5)?",
    ("No", "Yes"),
    horizontal=True,
    key="granularity_choice" # Added a key here for good practice
)

# Apply the key fix to the file uploader
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
    # Determine step size from user input
    step_size = 0.5 if mark_granularity == "Yes" else 1.0

    input_filename = uploaded_file.name
    base, ext = os.path.splitext(input_filename)
    output_filename = f"{base}_filled{ext}"

    df_raw = pd.read_csv(uploaded_file, header=None, dtype=str)
    n_rows, n_cols = df_raw.shape

    section_row = df_raw.iloc[0].fillna('')
    sec_info = section_row.tolist()[2:]
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

    question_start = 2
    question_ranges = []
    cur = question_start
    for sec in sections:
        rng = (cur, cur + sec["count"])
        question_ranges.append(rng)
        cur += sec["count"]
    question_end = cur

    try:
        max_marks_all = pd.to_numeric(
            df_raw.iloc[3, question_start:question_end].tolist(), errors='raise'
        ).tolist()
    except Exception as e:
        st.error(f"Error parsing max marks in Row 4 (index 3). Details: {e}")
        st.stop()

    section_maxima = []
    cursor = 0
    for sec in sections:
        per_q = max_marks_all[cursor : cursor + sec['count']]
        section_maxima.append(per_q)
        cursor += sec['count']

    last_col = n_cols - 1
    record_start = 5
    df_out = df_raw.copy()

    for i in range(record_start, n_rows):
        row = df_out.iloc[i]
        try:
            name = str(row[0]).strip()
            if not name or name.lower() == 'nan': continue
            total_marks = float(str(row[last_col]).strip())
            if pd.isna(total_marks): continue
        except (ValueError, IndexError):
            continue

        remaining_total = total_marks
        marks_for_student = []
        possible = True

        for sec_idx in range(len(sections)):
            per_q_max = section_maxima[sec_idx]
            req = sections[sec_idx]['choose']

            max_for_section = sum(sorted(per_q_max, reverse=True)[:req])
            marks_to_allocate = min(remaining_total, max_for_section)

            # Round the allocated marks to the nearest valid step
            marks_to_allocate = round(marks_to_allocate / step_size) * step_size

            section_marks = partition_with_steps(marks_to_allocate, per_q_max, req, step_size)

            if section_marks is None:
                possible = False
                break

            marks_for_student.extend(section_marks)
            remaining_total -= sum(section_marks)

        if not possible or abs(remaining_total) > 1e-6:
            st.error(f"Could not generate marks for student in row {i+1} ({name}) with total {total_marks}. Skipped.")
            continue

        for k, m in enumerate(marks_for_student):
            df_out.iat[i, question_start + k] = str(int(m)) if m == int(m) else str(m)

        final_sum = sum(marks_for_student)
        df_out.iat[i, last_col] = str(int(final_sum)) if final_sum == int(final_sum) else str(final_sum)

    st.success("Marks have been successfully generated.")

    num_students = n_rows - record_start
    if num_students > 0 and num_students <= 10:
        st.subheader("Output Preview")
        st.dataframe(df_out.iloc[record_start:, :])
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
