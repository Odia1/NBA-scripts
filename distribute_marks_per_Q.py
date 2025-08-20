import streamlit as st
import pandas as pd
import os
import random
import io

st.title("Automated Assignment of Marks Per Question")

st.markdown("""
**Instructions:**
- Upload your assessment CSV with any column names (template-agnostic).
- First two columns: Name, Roll.
- Any number of question columns follow; the last 5 columns are ignored (COs/etc), with the final column being total marks.
- Student records start from row 6.
- Output preserves all input columns and rows, only question marks in each student row are updated to random multiples of 0.5 according to constraints.
- File preview limited to 5 or fewer students. -- Prof. Priyadarsan Patra
""")

uploaded_file = st.file_uploader("Upload the input CSV", type=["csv"])

def random_partition_with_half_steps(total, max_bounds):
    """
    Partition total into len(max_bounds) numbers,
    with each at most max_bounds[i], all are multiples of 0.5.
    Returns a list of floats (multiples of 0.5) or None if impossible.
    """
    steps = 2  # since 0.5 is the step
    int_total = int(round(total * steps))
    int_max_bounds = [int(round(mb * steps)) for mb in max_bounds]

    n = len(max_bounds)
    parts = []
    remaining = int_total
    for i in range(n):
        max_sum_rest = sum(int_max_bounds[i+1:])
        lo = max(0, remaining - max_sum_rest)
        hi = min(int_max_bounds[i], remaining)
        if lo > hi:
            return None
        if i < n - 1:
            val = random.randrange(lo, hi + 1)
        else:
            val = remaining
        parts.append(val)
        remaining -= val
    if remaining != 0:
        return None
    return [v / steps for v in parts]

if uploaded_file:
    input_filename = uploaded_file.name
    base, ext = os.path.splitext(input_filename)
    output_filename = f"{base}_filled{ext}"

    df_raw = pd.read_csv(uploaded_file, header=None, dtype=str)
    n_rows, n_cols = df_raw.shape
    record_start = 5  # Data starts from 6th row (index 5)

    question_start = 2
    question_end = n_cols - 5  # last 5 cols = 5 COs/others, last col (n_cols-1) = Total

    try:
        max_marks_list = pd.to_numeric(
            df_raw.iloc[2, question_start:question_end].tolist(),
            errors='raise'
        ).tolist()  # keep floats
    except Exception as e:
        st.error(f"Could not parse max marks in header row 3 (index 2). Details: {e}")
        st.stop()

    last_col = n_cols - 1  # final column: Total

    df_out = df_raw.copy()
    output_rows = 0
    for i in range(record_start, n_rows):
        row = df_out.iloc[i]
        try:
            total_marks = float(str(row[last_col]).strip())
        except:
            st.warning(f"Could not read total marks for row {i+1}. Skipping student.")
            continue

        max_possible = sum(max_marks_list)
        if total_marks < 0 or total_marks > max_possible + 1e-6:
            st.warning(f"Total marks {total_marks} at row {i+1} is impossible (max {max_possible}). Skipping student.")
            continue

        marks = random_partition_with_half_steps(total_marks, max_marks_list)
        if marks is None:
            st.error(f"Could not generate marks for row {i+1}. Skipped.")
            continue

        # Update only the question columns in df_out for this student
        for colidx, m in enumerate(marks, start=question_start):
            df_out.iat[i, colidx] = str(m)
        # Update total column to sum for precision safety
        df_out.iat[i, last_col] = str(round(sum(marks), 2))
        output_rows += 1

    num_students = n_rows - record_start

    if num_students <= 5:
        st.subheader("Input Student Records")
        st.dataframe(df_raw.iloc[record_start:, :])
        st.subheader("Output (All Columns, Assigned Marks in 0.5 steps)")
        st.dataframe(df_out.iloc[record_start:, :])
    else:
        st.info(f"Input and output data not displayed (record count: {num_students}, limit for display: 5)")

    out_buf = io.StringIO()
    df_out.to_csv(out_buf, index=False, header=False)
    st.download_button(
        label="Download CSV",
        data=out_buf.getvalue(),
        file_name=output_filename,
        mime="text/csv"
    )
