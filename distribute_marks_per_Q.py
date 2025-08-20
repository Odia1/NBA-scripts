
import streamlit as st
import pandas as pd
import os
import random

st.title("Automated Assignment of Marks Per Question")

st.markdown("""
**Instructions:**
- Upload your assessment CSV with *any* column/total labels, up to 5 header rows, and student records after.
- The first two columns are always Name/Roll, then some question columns (these will have numeric max marks in row 2 or 3), then five ignored columns, then the last column with total marks.
- File preview limited to 5 or fewer students. -- Prof. Priyadarsan Patra
""")

uploaded_file = st.file_uploader("Upload the input CSV", type=["csv"])

if uploaded_file:
    input_filename = uploaded_file.name
    base, ext = os.path.splitext(input_filename)
    output_filename = f"{base}_filled{ext}"

    df_raw = pd.read_csv(uploaded_file, header=None, dtype=str)
    n_cols = df_raw.shape[1]
    record_start = 5  # Data starts from 6th row (index 5)

    # Find question columns by checking for numeric "max marks" in header row 2 or 3.
    # We'll use row 2 (index 2) for max marks
    # Any column that can be *fully* and *positively* converted to int in that row is a question column
    max_marks_row = df_raw.iloc[2].fillna('').astype(str)
    question_indices = []
    max_marks_list = []
    question_labels = []
    for col in range(2, n_cols - 6):  # Be tolerant, go broader than you expect (last 6 col: 5 CO + 1 total)
        val = max_marks_row.iloc[col].strip()
        if val.isdigit():
            question_indices.append(col)
            max_marks_list.append(int(val))
            # The first row (index 0) is the label
            question_labels.append(str(df_raw.iloc[0, col]))
        else:
            break  # Expect contiguous question columns; stop at first non-question col.

    n_questions = len(question_indices)
    if n_questions == 0:
        st.error("Could not detect any question columns. Are the max marks entered as integers in header row 3?")
        st.stop()

    # Now reliably detect last col as the Total
    last_col = n_cols - 1

    # Assignment function: always possible if value in range
    def random_partition_with_upper_bounds(total, max_bounds):
        n = len(max_bounds)
        marks = []
        remaining = total
        for j in range(n):
            max_sum_rest = sum(max_bounds[j+1:])
            lo = max(0, remaining - max_sum_rest)
            hi = min(max_bounds[j], remaining)
            if lo > hi:
                return None
            if j < n-1:
                val = random.randint(lo, hi)
            else:
                val = remaining
            marks.append(val)
            remaining -= val
        if remaining != 0:
            return None
        return marks

    output_rows = []
    for i in range(record_start, len(df_raw)):
        row = df_raw.iloc[i]
        name = row[0]
        roll = row[1]
        try:
            total_marks = int(str(row[last_col]).strip())
        except:
            st.warning(f"Could not read total marks for {name} at row {i+1}. Skipping student.")
            continue

        if total_marks < 0 or total_marks > sum(max_marks_list):
            st.warning(f"Total marks {total_marks} for {name} is not possible given question bounds (max {sum(max_marks_list)}). Skipping student.")
            continue

        marks = random_partition_with_upper_bounds(total_marks, max_marks_list)
        if marks is None:
            st.error(f"Could not generate marks for {name} ({roll}) - skipped.")
            continue

        output_rows.append([name, roll] + marks + [sum(marks)])

    # Column headers
    output_columns = (
        [df_raw.iloc[0, 0], df_raw.iloc[0, 1]] +
        question_labels +
        ["Total Marks"]
    )

    df_result = pd.DataFrame(output_rows, columns=output_columns)
    num_students = len(df_raw) - record_start

    if num_students <= 5:
        st.subheader("Input Student Records")
        st.dataframe(df_raw.iloc[record_start:, :])
        st.subheader("Output (Assigned Marks)")
        st.dataframe(df_result)
    else:
        st.info(f"Input and output data not displayed (record count: {num_students}, limit for display: 5)")

    st.download_button(
        label="Download CSV",
        data=df_result.to_csv(index=False),
        file_name=output_filename,
        mime="text/csv"
    )
