import streamlit as st
import pandas as pd
import os
import random

st.title("Automated Assignment of Marks Per Question")

st.markdown("""
**Instructions:**
- Upload your assessment CSV with any column labels, 5 header rows, student data from row 6.
- First two columns: Name, Roll. Then any number of question columns. Then 5 CO columns (ignored). Last column is total marks.
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

    # Questions: columns 2 to n_cols-5 (exclusive as end in Python -- not counting the last 5 columns)
    question_start = 2
    question_end = n_cols - 5

    question_labels = df_raw.iloc[0, question_start:question_end].tolist()
    try:
        max_marks_list = pd.to_numeric(
            df_raw.iloc[2, question_start:question_end].tolist(),
            errors='raise'
        ).astype(int).tolist()
    except Exception as e:
        st.error(f"Error: Could not parse 'max marks' row. Please check that the third header row has integers for maximum marks. Details: {e}")
        st.stop()

    last_col = n_cols - 1  # Final column is Total

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

        max_possible = sum(max_marks_list)
        if total_marks < 0 or total_marks > max_possible:
            st.warning(f"Total marks {total_marks} for {name} is not possible given question bounds (max {max_possible}). Skipping student.")
            continue

        marks = random_partition_with_upper_bounds(total_marks, max_marks_list)
        if marks is None:
            st.error(f"Could not generate marks for {name} ({roll}) - skipped.")
            continue

        output_rows.append([name, roll] + marks + [sum(marks)])

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
