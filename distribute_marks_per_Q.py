import streamlit as st
import pandas as pd
import numpy as np
import io
import os
import random

st.title("Automated Assignment of Marks Per Question")

st.markdown("""
**Instructions:**
- Upload your assessment CSV with *any* column/total labels, up to 5 header rows, and student records after.
- Ensure the structure: the first two columns are always Name/Roll, last 5 columns before total are ignored, last column is total marks.
- Click "Assign Random Marks" and download the result. -- Prof. Priyadarsan Patra
""")

uploaded_file = st.file_uploader("Upload the input CSV", type=["csv"])

if uploaded_file:
    input_filename = uploaded_file.name
    base, ext = os.path.splitext(input_filename)
    output_filename = f"{base}_filled{ext}"

    df_raw = pd.read_csv(uploaded_file, header=None)
    n_cols = df_raw.shape[1]
    record_start = 5                    # Student records start from row 5 (0-based)

    first_two_cols = [0, 1]
    last_col = n_cols - 1
    last_5_co_cols = list(range(last_col-5, last_col))

    question_start = 2
    question_end = last_col - 5         # exclusive

    n_questions = question_end - question_start

    # Get question labels, max marks (from header rows 0 and 2)
    question_labels = df_raw.iloc[0, question_start:question_end].tolist()
    try:
        max_marks_list = df_raw.iloc[2, question_start:question_end].astype(int).tolist()
    except:
        st.error("Error: Could not parse 'max marks' row. Please check that the third header row has integers for maximum marks.")
        st.stop()

    output_rows = []
    for i in range(record_start, len(df_raw)):
        row = df_raw.iloc[i]
        name = row[0]
        roll = row[1]
        try:
            total_marks = int(row[last_col])
        except:
            st.warning(f"Could not read total marks for {name} at row {i+1}. Skipping student.")
            continue

        # Random partition in bounds
        def random_partition_with_upper_bounds(total, max_bounds):
            n = len(max_bounds)
            marks = [0]*n
            remaining = total

            # Shuffle question order (<-- randomness in outcome)
            indices = list(range(n))
            random.shuffle(indices)
            for idx in indices:
                max_for_rest = sum(max_bounds[j] for j in indices[indices.index(idx)+1:])
                min_for_this = max(0, remaining - max_for_rest)
                max_for_this = min(max_bounds[idx], remaining)
                if max_for_this < min_for_this:
                    # Impossible
                    return None
                val = random.randint(min_for_this, max_for_this)
                marks[idx] = val
                remaining -= val
            return marks if remaining == 0 else None

        # Try generating marks
        for attempt in range(1000):
            marks = random_partition_with_upper_bounds(total_marks, max_marks_list)
            if marks is not None:
                break
        else:
            st.error(f"Could not generate marks for {name} ({roll}) - skipped.")
            continue

        output_rows.append([name, roll] + marks + [sum(marks)])

    # Column headers
    output_columns = (
        [df_raw.iloc[0,0], df_raw.iloc[0,1]] +
        question_labels +
        ["Total Marks"]
    )

    df_result = pd.DataFrame(output_rows, columns=output_columns)
    # Number of student records
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
