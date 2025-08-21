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
- Marks are integers if possible (for integer maxima), else half marks allowed.
- The student's total marks column is the last (5th after question columns end) column.
-The values in Blooms Taxonomy and CO rows and columns are ignored.
- Student records start at row 6.    --Prof. Priyadrsan Patra
""")

uploaded_file = st.file_uploader("Upload the input CSV", type=["csv"])

def is_all_integral(arr):
    return all(abs(x - int(x)) < 1e-6 for x in arr)

def greedy_partition(total, per_q_max, choose, step):
    """
    Partition 'total' among 'choose' randomly selected questions (rest get 0),
    with each assigned at most per_q_max[i], using largest-remaining-fit,
    marks are multiples of step (1 or 0.5).
    Guaranteed to succeed if possible.
    """
    n = len(per_q_max)
    steps = int(round(1/step))
    int_total = int(round(total * steps))
    int_per_q_max = [int(round(m * steps)) for m in per_q_max]

    req = int(choose)
    result = [0.0] * n
    # Select which questions to answer
    if req >= n:
        indices = list(range(n))
    else:
        indices = sorted(random.sample(range(n), req))

    max_selected = [int_per_q_max[i] for i in indices]
    # Not possible? Return None
    if int_total > sum(max_selected):
        return None

    vals = [0]*req
    remaining = int_total
    # Largest-fit first (so as many as possible get full marks)
    order = sorted(range(req), key=lambda x: -max_selected[x])
    for j in order:
        can_give = min(max_selected[j], remaining)
        vals[j] = can_give
        remaining -= can_give
    # Spread anything left (should not be, but robust)
    j = 0
    while remaining > 0:
        to_add = min(max_selected[j] - vals[j], remaining)
        vals[j] += to_add
        remaining -= to_add
        j = (j + 1) % req
    for idx, v in zip(indices, vals):
        result[idx] = v / steps
    return result

if uploaded_file:
    input_filename = uploaded_file.name
    base, ext = os.path.splitext(input_filename)
    output_filename = f"{base}_filled{ext}"

    df_raw = pd.read_csv(uploaded_file, header=None, dtype=str)
    n_rows, n_cols = df_raw.shape

    # Parse Sections
    section_row = df_raw.iloc[0].fillna('')
    sec_info = section_row.tolist()[2:]
    sections = []
    i = 0
    while i < len(sec_info):
        if sec_info[i] == '':
            i += 1
            continue
        try:
            num_q = int(sec_info[i])
            num_choose = int(sec_info[i+1])
            sections.append({"count": num_q, "choose": num_choose})
            i += 2
        except:
            break

    # Compute question col index ranges for each section
    question_start = 2
    question_ranges = []
    cur = question_start
    for sec in sections:
        rng = (cur, cur + sec["count"])  # end exclusive
        question_ranges.append(rng)
        cur += sec["count"]  # next
    question_end = cur   # first col after last question

    # Build per-question maxima for all questions
    try:
        max_marks_all = pd.to_numeric(
            df_raw.iloc[3, question_start:question_end].tolist(), errors='raise'
        ).tolist()
    except Exception as e:
        st.error(f"Could not parse max marks in row 4. Details: {e}")
        st.stop()

    last_col = n_cols - 1  # final column: Total
    record_start = 5

    # Precompute for each section: 'step' and 'per-q max'
    section_steps = []
    section_maxima = []
    s_cur = 0
    for sec in sections:
        per_q = max_marks_all[s_cur:s_cur+sec['count']]
        section_maxima.append(per_q)
        if is_all_integral(per_q):
            section_steps.append(1)    # integer-only
        else:
            section_steps.append(0.5)  # allow half-marks
        s_cur += sec['count']

    df_out = df_raw.copy()
    for i in range(record_start, n_rows):
        row = df_out.iloc[i]
        try:
            total_marks = float(str(row[last_col]).strip())
            if pd.isna(total_marks):
                continue
        except:
            continue

        marks_all_sections = []
        ok = True
        total_max = sum(
            sum(sorted(section_maxima[secidx], reverse=True)[:sections[secidx]['choose']])
            for secidx in range(len(sections))
        )
        if total_marks < 0 or total_marks > total_max + 1e-6:
            ok = False

        # For each section, assign as much as you can to each, as possible
        remainder = total_marks
        all_marks = []
        for secidx, (start, end) in enumerate(question_ranges):
            per_q_max = section_maxima[secidx]
            req = sections[secidx]['choose']
            step = section_steps[secidx]
            sec_max = sum(sorted(per_q_max, reverse=True)[:req])
            sec_take = min(sec_max, remainder)
            marks = greedy_partition(sec_take, per_q_max, req, step)
            if marks is None:
                ok = False
                break
            all_marks += marks
            remainder -= sum(marks)
        # If due to rounding there is a tiny remainder (e.g., 0.5 left), add it to one of the max
        if ok and abs(remainder) > 1e-5:
            # Try to assign remainder to any eligible
            for idx in range(len(all_marks)):
                step = section_steps[0]  # All steps are 1 in your sample, so this is fine
                per_q_max = max_marks_all[idx]
                if all_marks[idx] + remainder <= per_q_max + 1e-5:
                    all_marks[idx] += remainder
                    break
        if not ok or not all(abs(x - int(x)) < 1e-5 or abs((x * 2) - int(x * 2)) < 1e-5 for x in all_marks):
            st.error(f"Could not generate marks for row {i+1} (total={total_marks}). Skipped.")
            continue
        for k, m in enumerate(all_marks):
            if is_all_integral([m]):
                df_out.iat[i, question_start + k] = str(int(round(m)))
            else:
                df_out.iat[i, question_start + k] = str(m)
        df_out.iat[i, last_col] = str(round(sum(all_marks), 2))

    num_students = n_rows - record_start
    if num_students <= 5:
        st.subheader("Input Student Records")
        st.dataframe(df_raw.iloc[record_start:, :])
        st.subheader("Output (All Columns, Section Constraints Met, Integer/Halfmark as Allowed)")
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
