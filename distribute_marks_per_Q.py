import streamlit as st
import pandas as pd
import random
import os
import io

st.title("Automated Assignment of Marks Per Section Constraints")

st.markdown("""
**Instructions:**
- The first row must be the "SectionMax,Required,..." row, describing for each section: (#questions in section, #student must answer).
- The third header row (index=2) is per-question mark maxima, which may be floats or ints.
- For each section: if all maxima are integers, only integer marks are assigned. If any maxima are fractional, assigned marks may be multiples of 0.5.
- Columns after questions (like CO, etc.) are left untouched.
- Student records start at row 6.  --Prof. Priyadarsan Patra
""")

uploaded_file = st.file_uploader("Upload the input CSV", type=["csv"])

def is_all_integral(arr):
    """Test if all elements are integer-valued."""
    return all(abs(x - int(x)) < 1e-6 for x in arr)

def random_partition_with_step(total, per_q_max, choose, step):
    """
    Partition total among choose questions, at most per_q_max each,
    with each assigned mark being a multiple of step (0.5 or 1).
    Only nonzero marks for chosen questions.
    """
    n = len(per_q_max)
    steps = int(round(1/step))
    int_total = int(round(total * steps))
    int_max_bounds = [int(round(m * steps)) for m in per_q_max]

    req = int(choose)
    if req == 0:
        if abs(total) < 1e-5:
            return [0.0] * n
        return None

    indices = sorted(random.sample(range(n), req))
    max_selected = [int_max_bounds[i] for i in indices]
    remaining = int_total
    vals = []
    for j in range(req):
        max_sum_rest = sum(max_selected[j+1:])
        lo = max(0, remaining - max_sum_rest)
        hi = min(max_selected[j], remaining)
        if lo > hi:
            return None
        if j < req - 1:
            val = random.randint(lo, hi)
        else:
            val = remaining
        vals.append(val)
        remaining -= val
    if remaining != 0:
        return None
    result = [0.0] * n
    for idx, v in zip(indices, vals):
        result[idx] = v / steps
    return result

if uploaded_file:
    input_filename = uploaded_file.name
    base, ext = os.path.splitext(input_filename)
    output_filename = f"{base}_filled{ext}"

    df_raw = pd.read_csv(uploaded_file, header=None, dtype=str)
    n_rows, n_cols = df_raw.shape

    # --- Parse Sections information ---
    section_row = df_raw.iloc[0].fillna('')
    sec_info = section_row.tolist()[2:]  # Skip first 2 (SectionMax,Required)
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
    question_start = 2  # Questions start at col 2
    question_ranges = []
    cur = question_start
    for sec in sections:
        rng = (cur, cur + sec["count"])  # end exclusive
        question_ranges.append(rng)
        cur += sec["count"]  # move to next
    question_end = cur   # first col after last question

    # --- Build per-question maxima for all questions ---
    try:
        max_marks_all = pd.to_numeric(
            df_raw.iloc[2, question_start:question_end].tolist(),
            errors='raise'
        ).tolist()  # floats
    except Exception as e:
        st.error(f"Could not parse max marks in row 3. Details: {e}")
        st.stop()

    last_col = n_cols - 1  # final column: Total
    record_start = 5

    # Precompute for each section: which step is allowed
    section_steps = []
    section_maxima = []
    s_cur = 0
    for sec in sections:
        per_q = max_marks_all[s_cur:s_cur+sec['count']]
        if is_all_integral(per_q):
            section_steps.append(1)    # integer-only
        else:
            section_steps.append(0.5)  # allow half-marks
        section_maxima.append(per_q)
        s_cur += sec['count']

    df_out = df_raw.copy()
    for i in range(record_start, n_rows):
        row = df_out.iloc[i]
        try:
            total_marks = float(str(row[last_col]).strip())
            if pd.isna(total_marks):
                raise ValueError("NaN total")
        except Exception as e:
            continue

        # For each section, determine per-q maxima, choose, and step
        marks_all_sections = []
        ok = True
        # First, for all but last section, randomly pick the per-section total
        max_for_section = []
        for m, sec in zip(section_maxima, sections):
            max_for_section.append(sum(sorted(m, reverse=True)[:sec["choose"]]))
        total_max = sum(max_for_section)
        if total_marks < 0 or total_marks > total_max + 1e-6:
            ok = False
        # Randomly split total_marks among sections, respecting per-section maxima
        attempts = 800
        success = False
        for _ in range(attempts):
            splits = []
            sr = len(sections)
            # For even splitting using least step size so sum matches
            step_all = min(section_steps)
            all_steps = [int(round(1/st)) for st in section_steps]
            max_s = [int(round(m*st)) for m, st in zip(max_for_section, all_steps)]
            rem = int(round(total_marks * min(all_steps))) # use smallest granularity for section split
            for j in range(sr):
                max_sum_rest = sum(max_s[j+1:])
                lo = max(0, rem - max_sum_rest)
                hi = min(max_s[j], rem)
                if lo > hi:
                    ok = False
                    break
                if j < sr - 1:
                    val = random.randint(lo, hi)
                else:
                    val = rem
                splits.append(val / all_steps[j])
                rem -= val
            if not ok:
                continue
            # Now, assign to each section
            marks_all_sections = []
            for sec_idx, (start, end) in enumerate(question_ranges):
                per_q_max = section_maxima[sec_idx]
                req = sections[sec_idx]['choose']
                sec_total = splits[sec_idx]
                step = section_steps[sec_idx]
                sec_attempts = 30
                done = False
                for _ in range(sec_attempts):
                    marks = random_partition_with_step(sec_total, per_q_max, req, step)
                    if marks is not None:
                        marks_all_sections += marks
                        done = True
                        break
                if not done:
                    ok = False
                    break
            if ok and len(marks_all_sections) == question_end - question_start:
                for k, m in enumerate(marks_all_sections):
                    df_out.iat[i, question_start + k] = str(int(m) if is_all_integral([m]) else m)
                df_out.iat[i, last_col] = str(round(sum(marks_all_sections), 2))
                success = True
                break
        if not success:
            st.error(f"Could not generate marks for row {i+1} (total={total_marks}). Skipped.")
            continue

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
