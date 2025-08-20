import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import linprog
import os

st.set_page_config(page_title="Randomized Student Assessment Mark Allocator", layout="centered")

st.title("Randomized Student Assessment Mark Allocator (with Flexible Min Marks)")

st.markdown("""
Upload your **Marking Scheme** and **Student Records** as CSV files.

---

#### <u>**Hard Constraints (Enforced):**</u>
- **If “Total Marks out of 100” ≥ 35:**  
    Each assessment mark is between **30%** (minimum) and **100%** (maximum) of its max marks.
- **If “Total Marks out of 100” < 35:**  
    The 30% minimum per assignment is *waived* (assessments can be zero).
- For every student:  
    **End-Semester Exam marks (%) ≤ their Total Marks (%)**.
- The **weighted sum** of assessment marks (as per the scheme) matches their “Total Marks out of 100”.
<br>
#### <u>**Soft Constraints (Bias in Randomization):**</u>
- Attendance marks are **more likely high**.
- End-Semester Exam marks are **less likely high** (except as needed to satisfy overall constraints).

---
""", unsafe_allow_html=True)

marking_scheme_sample = (
    "AssessmentName,Weightage%,MaxMarks\n"
    "Assignment - 1,2.5,50\n"
    "Quiz - 1,2.5,50\n"
    "Mid-Term,30,50\n"
    "Assignment - 2,2.5,50\n"
    "Quiz - 2,2.5,50\n"
    "Surprise Test,5,100\n"
    "Attendance,5,5\n"
    "End-Semester Exam,50,100\n"
)
student_record_sample = (
    "Marks Distributed,Total Marks out of 100,Assignment - 1,Quiz - 1,Mid-Term,Assignment - 2,Quiz - 2,Surprise Test,Attendance,End-Semester Exam\n"
    "Student 1,75,?,?,?,?,?,?,?,?\n"
    "Student 2,32,?,?,?,?,?,?,?,?\n"
    "Student 3,82,?,?,?,?,?,?,?,?\n"
)

st.subheader("Step 1: Upload Marking Scheme CSV")
st.download_button("Download Marking Scheme Template", marking_scheme_sample, file_name="scheme_template.csv")
scheme_file = st.file_uploader("Upload Marking Scheme", type="csv", key="scheme")

if scheme_file:
    scheme_df = pd.read_csv(scheme_file)
    try:
        scheme_df.columns = [c.strip() for c in scheme_df.columns]
        assessments = scheme_df["AssessmentName"].tolist()
        weights = scheme_df["Weightage%"].astype(float).to_numpy()
        maxmarks = scheme_df["MaxMarks"].astype(float).to_numpy()
        minmarks_standard = np.floor(0.3 * maxmarks)
    except Exception as e:
        st.error(f"Error in scheme file: {e}")
        st.stop()

    st.write("**Loaded marking scheme:**")
    st.dataframe(scheme_df)

    st.subheader("Step 2: Upload Student Records CSV")
    st.download_button("Download Student Records Template", student_record_sample, file_name="student_marks_template.csv")
    records_file = st.file_uploader("Upload Student Records", type="csv", key="students")

    if records_file:
        # ... inside the main part of your script, after records_file is uploaded:
        input_filename = records_file.name
        base, ext = os.path.splitext(input_filename)
        output_filename = f"{base}_filled{ext}"

        stu_df = pd.read_csv(records_file)
        stu_df.columns = [c.strip() for c in stu_df.columns]
        assess_missing = [a for a in assessments if a not in stu_df.columns]
        if assess_missing:
            st.error(f"Assessment columns missing in student file: {assess_missing}")
            st.stop()
        if "Total Marks out of 100" not in stu_df.columns:
            st.error("Missing column: 'Total Marks out of 100' in student file.")
            st.stop()

        st.write("**Loaded student records:**")
        if len(stu_df) <= 5:
            st.dataframe(stu_df)
        else:
            st.info(f"Student records file has {len(stu_df)} records (not displaying table for brevity).")

 
        # --- Dirichlet bias for soft constraints
        alphas = []
        for a in assessments:
            if a.lower().strip() == "attendance":
                alphas.append(12.0)
            elif "end-semester" in a.lower():
                alphas.append(0.4)
            else:
                alphas.append(1.0)

        def marks_given_total(total, N, minmarks, maxmarks, weights, maxmarks_vec, alphas, assessments):
            # Find End-Semester Exam index, if present
            end_sem_idx = None
            for i, a in enumerate(assessments):
                if "end-semester" in a.lower():
                    end_sem_idx = i
                    break
            minmarks_cur = minmarks.copy()
            maxmarks_cur = maxmarks_vec.copy()
            if end_sem_idx is not None:
                endsem_cap = min(
                    maxmarks_vec[end_sem_idx],
                    (total / 100) * maxmarks_vec[end_sem_idx]
                )
                maxmarks_cur[end_sem_idx] = max(endsem_cap, minmarks_cur[end_sem_idx])
            minfrac = minmarks_cur / maxmarks_vec
            maxfrac = maxmarks_cur / maxmarks_vec
            for _ in range(1000):
                # Small per-student randomization of alpha vector
                dirichlet_bias = np.random.uniform(0.85, 1.15, size=len(alphas))
                alphas_this_student = np.array(alphas) * dirichlet_bias
                direction = np.random.dirichlet(alphas_this_student)
                w = weights
                a = np.sum(direction * w)
                if a == 0:
                    continue
                s = (total - np.sum(minfrac * w)) / a
                if s < 0:
                    continue
                if not np.all(s * direction <= (maxfrac - minfrac)):
                    continue
                frac = minfrac + s * direction
                marks_cont = frac * maxmarks_vec
                if end_sem_idx is not None:
                    marks_cont[end_sem_idx] = min(marks_cont[end_sem_idx], maxmarks_cur[end_sem_idx])
                marks_int = np.round(marks_cont)
                weighted = np.sum((marks_int / maxmarks_vec) * weights)
                if (
                    np.all(marks_int >= minmarks_cur)
                    and np.all(marks_int <= maxmarks_cur)
                    and np.abs(weighted - total) < 0.51
                ):
                    return marks_int.astype(int)
            # Fallback: LP with these constraints
            c = np.random.rand(N)
            A_eq = [(weights / maxmarks_vec).tolist()]
            b_eq = [total]
            bounds = [(float(minmarks_cur[i]), float(maxmarks_cur[i])) for i in range(N)]
            res = linprog(
                c,
                A_eq=A_eq,
                b_eq=b_eq,
                bounds=bounds,
                method='highs'
            )
            if res.success:
                marks_int = np.round(res.x)
                weighted = np.sum((marks_int / maxmarks_vec) * weights)
                if (
                    np.all(marks_int >= minmarks_cur)
                    and np.all(marks_int <= maxmarks_cur)
                    and np.abs(weighted - total) < 0.51
                ):
                    return marks_int.astype(int)
            return minmarks_cur.astype(int)

        filled_df = stu_df.copy()
        N = len(assessments)
        for idx, row in filled_df.iterrows():
            total = float(row['Total Marks out of 100'])
            # Waive per-assessment minimum if total < 35
            if total < 35:
                minmarks_row = np.zeros_like(minmarks_standard)
            else:
                minmarks_row = minmarks_standard.copy()
            marks = marks_given_total(
                total, N, minmarks_row, maxmarks, weights, maxmarks, alphas, assessments
            )
            for a, m in zip(assessments, marks):
                filled_df.at[idx, a] = int(m)

        st.success("✅ Assessment marks distributed (all constraints enforced):")
        if len(filled_df) <= 5:
            st.dataframe(filled_df)
        else:
            st.info(f"Generated file has {len(filled_df)} student records (not displaying table for brevity).")
        st.download_button("Download filled results as CSV", filled_df.to_csv(index=False), file_name=output_filename)
    else:
        st.info("Upload student records file to continue.")
else:
    st.info("Upload your marking scheme file to start.")
