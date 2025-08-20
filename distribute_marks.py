import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import linprog

st.set_page_config(page_title="Randomized Student Assessment Mark Allocator", layout="centered")

st.title("Randomized Student Assessment Mark Allocator (with Constraints)")

st.markdown("""
Upload your **Marking Scheme** and **Student Records** as CSV files.

---

#### <u>**Hard Constraints (Always Enforced):**</u>
- Each assessment mark is between **30%** (minimum) and **100%** (maximum) of its max marks.
- For each student:  
    **End-Semester Exam marks (%) ≤ their Total Marks (%)** (i.e. "End-Semester Exam" marks are not better than their total marks percentage).
- The **weighted sum** of assessment marks (as per the scheme) matches their “Total Marks out of 100”.
<br>
#### <u>**Soft Constraints (Bias in Randomization):**</u>
- Attendance marks are **more likely high**.
- End-Semester Exam marks are **less likely high**, except as needed to satisfy overall constraints.

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
    "Student 2,82,?,?,?,?,?,?,?,?\n"
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
        minmarks = np.floor(0.3 * maxmarks)
    except Exception as e:
        st.error(f"Error in scheme file: {e}")
        st.stop()

    st.write("**Loaded marking scheme:**")
    st.dataframe(scheme_df)

    st.subheader("Step 2: Upload Student Records CSV")
    st.download_button("Download Student Records Template", student_record_sample, file_name="student_marks_template.csv")
    records_file = st.file_uploader("Upload Student Records", type="csv", key="students")

    if records_file:
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
        st.dataframe(stu_df)

        # --- Dirichlet soft constraint settings ---
        alphas = []
        for a in assessments:
            if a.lower().strip() == "attendance":
                alphas.append(12.0)  # heavily favor high
            elif "end-semester" in a.lower():
                alphas.append(0.4)   # heavily favor low
            else:
                alphas.append(1.0)   # neutral

        def marks_given_total(total, N, minmarks, maxmarks, weights, maxmarks_vec, alphas, assessments):
            # Find End-Semester Exam index, if present
            end_sem_idx = None
            for i, a in enumerate(assessments):
                if "end-semester" in a.lower():
                    end_sem_idx = i
                    break
            # Copy min/max for per-student adjustment
            minmarks_cur = minmarks.copy()
            maxmarks_cur = maxmarks_vec.copy()
            # Enforce end-semester hard cap based on student's total%
            if end_sem_idx is not None:
                endsem_cap = min(
                    maxmarks_vec[end_sem_idx],
                    (total / 100) * maxmarks_vec[end_sem_idx]
                )
                # End-semester cannot be above its minmark even if cap is lower
                maxmarks_cur[end_sem_idx] = max(endsem_cap, minmarks_cur[end_sem_idx])
            minfrac = minmarks_cur / maxmarks_vec
            maxfrac = maxmarks_cur / maxmarks_vec
            for _ in range(1000):
                direction = np.random.dirichlet(alphas)
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
                # End-sem hard cap, after float computations & rounding
                if end_sem_idx is not None:
                    marks_cont[end_sem_idx] = min(marks_cont[end_sem_idx], maxmarks_cur[end_sem_idx])
                marks_int = np.round(marks_cont)
                weighted = np.sum((marks_int / maxmarks_vec) * weights)
                # Final checks
                if (
                    np.all(marks_int >= minmarks_cur)
                    and np.all(marks_int <= maxmarks_cur)
                    and np.abs(weighted - total) < 0.51
                ):
                    return marks_int.astype(int)
            # Fallback: LP with these hard constraints
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
            # ... inside for-loop over students
            total = float(row['Total Marks out of 100'])

            # Randomize alphas a little for each student even with same total
            row_noise = np.random.uniform(0.8, 1.2, size=len(alphas))
            alphas_for_this_student = np.array(alphas) * row_noise
            
            marks = marks_given_total(
                total, N, minmarks, maxmarks, weights, maxmarks, alphas_for_this_student, assessments
            )
            for a, m in zip(assessments, marks):
                filled_df.at[idx, a] = int(m)
                st.success("✅ Assessment marks distributed (all constraints enforced):")
                st.dataframe(filled_df)
                st.download_button("Download filled results as CSV", filled_df.to_csv(index=False), file_name="filled_student_marks.csv")
    else:
        st.info("Upload student records file to continue.")
else:
    st.info("Upload your marking scheme file to start.")

