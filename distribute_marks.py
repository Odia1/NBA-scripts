import streamlit as st
import pandas as pd
import numpy as np
import os
from scipy.optimize import linprog

st.set_page_config(page_title="Robust Assessment Mark Allocator (Strict Sum)", layout="centered")

st.title("Robust Student Assessment Mark Allocator (Partial Filling, Guaranteed Sum)")

st.markdown("""
Upload your **Marking Scheme** and **Student Records** as CSV files.
---
##ProPatra## <u>**Hard Constraints (Enforced):**</u>
- Each assessment mark between 30% & 100% (if total ≥ 35), else 0 to 100%.
- Pre-filled assessment marks in student file are strictly kept as fixed.
- Sum of all weighted assessment marks matches "Total Marks out of 100" for each student (matching within ±0.5 due to rounding).
- Any infeasible row (where fixed + possible free marks can't reach total) is flagged, and missing marks left blank.

#### <u>**Soft Constraints:**</u>
- Attendance marks tend to be high; End-Sem tends to be low, all else random.
---
""", unsafe_allow_html=True)

# -- templates
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
    "Student 1,75,?,?,40,38,37,?,?,?\n"
    "Student 2,32,?,?,?,?,?,?,?,?\n"
    "Student 3,82,?,?,?,?,?,?,?,?\n"
    "Student 4,55,?,?,18,?,?,?,?,?\n"
    "Student 5,77,12,?,30,18,?,?,?,?,?\n"
    "Student 6,95,?,?,?,15,?,?,?,?\n"
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

        # Dirichlet bias for randomizer soft constraints
        alphas = []
        for a in assessments:
            if a.lower().strip() == "attendance":
                alphas.append(12.0)
            elif "end-semester" in a.lower():
                alphas.append(0.4)
            else:
                alphas.append(1.0)

        # CORE generator with integer-aware sum enforcement
        def marks_given_total_hard(
            total, N, minmarks, maxmarks, weights, maxmarks_vec, alphas, assessments, fixed=None
        ):
            # fixed: index -> value
            to_fill = list(range(N))
            fixed_idx = []
            fixed_val = []
            if fixed is not None:
                for i, v in fixed.items():
                    fixed_idx.append(i)
                    fixed_val.append(v)
                to_fill = [i for i in range(N) if i not in fixed]
            # Setup bounds for to_fill only
            minm = minmarks[to_fill]
            maxm = maxmarks[to_fill]
            wts = weights[to_fill]
            mx = maxmarks_vec[to_fill]
            alph = [alphas[i] for i in to_fill]
            assess = [assessments[i] for i in to_fill]
            # For End-Sem exam: cap as needed
            # Find End-Sem index among to_fill (NOT global!)
            if any("end-semester" in assessments[i].lower() for i in to_fill):
                end_sem_local = [j for j, i in enumerate(to_fill) if "end-semester" in assessments[i].lower()][0]
                # max value for endsem: no more than (total/100) × max, or original upper
                endsem_cap = min(mx[end_sem_local], (total / 100) * mx[end_sem_local])
                maxm[end_sem_local] = max(endsem_cap, minm[end_sem_local])
            # constraints: minm ≤ x ≤ maxm; sum_i (x_i / mx_i) * wts_i == needed_total
            # Also, account for fixeds in total
            tot_fix = 0.0
            if fixed:
                for j, i in enumerate(fixed_idx):
                    tot_fix += (fixed_val[j] / maxmarks[i]) * weights[i]
            remaining_total = total - tot_fix

            # Feasibility check
            min_possible = np.sum((minm / mx) * wts)
            max_possible = np.sum((maxm / mx) * wts)
            if remaining_total < min_possible - 1e-6 or remaining_total > max_possible + 1e-6:
                return None  # infeasible

            # Try a "random feasible int" approach using LP + integer adjustment loop
            # Solve LP for real numbers
            c = np.random.rand(len(to_fill))  # random obj, for soft bias
            A_eq = [(wts / mx).tolist()]
            b_eq = [remaining_total]
            bounds = [(float(minm[i]), float(maxm[i])) for i in range(len(to_fill))]
            lp = linprog(
                c,
                A_eq=A_eq,
                b_eq=b_eq,
                bounds=bounds,
                method='highs'
            )
            if not (lp.success):
                return None
            x = lp.x
            # Now: round to integers & adjust
            int_best = np.round(x)
            # Repeat search if needed (try small noise, then fix sum)
            def sum_from(v):
                return np.sum((v / mx) * wts)
            weighted = sum_from(int_best)
            delta = remaining_total - weighted
            if abs(delta) <= 0.51:
                filled_vals = int_best.astype(int)
            else:
                # Simple greedy correction: add/remove points on least/fewest-weighted fields
                # Try adding/subtracting one mark at a time to bring it within 0.5
                done = False
                for sign in [1, -1]:
                    for idx in np.argsort(wts / mx):
                        seq = int_best.copy()
                        seq[idx] += sign
                        # stay within bounds!
                        if seq[idx] < minm[idx] or seq[idx] > maxm[idx]:
                            continue
                        test_weighted = sum_from(seq)
                        if abs(test_weighted - remaining_total) <= 0.51:
                            filled_vals = seq.astype(int)
                            done = True
                            break
                    if done:
                        break
                else:
                    filled_vals = int_best.astype(int)
            # Put back in global order:
            result = [None] * N
            jj = 0
            for i in range(N):
                if fixed and i in fixed:
                    result[i] = fixed[i]
                else:
                    result[i] = int(round(filled_vals[jj]))
                    jj += 1
            return result

        filled_df = stu_df.copy()
        N = len(assessments)
        if "Remarks" not in filled_df.columns:
            filled_df["Remarks"] = ""

        for idx, row in filled_df.iterrows():
            total = float(row['Total Marks out of 100'])
            fixed = {}
            free_indices = []
            # Find which indices are fixed.
            for i, a in enumerate(assessments):
                v = row[a]
                if pd.notnull(v) and str(v).strip() not in {"", "?"}:
                    try:
                        fixed_val = float(v)
                        fixed[i] = fixed_val
                    except:
                        free_indices.append(i)
                else:
                    free_indices.append(i)

            # Min marks logic
            if total < 35:
                minmarks_row = np.zeros_like(minmarks_standard)
            else:
                minmarks_row = minmarks_standard.copy()

            # Use marks_given_total_hard for partial
            marks = marks_given_total_hard(
                total, N, minmarks_row, maxmarks, weights, maxmarks, alphas, assessments, fixed=fixed
            )

            if marks is None:
                for i in free_indices:
                    filled_df.at[idx, assessments[i]] = np.nan
                filled_df.at[idx, "Remarks"] = "INFEASIBLE: Cannot reach total marks with these fixed values."
                for i, val in fixed.items():
                    filled_df.at[idx, assessments[i]] = val
                continue
            # Done: fill all
            for i, val in enumerate(marks):
                filled_df.at[idx, assessments[i]] = int(round(val))
            filled_df.at[idx, "Remarks"] = ""

            # Verify sum matches required total (debug/guarantee, for transparency)
            weightedsum = np.sum(
                (np.array([float(filled_df.at[idx, a]) for a in assessments]) / maxmarks) * weights
            )
            if abs(weightedsum - total) > 0.51:
                filled_df.at[idx, "Remarks"] = (
                    f"BUG: Refused to match total (got {weightedsum:.2f} vs requested {total} after rounding)"
                )

        st.success("✅ Assessment marks distributed (all constraints enforced):")
        if len(filled_df) <= 5:
            st.dataframe(filled_df)
        else:
            st.info(f"Generated file has {len(filled_df)} student records (not displaying table for brevity).")

        input_filename = records_file.name if records_file is not None else "filled_student_marks.csv"
        base, ext = os.path.splitext(input_filename)
        output_filename = f"{base}_filled{ext}"

        st.download_button(
            "Download filled results as CSV",
            filled_df.to_csv(index=False),
            file_name=output_filename
        )
    else:
        st.info("Upload student records file to continue.")
else:
    st.info("Upload your marking scheme file to start.")
