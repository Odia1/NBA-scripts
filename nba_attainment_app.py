import pandas as pd
import numpy as np
import streamlit as st
from io import BytesIO

st.title("CO, PO, PSO Attainment Calculator for NBA")

st.sidebar.markdown("""
**Instructions:**
- Upload the **Assessment-to-CO mapping** CSV (see template for structure).
- Upload the **Student Marks** CSV (students as rows, assessments as columns).
- Upload the **CO-PO-PSO mapping** CSV (rows: COs, columns: POs/PSOs, entries: weights/levels).
    
Download the computed Excel after clicking Submit.
""")

# ---------- File Uploads ----------
mapping_file = st.sidebar.file_uploader("Assessment-to-CO Mapping CSV", type="csv")
marks_file = st.sidebar.file_uploader("Student Marks CSV", type="csv")
copopso_file = st.sidebar.file_uploader("CO-PO-PSO Mapping CSV", type="csv")

# ---------- Helper Functions ----------
def parse_mapping(df):
    # Expect columns: Assessment Method, Name, 'Weightage / Maximum-Marks', CO1, CO2, ...
    # Output: mapping dataframe with assessment, weight, max_marks, COs mapped (list)
    records = []
    co_cols = [col for col in df.columns if col.startswith("CO")]
    current = {}
    for idx, row in df.iterrows():
        name = row['Name']
        if "Weightage" in str(row['Assessment Method']):
            current = {
                "assessment": name,
                "weight": float(str(row['Weightage / Maximum-Marks']).strip()),
                "co_map": [co for co in co_cols if str(row[co]).strip() == '✓'],
            }
        elif "Max Marks" in str(row['Assessment Method']):
            current["max_marks"] = float(str(row['Weightage / Maximum-Marks']).strip())
            records.append(current.copy())
    return pd.DataFrame(records)

def compute_co_attainment(mapping_df, marks_df):
    # mapping_df: rows with assessment, weight, max_marks, co_map (list)
    # marks_df: index: student, columns: assessments
    cos = set([co for x in mapping_df['co_map'] for co in x])
    students = marks_df.index
    co_att = pd.DataFrame(index=students, columns=cos, dtype=float)
    for co in cos:
        # All assessments mapping to this CO
        mapping_sub = mapping_df[mapping_df['co_map'].apply(lambda x: co in x)]
        total_weight = mapping_sub['weight'].sum()
        for student in students:
            acc = 0.0
            for _, row in mapping_sub.iterrows():
                asmt = row.assessment
                if asmt not in marks_df.columns: continue
                val = min(marks_df.loc[student, asmt], row['max_marks'])
                perc = val / row['max_marks']
                acc += perc * row['weight']
            co_att.loc[student, co] = (acc / total_weight) * 100 if total_weight else np.nan
    return co_att

def compute_po_attainment(avg_co, copopso_df):
    pos = copopso_df.columns.tolist()
    cos = copopso_df.index.tolist()
    po_att = {}
    for po in pos:
        weights = copopso_df[po].apply(lambda x: float(x) if str(x).strip().replace('.','',1).isdigit() else 0)
        relevant_cos = [co for co in cos if weights[co] > 0]
        denom = weights[relevant_cos].sum()
        numer = sum(avg_co[co] * weights[co] for co in relevant_cos)
        po_att[po] = round(numer / denom, 2) if denom else np.nan
    return pd.DataFrame.from_dict(po_att, orient='index', columns=['Attainment (%)'])

def to_excel(dfs: dict):
    """Converts DataFrames in dict to Excel file stored in a BytesIO object"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        for sheet, df in dfs.items():
            df.to_excel(writer, sheet_name=sheet)
    output.seek(0)
    return output

# ---------- Submit Button and Main Logic ----------
if st.sidebar.button("Submit") and mapping_file and marks_file and copopso_file:

    # Read data files
    mapping_raw = pd.read_csv(mapping_file)
    marks_raw = pd.read_csv(marks_file)
    copopso_raw = pd.read_csv(copopso_file, index_col=0)
    # Ensure student IDs used as index for marks
    if marks_raw.columns[0].lower().startswith("student"):
        marks_raw = marks_raw.set_index(marks_raw.columns[0])

    # Prepare mapping table
    mapping_df = parse_mapping(mapping_raw)
    co_list = sorted({co for row in mapping_df['co_map'] for co in row})

    # Compute Studentwise CO attainment
    co_attainment = compute_co_attainment(mapping_df, marks_raw)
    co_attainment_mean = co_attainment.mean(axis=0).to_frame("Average (%)").T

    st.subheader("Sample CO Attainment (first 5 students):")
    st.write(co_attainment.head())

    # Compute PO/PSO Attainment
    po_attainment = compute_po_attainment(co_attainment.mean(axis=0), copopso_raw)
    st.subheader("PO/PSO Attainment:")
    st.write(po_attainment)

    # Prepare Excel output
    output = to_excel({
        "CO Attainment (studentwise)": co_attainment,
        "CO Attainment (average)": co_attainment_mean,
        "PO/PSO Attainment": po_attainment
    })

    st.success('Calculation complete.')
    st.download_button(
        label="Download Attainment Excel",
        data=output,
        file_name="Attainment_Output.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

else:
    st.info("Please upload all three CSV files and click Submit.")

# ---------- Sample Templates Section ----------
st.sidebar.markdown("----\n**Templates:**")
with st.sidebar.expander("Show Example Templates"):
    st.markdown("""
**Assessment-to-CO Mapping:**  
(Columns: Assessment Method, Name, Weightage / Maximum-Marks, CO1, CO2, CO3, ...)

| Assessment Method | Name        | Weightage / Maximum-Marks | CO1 | CO2 | CO3 | CO4 |
|:------------------|:------------|:------------------------:|:---:|:---:|:---:|:---:|
| Weightage         | Assignment-1 | 2.5                      | ✓   | ✓   |     |     |
| Max Marks         | Assignment-1 | 50                       |     |     |     |     |
| ...               | ...          | ...                      |     |     |     |     |

**Student Marks:**

| Student   | Assignment-1 | Quiz-1 | ... |
|:----------|:-------------|:-------|-----|
| Student 1 | 30           | 40     | ... |
| Student 2 | 35           | 42     | ... |

**CO-PO-PSO Mapping:**  
(Rows: COs, Columns: POs+PSOs, Values=weight/level)

|      | PO1 | PO2 | ... |
|:-----|:----|:----|-----|
| CO1  | 1   | 2   | ... |
| CO2  | 3   | 3   | ... |
| ...  | ... | ... | ... |
""")
