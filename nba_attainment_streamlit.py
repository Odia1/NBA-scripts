import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

# ---------------- Parsing Functions -----------------

def parse_mapping(df):
    co_cols = [c for c in df.columns if c.strip().upper().startswith("CO")]
    records = []
    for _, row in df.iterrows():
        assessment = str(row['AssessmentName']).strip()
        weight = float(row['Weightage%'])
        max_marks = float(row['Marks'])
        co_map = [co for co in co_cols if str(row[co]).strip() in ["✓", "1"]]
        records.append({
            'assessment': assessment,
            'weight': weight,
            'max_marks': max_marks,
            'co_map': co_map
        })
    return pd.DataFrame(records)

# -----------------------------------------------------

def random_weighted_marks_distribution(marks_df, mapping_df, assessment_cols, total_col="Total Marks", min_frac=0.3, seed=None):
    if seed is not None:
        np.random.seed(seed)
    fill_df = marks_df.copy()
    info = {row['assessment']: (row['weight'], row['max_marks']) for _, row in mapping_df.iterrows()}
    for idx, row in marks_df.iterrows():
        total_marks = float(row[total_col])
        weights = np.array([info[a][0] for a in assessment_cols], dtype=float)
        max_marks = np.array([info[a][1] for a in assessment_cols], dtype=float)
        min_marks = np.floor(min_frac * max_marks)
        allowance = max_marks - min_marks
        marks_left = total_marks - min_marks.sum()
        if marks_left < 0:
            used = (total_marks / min_marks.sum()) * min_marks
            filled = np.floor(used)
            rem = int(round(total_marks - filled.sum()))
            if rem > 0:
                choices = np.flatnonzero(used > filled)
                for i in np.random.choice(choices, rem, replace=True):
                    filled[i] += 1
        else:
            if weights.sum() > 0 and marks_left > 0:
                fractions = np.random.dirichlet(weights) * marks_left
            else:
                fractions = np.zeros_like(weights)
            alloc = np.minimum(np.floor(fractions), allowance)
            marks = min_marks + alloc
            residual = int(round(total_marks - marks.sum()))
            for _ in range(abs(residual)):
                if residual > 0:
                    inds = [i for i in range(len(assessment_cols)) if marks[i] < max_marks[i]]
                    if inds:
                        marks[np.random.choice(inds)] += 1
                elif residual < 0:
                    inds = [i for i in range(len(assessment_cols)) if marks[i] > min_marks[i]]
                    if inds:
                        marks[np.random.choice(inds)] -= 1
            filled = marks
        fill_df.loc[idx, assessment_cols] = filled.astype(int)
    return fill_df

# -----------------------------------------------------

def compute_co_attainment(mapping_df, marks_df):
    cos = set([co for x in mapping_df['co_map'] for co in x])
    students = marks_df.index
    co_att = pd.DataFrame(index=students, columns=cos, dtype=float)
    for co in cos:
        mapping_sub = mapping_df[mapping_df['co_map'].apply(lambda x: co in x)]
        total_weight = mapping_sub['weight'].sum()
        for student in students:
            acc = 0.0
            for _, row in mapping_sub.iterrows():
                asmt = row.assessment
                if asmt not in marks_df.columns: continue
                val = min(marks_df.loc[student, asmt], row['max_marks'])
                perc = val / row['max_marks'] if row['max_marks'] > 0 else 0
                acc += perc * row['weight']
            co_att.loc[student, co] = (acc / total_weight) * 100 if total_weight else np.nan
    return co_att

# -----------------------------------------------------

def compute_po_attainment(avg_co, copopso_df):
    pos = copopso_df.columns.tolist()
    cos = copopso_df.index.tolist()
    po_att = {}
    for po in pos:
        weights = copopso_df[po].apply(lambda x: float(x) if str(x).strip().replace('.', '', 1).isdigit() else 0)
        relevant_cos = [co for co in cos if weights[co] > 0]
        denom = weights[relevant_cos].sum()
        numer = sum(avg_co[co] * weights[co] for co in relevant_cos)
        po_att[po] = round(numer / denom, 2) if denom else np.nan
    return pd.DataFrame.from_dict(po_att, orient='index', columns=['Attainment (%)'])

# -----------------------------------------------------

def to_excel(dfs: dict):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        for sheet, df in dfs.items():
            df.to_excel(writer, sheet_name=sheet)
    output.seek(0)
    return output

# ---------------- Streamlit UI ------------------------

st.title("NBA CO/PO/PSO Attainment Tool")

# Step 1: Mapping
st.header("Step 1: Upload Assessment-to-CO Mapping")
mapfile = st.file_uploader("Assessment-to-CO mapping file", type="csv")
if mapfile:
    mapping_raw = pd.read_csv(mapfile)
    st.write(mapping_raw.head())
    mapping_df = parse_mapping(mapping_raw)
    assessment_cols = mapping_df['assessment'].tolist()
    st.write(mapping_df)
else:
    mapping_df, assessment_cols = None, None

# Step 2: Student Marks
st.header("Step 2: Upload Student Marks (with Total Marks)")
marksfile = st.file_uploader("Student marks file", type="csv")
if marksfile:
    marks_blank = pd.read_csv(marksfile)
    if not marks_blank.index.name and marks_blank.columns[0].lower().startswith("student"):
        marks_blank = marks_blank.set_index(marks_blank.columns[0])
    st.write(marks_blank)
else:
    marks_blank = None

if st.button("Distribute Marks", disabled=not (marks_blank is not None and mapping_df is not None)):
    filled = random_weighted_marks_distribution(marks_blank, mapping_df, assessment_cols)
    st.write(filled)
    st.session_state['marks_distributed'] = filled
else:
    filled = st.session_state.get('marks_distributed', None)

# Step 3: CO-PO-PSO Mapping
st.header("Step 3: Upload CO-PO-PSO mapping")
copopsofile = st.file_uploader("CO-PO-PSO mapping", type="csv")
if copopsofile:
    copopso_df = pd.read_csv(copopsofile, index_col=0)
    st.write(copopso_df.head())
else:
    copopso_df = None

# Step 4: Compute
if st.button("Compute Attainment", disabled=not (mapping_df is not None and copopso_df is not None and (filled is not None or marks_blank is not None))):
    filled_df = filled if filled is not None else marks_blank.copy()
    for a in assessment_cols:
        filled_df[a] = pd.to_numeric(filled_df[a], errors='coerce').fillna(0).astype(int)
    co_attainment = compute_co_attainment(mapping_df, filled_df)
    co_attainment_mean = co_attainment.mean(axis=0).to_frame("Average (%)").T
    po_attainment = compute_po_attainment(co_attainment.mean(axis=0), copopso_df)
    output = to_excel({
        "Studentwise CO Attainment": co_attainment,
        "Average CO Attainment": co_attainment_mean,
        "PO/PSO Attainment": po_attainment,
        "Filled Marks": filled_df
    })
    st.write(co_attainment.head())
    st.write(po_attainment)
    st.download_button(
        label="Download Attainment Excel",
        data=output,
        file_name="Attainment_Output.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
