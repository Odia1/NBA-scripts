import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

# ------------------ Helper Functions ------------------

def parse_mapping(df):
    # Parse your mapping CSV (uses revised structure with weighting/max marks)
    records = []
    co_cols = [col for col in df.columns if col.startswith("CO")]
    current = {}
    for idx, row in df.iterrows():
        name = str(row['Name'])
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

def random_weighted_marks_distribution(marks_df, mapping_df, assessment_cols, total_col="Total Marks", min_frac=0.3, seed=None):
    ''' Distributes Total Marks for each student such that:
    - Each assessment gets at least min_frac * MAX_MARK
    - No assessment exceeds its max mark
    - Distribution is weighted by assessment weight
    - Sum per student equals Total Marks
    '''
    if seed is not None:
        np.random.seed(seed)
    fill_df = marks_df.copy()
    # Lookup: assessment name -> (weight, max_mark)
    info = {row['assessment']: (row['weight'], row['max_marks']) for i, row in mapping_df.iterrows()}
    for idx, row in marks_df.iterrows():
        total_marks = float(row[total_col])
        weights = np.array([info[a][0] for a in assessment_cols], dtype=float)
        max_marks = np.array([info[a][1] for a in assessment_cols], dtype=float)
        min_marks = np.floor(min_frac * max_marks)
        allowance = max_marks - min_marks
        # Proportionally distribute remainder after minima as per weights:
        marks_left = total_marks - min_marks.sum()
        if marks_left < 0:
            # Can't satisfy minimum per assessment, rescale
            used = (total_marks / min_marks.sum()) * min_marks
            filled = np.floor(used)
            rem = int(total_marks - filled.sum())
            # assign the remaining 1-by-1 randomly
            for i in np.random.choice(np.where(used > filled)[0], rem, replace=True):
                filled[i] += 1
        else:
            # Weighted splits
            if weights.sum() == 0:  # fall back uniform
                split = np.zeros_like(weights)
            else:
                split = np.random.dirichlet(weights) * marks_left
            # Cap by allowance
            alloc = np.minimum(np.floor(split), allowance)
            marks = min_marks + alloc
            # Residual marks left (from flooring or over-capping?)
            residual = int(round(total_marks - marks.sum()))
            # Spread remainder
            for _ in range(abs(residual)):
                # Among those not at cap
                inds = [i for i in range(len(assessment_cols)) if marks[i] < max_marks[i]]
                if residual > 0 and inds:
                    i = np.random.choice(inds)
                    marks[i] += 1
                elif residual < 0:
                    # Remove excess from any over-min slots
                    inds = [i for i in range(len(assessment_cols)) if marks[i] > min_marks[i]]
                    if inds:
                        i = np.random.choice(inds)
                        marks[i] -= 1
            filled = marks
        fill_df.loc[idx, assessment_cols] = filled.astype(int)
    return fill_df

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
                perc = val / row['max_marks']
                acc += perc * row['weight']
            co_att.loc[student, co] = (acc / total_weight ) * 100 if total_weight else np.nan
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
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        for sheet, df in dfs.items():
            df.to_excel(writer, sheet_name=sheet)
    output.seek(0)
    return output

# ------------------ Streamlit UI ------------------
st.title("NBA CO/PO/PSO Attainment Tool and Random Marks Distributor")

st.header("Step 1: Upload assessment mapping (CSV)")
mapfile = st.file_uploader("Assessment-to-CO mapping file", type="csv")
mapping_df = None
if mapfile:
    mapping_raw = pd.read_csv(mapfile)
    mapping_df = parse_mapping(mapping_raw)
    assessment_cols = mapping_df['assessment'].tolist()
    st.write("Assessments as found in mapping:", assessment_cols)
    st.write(mapping_df)

st.header("Step 2: Upload blank* student marks file (with only 'Total Marks' filled, CSV)")
marksfile = st.file_uploader("Student marks file (to be filled)", type="csv")
marks_blank = None
if marksfile:
    marks_blank = pd.read_csv(marksfile)
    # Set index
    if not marks_blank.index.name and marks_blank.columns[0].lower().startswith("student"):
        marks_blank = marks_blank.set_index(marks_blank.columns[0])
    st.write(marks_blank)

if st.button("Distribute Total Marks randomly to assignments (subject to constraints)", disabled=not (marks_blank is not None and mapping_df is not None)):
    filled = random_weighted_marks_distribution(marks_blank, mapping_df, assessment_cols, "Total Marks", min_frac=0.3, seed=None)
    st.success('Distributed marks:')
    st.write(filled)
    # Download link
    out_csv = filled.reset_index().to_csv(index=False)
    st.download_button(
        label='Download distributed marks as CSV',
        data=out_csv,
        file_name='marks_distributed.csv',
        mime='text/csv'
    )
    # Save for downstream steps
    st.session_state['marks_distributed'] = filled
else:
    filled = None
    if 'marks_distributed' in st.session_state:
        filled = st.session_state['marks_distributed']

st.divider()

st.header("Step 3: Upload CO-PO-PSO mapping file (CSV)")
copopsofile = st.file_uploader("CO-PO-PSO mapping", type="csv")
copopso_df = None
if copopsofile:
    copopso_df = pd.read_csv(copopsofile, index_col=0)
    st.write(copopso_df)

st.markdown("#### Step 4: Compute and Download Attainment (using above files and marks as distributed or your own)")
if st.button("Compute CO/PO/PSO attainment",
             disabled=not (mapping_df is not None and copopso_df is not None and (filled is not None or marks_blank is not None))):

    if filled is not None:
        filled_df = filled
    elif marks_blank is not None:
        filled_df = marks_blank
    else:
        st.error("No marks data available.")
        st.stop()

    # Make sure all assessment columns are int
    for a in assessment_cols:
        filled_df[a] = pd.to_numeric(filled_df[a], errors='coerce').fillna(0).astype(int)
    co_attainment = compute_co_attainment(mapping_df, filled_df)
    co_attainment_mean = co_attainment.mean(axis=0).to_frame("Average (%)").T
    po_attainment = compute_po_attainment(co_attainment.mean(axis=0), copopso_df)
    # Excel Output
    output = to_excel({
        "CO Attainment (studentwise)": co_attainment,
        "CO Attainment (average)": co_attainment_mean,
        "PO/PSO Attainment": po_attainment,
        "Filled Marks": filled_df
    })

    st.success('Calculation complete.')
    st.write("Sample CO Attainment:")
    st.write(co_attainment.head())
    st.write("PO/PSO Attainment:")
    st.write(po_attainment)
    st.download_button(
        label="Download Attainment Excel",
        data=output,
        file_name="Attainment_Output.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# Template section
with st.expander("Show Example Templates"):
    st.markdown("""
**Assessment-to-CO Mapping:**  
| Assessment Method | Name        | Weightage / Maximum-Marks | CO1 | CO2 | CO3 | CO4 |
|:------------------|:------------|:------------------------:|:---:|:---:|:---:|:---:|
| Weightage         | Assignment-1 | 2.5                      | ✓   | ✓   |     |     |
| Max Marks         | Assignment-1 | 50                       |     |     |     |     |
| ...               | ...          | ...                      |     |     |     |     |

**Student Marks (to be filled):**
| Student   | Assignment-1 | Quiz-1 | ... | Total Marks |
|:----------|:-------------|:-------|-----|-------------|
| Student 1 |              |        | ... | 75          |
| Student 2 |              |        | ... | 82          |

**CO-PO-PSO Mapping:**  
(Rows: COs, Columns: POs+PSOs, Values=weight/level)
|      | PO1 | PO2 | ... |
|:-----|:----|:----|-----|
| CO1  | 1   | 2   | ... |
| CO2  | 3   | 3   | ... |
| ...  | ... | ... | ... |
""")
