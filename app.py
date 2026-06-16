import streamlit as st
st.title('DataGenome AI')
st.write('Upload datasets and explore schemas.')
files=st.file_uploader('Upload CSV files', accept_multiple_files=True)
if files:
    st.success(f'{len(files)} files uploaded')
