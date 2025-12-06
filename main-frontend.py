import requests
import streamlit as st

backend_url = 'http://localhost:8000/predict'

st.set_page_config(page_title = 'Email Classifier', layout = 'centered')

st.markdown("""
# Email Category Classifier  
#### Enter the email text below, and the model will predict its category.
""")

email_text = st.text_area(
    label = '',
    height = 500,
    max_chars = None,
    placeholder = 'Paste or type the full email here...',
)

if st.button('Predict Category', use_container_width = True):

    if not email_text.strip():
        st.warning('Please enter the email text!')
    else:
        try:
            resp = requests.post(
                backend_url,
                json = {'text': email_text},
                timeout = 10
            )
            result = resp.json()

            if 'category' in result:
                st.success('Prediction completed!')

                st.markdown(f"""
                ## Predicted Category: **{result['category']}**
                """)

                st.caption(f"Category ID: {result['prediction_id']}")

            else:
                st.error('Error: The server did not return a category.')

        except Exception as e:
            st.error('Connection error with API.')
            st.error(str(e))