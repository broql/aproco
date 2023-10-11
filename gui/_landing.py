import re
import json
import datetime

import streamlit as st


def run_landing_page(run_chat):
    st.title('Welcome!')

    st.markdown('Please fill in the following information to continue:')

    email = st.text_input(label='Your email address')

    position = st.text_input(label='Your position in the company')

    terms = st.checkbox('I agree to the terms and conditions')

    if st.button('Enter'):
        if check_email(email) and position and terms:
            st.session_state.email = email
            st.session_state.position = position
            st.session_state.terms = True
            # save user data to users.json file
            with open('data/users.json', 'a') as f:
                user_json = {
                    'email': st.session_state.email,
                    'position': st.session_state.position,
                    'timestamp': datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                }
                f.write(json.dumps(user_json))
                f.write('\n')

            st.experimental_rerun()
        else:
            st.error('You must fill all the fields and agree to the terms and conditions to continue.')

def check_email(email):
    regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b'
    
    return re.fullmatch(regex, email)