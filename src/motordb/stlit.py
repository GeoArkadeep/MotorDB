import streamlit as st
import streamlit.components.v1 as stc
import base64
import json
from PIL import Image
from io import BytesIO


if 'rdata' not in st.session_state:
    st.session_state.rdata = None

# Declare the custom component
jsui = stc.declare_component('my_component', path='./components/jsui')

# Use the component
received_data = jsui()

if received_data is not None:
    # Your data from calculate() will be available here
    st.session_state.rdata = received_data
    data = json.loads(st.session_state.rdata)
    st.write(data)
    lines = data['lines']
    point = data['point']
    image_base64 = data['imageBase64']
    image_data = base64.b64decode(image_base64.split(',')[1])
    image = Image.open(BytesIO(image_data))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    st.image(image)