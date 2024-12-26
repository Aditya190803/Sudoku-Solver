import time
import streamlit as st
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from recognize_sudoku import recognize
from sudoku_solver import read_from_file, all_board_non_zero, solve

# Initialize variables
old_sudoku, model = None, None
input_shape = (28, 28, 1)
num_classes = 9

# Build the model
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# Load pre-trained weights
model.load_weights("digitRecognition.h5")

# Streamlit app configuration
st.set_page_config(
    page_title="Sudoku Solver App",
    page_icon=":shark:",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("Sudoku Solver")

# Hide Streamlit footer
hide_streamlit_style = """
    <style>
    footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# File upload
file = st.file_uploader("Upload Sudoku image", type=["jpg", "png"])

if file:
    img = read_from_file(file)
    start = time.time()
    grid, solved_image = recognize(img, model, old_sudoku)
    solve_duration = time.time() - start

    if not all_board_non_zero(grid):
        st.error('Could not detect 😔 sudoku. Please try again with another resolution.')
    else:
        st.success("Solved!")
        st.markdown(
            f"<center><h3>Solved in {solve_duration:.5f} seconds</h3></center>",
            unsafe_allow_html=True
        )
        st.image(solved_image, caption='Solved Sudoku', use_container_width=True)
