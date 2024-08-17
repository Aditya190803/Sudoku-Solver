# Sudoku Solver

A Streamlit app that solves Sudoku puzzles using computer vision and machine learning.

## Installation

1. Clone the repository: `git clone https://github.com/Aditya190803/sudoku-solver.git`
2. Install requirements: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`

## Usage

1. Upload a Sudoku image file (jpg or png) to the app.
2. The app will display the predicted Sudoku board.
3. Correct any incorrect predictions by inputting the correct values.
4. Click "Solve" to solve the Sudoku puzzle.
5. The app will display the solved Sudoku board.

## Note

* The app uses a pre-trained MNIST model for digit recognition.
* The app assumes that the Sudoku image is well-lit and the digits are clearly visible.
* The app may not work well with poorly lit or distorted images.
