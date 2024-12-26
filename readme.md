# Sudoku Solver

A Streamlit app that solves Sudoku puzzles using computer vision and machine learning.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Aditya190803/sudoku-solver.git
   ```

2. **Navigate to the project directory**:
   ```bash
   cd sudoku-solver
   ```

3. **Install requirements**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app**:
   ```bash
   streamlit run app.py
   ```

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
