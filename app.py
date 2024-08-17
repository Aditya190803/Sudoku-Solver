import streamlit as st
import cv2
import imutils
from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border
from tensorflow.keras.models import load_model
import numpy as np
from Sudoku import solveSudoku
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt

# Load model
model = load_model('model/model_mnist/')

# Function to find puzzle
def find_puzzle(img):
    real = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 1)

    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh = cv2.bitwise_not(thresh)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    puzzle_cnt = None

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            puzzle_cnt = approx
            break

    if puzzle_cnt is None:
        raise Exception("Could not find Sudoku puzzle outline. Try debugging your thresholding and contour steps.")

    cv2.drawContours(real, [puzzle_cnt], -1, (0, 255, 0), 2)

    puzzle = four_point_transform(img, puzzle_cnt.reshape(4, 2))
    warped = four_point_transform(gray, puzzle_cnt.reshape(4, 2))

    return puzzle, warped

# Function to extract digit
def extract_digit(cell):
    thresh = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    thresh = clear_border(thresh)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    if len(cnts) == 0:
        return None

    c = max(cnts, key=cv2.contourArea)
    mask = np.zeros(thresh.shape, dtype="uint8")
    cv2.drawContours(mask, [c], -1, 255, -1)

    (h, w) = thresh.shape
    percentFilled = cv2.countNonZero(mask) / float(w * h)

    if percentFilled < 0.03:
        return None

    digit = cv2.bitwise_and(thresh, thresh, mask=mask)
    kernel = np.ones((1, 1), np.uint8)
    digit = cv2.dilate(digit, kernel, iterations=1)

    return digit

# Function to display numbers on board
def display_numbers_on_board(board, puzzle):
    x = puzzle.copy()
    k = 0
    for i in range(9):
        for j in range(9):
            startX, startY, endX, endY = cell_locs[k]
            testX = int((endX - startX) * 0.33)
            testY = int((endY - startY) * -0.2)
            testX += startX
            testY += endY
            cv2.putText(x, str(board[i][j]), (testX, testY), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            k += 1
    plt.figure(figsize=(10, 8))
    plt.imshow(x)
    plt.xticks([])
    plt.yticks([])
    plt.show()
    return x

# Streamlit app
st.title("Sudoku Solver")

# Upload image
img_path = st.file_uploader("Upload Sudoku image", type=["jpg", "png"])

if img_path:
    img = cv2.imdecode(np.frombuffer(img_path.read(), np.uint8), cv2.IMREAD_COLOR)
    img = imutils.resize(img, width=600)

    puzzle, warped = find_puzzle(img)
    puzzle = imutils.resize(puzzle, width=600)
    warped = imutils.resize(warped, width=600)

    step_x = warped.shape[1] // 9
    step_y = warped.shape[0] // 9

    board = np.zeros(shape=(9, 9), dtype='int')
    cell_locs = []

    for i in range(9):
        for j in range(9):
            topleftx = j * step_x
            toplefty = i * step_y
            rightendx = (j + 1) * step_x
            rightendy = (i + 1) * step_y
            cell = warped[toplefty:rightendy, topleftx:rightendx]
            digit = extract_digit(cell)
            if digit is not None:
                roi = cv2.resize(digit, tuple((28, 28)))
                roi = roi.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                pred = model.predict(roi).argmax(axis=1)[0]
                board[i, j] = pred
            cell_locs.append([topleftx, toplefty, rightendx, rightendy])

    # Display predicted board
    st.write("Predicted Board:")
    predicted_board = display_numbers_on_board(board, puzzle)
    st.pyplot(predicted_board)

    # Correct incorrect predictions
    while True:
        res = st.selectbox("Are all numbers predicted correctly?", ["Yes", "No"])
        if res == "No":
            cx, cy, ele = st.text_input("Input row no, col no, correct element of cell (e.g., 1 2 1):").split()
            try:
                board[int(cx), int(cy)] = int(ele)
            except:
                st.write("Out of range...")
            predicted_board = display_numbers_on_board(board, puzzle)
            st.pyplot(predicted_board)
        elif res == "Yes":
            break
        else:
            st.write("Wrong choice!!!")

    # Solve Sudoku
    solved = solveSudoku(board)

    # Display solved board
    st.write("Solved Board:")
    solved_board = display_numbers_on_board(solved, puzzle)
    st.pyplot(solved_board)