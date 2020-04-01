## Visual Sudoku Solver

This is a project that will use OpenCV in Python to recognise the sudoku grid and digits which will be classified using a neural network and then a C++ utility will solve the sudoku and output the final solved state.

This is currently ongoing but the majority of the work is done.
The file digit_process.py is the entry file with get_matrix being the main function that takes in the location of the image and outputs the matrix.

The sudoku folder contains the final (working) python package that at present only has the Keras model without the training code. This will be changed in the future.

The site folder contains the code of the Flask site that will provide the front end implementation.
