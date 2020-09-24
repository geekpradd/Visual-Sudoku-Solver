## Visual Sudoku Solver

This is a project that uses OpenCV in Python to recognise the sudoku grid and digits which will be classified using a neural network and then a C++ utility will solve the sudoku and output the final solved state.

The front end of the site is created in Flask and can be deployed on a webserver that has Keras/Tensorflow installed on it. It is possible to use the simpler neural network implementation based on NumPy by changing the model code but it gives poorer results and for usability the convulational version is recommended.

### Build Instructions

Simply clone the repository and run the following command in the site folder:
```
python app.py
```
Keras, TensorFlow along with SciPy need to be installed for the site to work along with Flask. If the standalone sudoku solver module is desired then simply import the sudoku model in the sudoku folder. Usage of this can be seen in the site folder as well.

The repository also contains a lot of the old standard numpy implementation along with testing code in the old folder. This can be safely neglected.

