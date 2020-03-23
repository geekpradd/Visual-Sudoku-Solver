#define PY_SSIZE_T_CLEAN
#include<python3.6/Python.h>
#include <bits/stdc++.h>
using namespace std;

vector<int> kuchbhi(9);
vector <vector<int> > sudokur(9, kuchbhi);

void psblts(vector<vector<int> > ar, set<int> pos[9][9]) {
	for (int i=0; i<9; i++) for (int j=0; j<9; j++) {
		if (ar[i][j] == 0) {
			set<int> random;
			for (int z=1; z<10; z++) random.insert(z);
			for (int x=0; x<9; x++) {
				random.erase(ar[x][j]);
				random.erase(ar[i][x]);
				random.erase(ar[3*(i/3) + x%3][3*(j/3) + x/3]);
			}
			pos[i][j].clear();
			pos[i][j].insert(random.begin(), random.end());
		}
	}
}

bool correct(vector <vector <int> > ar) {
	for (int i=0; i<9; i++) for (int j=0; j<9; j++) {
		for (int x=0; x<9; x++) {
			if (ar[i][j] == ar[x][j] && i != x)
				return false;
			if (ar[i][j] == ar[i][x] && j != x)
				return false;
			if (ar[i][j] == ar[3*(i/3) + x%3][3*(j/3) + x/3] && i != 3*(i/3) + x%3 && j != 3*(j/3) + x/3)
				return false;
		} 
	}
	return true;
}

bool vibhav(vector<vector<int> > ar, set<int> pos[9][9]) {
	bool completed = true;
	for (int i=0; i<9 && completed; i++) for (int j=0; j<9 && completed; j++) {
		if (ar[i][j] == 0) completed = false;
	}

	if (completed && correct(ar)) {
		for (int i=0; i<9; i++) {
			for (int j=0; j<9; j++) sudokur[i][j] = ar[i][j];
		}
		return true;
	}

	psblts(ar, pos);
	for (int i=0; i<9; i++) for (int j=0; j<9; j++) {
		if (pos[i][j].size() == 0 && ar[i][j] == 0) return false;
	}

	bool f = false;

	for (int i=0; i<9; i++) for (int j=0; j<9; j++) {
		if (pos[i][j].size() == 1 && ar[i][j] == 0) {
			ar[i][j] = *pos[i][j].begin();
			f = true;
		}
	}

	if (f) return vibhav(ar, pos);


	int n = 2;
	bool br = false;

	while (n < 10 && !br) {
		for (int i=0; i<9 && !br; i++) for (int j=0; j<9 && !br; j++) {
			if (int(pos[i][j].size()) == n && ar[i][j] == 0 && !br) {
				for (set<int> :: iterator it = pos[i][j].begin(); it != pos[i][j].end() && !br; it++) {
					ar[i][j] = *it;
					if(vibhav(ar, pos)) br = true;
				}
			}
		}
		n++;
	}
	return false;
}

static PyObject * sudoku_system(PyObject *self, PyObject *args)
{
	vector <vector<int> > sudoku(9,kuchbhi);
    const char *s;

    if (!PyArg_ParseTuple(args, "s", &s))
        return NULL;

    for (int i = 0; i < 9; ++i)
    {
    	for(int j=0;j<9;j++)
    		sudoku[i][j]=int(s[i*9+j]-'0');
    }
    set<int> pos[9][9];
	vibhav(sudoku, pos);
	PyObject* final= PyList_New(Py_ssize_t(9));
	PyObject* part[9];
	for (int i = 0; i < 9; ++i)
	{
		part[i]=PyList_New(Py_ssize_t(9));
		for (int j = 0; j < 9; ++j)
		{
			PyList_SetItem(part[i],j,PyLong_FromLong(sudokur[i][j]));
		}
		PyList_SetItem(final,i,part[i]);
	}
    return final;
}

static PyMethodDef sudokuMethods[] = {
    {"solve",  sudoku_system, METH_VARARGS,
     "Execute a shell command."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

static struct PyModuleDef sudokumodule = {
    PyModuleDef_HEAD_INIT,
    "sudoku",   /* name of module */
    "NULL", /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    sudokuMethods
};

PyMODINIT_FUNC PyInit_sudoku(void)
{
    return PyModule_Create(&sudokumodule);
}

int main(int argc, char *argv[])
{
    wchar_t *program = Py_DecodeLocale(argv[0], NULL);
    if (program == NULL) {
        fprintf(stderr, "Fatal error: cannot decode argv[0]\n");
        exit(1);
    }

    /* Add a built-in module, before Py_Initialize */
    PyImport_AppendInittab("sudoku", PyInit_sudoku);

    /* Pass argv[0] to the Python interpreter */
    Py_SetProgramName(program);

    /* Initialize the Python interpreter.  Required. */
    Py_Initialize();

    /* Optionally import the module; alternatively,
       import can be deferred until the embedded script
       imports it. */
    PyImport_ImportModule("sudoku");


    PyMem_RawFree(program);
    return 0;
}