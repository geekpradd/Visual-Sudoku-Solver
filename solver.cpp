#include <bits/stdc++.h>
using namespace std;

vector<int> empty_vec(9);


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

bool solve(vector<vector<int> > ar, set<int> pos[9][9], vector< vector<int> > &result) {
	bool completed = true;
	for (int i=0; i<9 && completed; i++) for (int j=0; j<9 && completed; j++) {
		if (ar[i][j] == 0) completed = false;
	}

	if (completed && correct(ar)) {
		for (int i=0; i<9; i++) {
			for (int j=0; j<9; j++) result[i][j] = ar[i][j];
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

	if (f) return solve(ar, pos);


	int n = 2;
	bool br = false;

	while (n < 10 && !br) {
		for (int i=0; i<9 && !br; i++) for (int j=0; j<9 && !br; j++) {
			if (pos[i][j].size() == n && ar[i][j] == 0 && !br) {
				for (set<int> :: iterator it = pos[i][j].begin(); it != pos[i][j].end() && !br; it++) {
					ar[i][j] = *it;
					if(solve(ar, pos)) br = true;
				}
			}
		}
		n++;
	}
	return false;
}

vector< vector<int> > solve_sudoku( vector< vector<int> > inp){
    vector< vector<int> > result(9, empty_vec); 
    set<int> pos[9][9];
    solve(sudoku, pos, result);

    return result;
}
int main() {
    vector <vector<int> > sudoku(9, empty_vec);
	for (int i=0; i<9; i++) for (int j=0; j<9; j++) cin >> sudoku[i][j];
    vector< vector<int> > res = solve_sudoku(sudoku);

	for (int i=0; i<9; i++) {
		for (int j=0; j<9; j++) cout << res[i][j] << ' ';
		cout << endl;
	}
	return 0;
}