#define push_back pb
#include <bits/stdc++.h>
using namespace std;

int num_layers = 4;
int nl1 = 784;
int nl2 = 200;
int nl3 = 30;
int nl4 = 10;
int vn = nl1*nl2+nl2*nl3+nl3*nl4+nl2+nl3+nl4;		// total no of weights and biases
double eta = 0.01;

double sigmoid(double z) {
	return 1/(1+exp(-z));
}

void backProp(vector <vector<double> > &l, vector <vector<vector<double> > > &w, vector <vector<double> > &del, int y[], int level) {
	if (level == 0) return;
	if (level == num_layers) {
		for (int i=0; i < del[level-1].size(); i++) {
			double ajL = l[level-1][i];
			del[level-1][i] = ajL*(1 - ajL)*(ajL - y[i]);
		}
		return;
	}
	for (int i=0; i < del[level-1].size(); i++) {
		double sum = 0;
		for (int k=0; k < del[level].size(); k++) {
			sum += (w[level-1][k][i] * del[level][k]);
		}
		double ajL = l[level-1][i];
		del[level-1][i] = sum * ajL * (1 - ajL);
	}
	return;
}

void feedforward(vector <vector<double> > &l, vector <vector<vector<double> > > &w, vector <vector<double> > &b) {
	for (int i=1; i < l.size(); i++) {
		for (int j=0; j < l[i].size(); j++) {
			l[i][j] = b[i-1][j];
			for (int k = 0; k < l[i-1].size(); k++) {
				l[i][j] += w[i-1][j][k]*l[i-1][k];
			}
			l[i][j] = sigmoid(l[i][j]);
		}
	}
}

vector<double> calcGradient(vector <vector<double> > &l, vector <vector<double> > &del) {
	vector<double> grad(vn, 0);
	for (int i=0; i < vn; i++) {
		if (i < nl2*nl1) grad[i] = l[0][i%nl1] * del[1][i/nl1];
		else if (i - nl2*nl1 < nl2*nl3) grad[i] = l[1][(i - nl2*nl1)%nl2] * del[2][(i - nl2*nl1)/nl2];
		else if (i - nl2*nl1 - nl2*nl3 < nl3*nl4) grad[i] = l[2][(i - nl2*nl1 - nl3*nl2)%nl3] * del[3][(i - nl2*nl1 - nl3*nl2)/nl3];
		else {
			int j = i - nl2*nl1 - nl2*nl3 - nl3*nl4;
			if (j < nl2) grad[i] = del[1][j];
			else if (j < nl2+nl3) grad[i] = del[2][j-nl2];
			else grad[i] = del[3][j-nl2-nl3];
		}
	}
	return grad;
}

int main() {
	vector<double> l1(nl1, 0.1), l2(nl2, 0.1), l3(nl3, 0.1), l4(nl4, 0.1);		//neurons
	vector<double> del1(nl1, 0), del2(nl2, 0), del3(nl3, 0), del4(nl4, 0);		//deltas
	vector<double> b2(nl2, 0), b3(nl3, 0), b4(nl4, 0);		//biases
	vector<vector <double> > w1(nl2, l1), w2(nl3, l2), w3(nl4, l3);		//weights

	vector <vector<double> > layers;
	layers.pb(l1); layers.pb(l2); layers.pb(l3); layers.pb(l4);

	vector <vector<double> > deltas;
	deltas.pb(del1); deltas.pb(del2); deltas.pb(del3); deltas.pb(del4);

	vector <vector<vector<double> > > weights;
	weights.pb(w1); weights.pb(w2); weights.pb(w3);

	vector <vector<double> > biases;
	biases.pb(b2); biases.pb(b3); biases.pb(b4);

	double* all_wnb[vn];		// references to all weights and biases

	for (int i=0; i<vn; i++) {
		if (i < nl2*nl1) all_wnb[i] = &weights[0][i/nl1][i%nl1];
		else if (i - nl2*nl1 < nl2*nl3) all_wnb[i] = &weights[1][(i - nl2*nl1)/nl2][(i - nl2*nl1)%nl2];
		else if (i - nl2*nl1 - nl2*nl3 < nl3*nl4) all_wnb[i] = &weights[2][(i - nl2*nl1 - nl3*nl2)/nl3][(i - nl2*nl1 - nl3*nl2)%nl3];
		else {
			int j = i - nl2*nl1 - nl2*nl3 - nl3*nl4;
			if (j < nl2) all_wnb[i] = &biases[0][j];
			else if (j < nl2+nl3) all_wnb[i] = &biases[1][j-nl2];
			else all_wnb[i] = &biases[2][j-nl2-nl3];
		}
	}

	int mini_batch = 100;			// no of test cases in each mini batch
	int n_of_mb = 600;				// no of mini batches
	int y[10];				// for storing correct answer

	for (int k = 0; k < n_of_mb; k++) {
		vector<double> gradSum(vn, 0);			// list of gradients for a particular mini batch
		for (int i=0; i < mini_batch; i++) {
			for (int j=0; j<nl1; j++) cin >> layers[0][j];
			feedforward(layers, weights, biases);

			for (int j=0; j<10; j++) cin >> y[j];
			backProp(layers, weights, deltas, y, num_layers);

			vector<double> gr = calcGradient(layers, deltas);
			transform(gradSum.begin(), gradSum.end(), gr.begin(), gradSum.begin(), plus<double>());
		}
		for (int i=0; i < vn; i++) {
			*all_wnb[i] -= gradSum[i] * eta / mini_batch;
		}
	}

	return 0;
}