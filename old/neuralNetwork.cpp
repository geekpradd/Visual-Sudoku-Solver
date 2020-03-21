#define push_back pb
#include <bits/stdc++.h>
using namespace std;
typedef unsigned char uchar;

int ReverseInt (int i)
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1=i&255;
    ch2=(i>>8)&255;
    ch3=(i>>16)&255;
    ch4=(i>>24)&255;
    return((int)ch1<<24)+((int)ch2<<16)+((int)ch3<<8)+ch4;
}
vector<vector<int> > read_mnist_images(string full_path, int number_of_images, int image_size) {

    ifstream file(full_path, ios::binary);

    if(file.is_open()) {
        int magic_number = 0, n_rows = 0, n_cols = 0;

        file.read((char *)&magic_number, sizeof(magic_number));
        magic_number = ReverseInt(magic_number);

        if(magic_number != 2051) throw runtime_error("Invalid MNIST image file!");

        file.read((char *)&number_of_images, sizeof(number_of_images)); number_of_images = ReverseInt(number_of_images);
        file.read((char *)&n_rows, sizeof(n_rows)); n_rows = ReverseInt(n_rows);
        file.read((char *)&n_cols, sizeof(n_cols)); n_cols = ReverseInt(n_cols);

        image_size = n_rows * n_cols;

        vector<vector<int> > _dataset(number_of_images, vector<int>(image_size));
        for(int i = 0; i < number_of_images; i++) {
            for (int j = 0; j < image_size; j++)
                file.read((char *)&_dataset[i][j], 1);
        }
        return _dataset;
    } else {
        throw runtime_error("Cannot open file `" + full_path + "`!");
    }
}

int* read_mnist_labels(string full_path, int number_of_labels) {

    ifstream file(full_path, ios::binary);

    if(file.is_open()) {
        int magic_number = 0;
        file.read((char *)&magic_number, sizeof(magic_number));
        magic_number = ReverseInt(magic_number);

        if(magic_number != 2049) throw runtime_error("Invalid MNIST label file!");

        file.read((char *)&number_of_labels, sizeof(number_of_labels)), number_of_labels = ReverseInt(number_of_labels);

        int* _dataset = new int[number_of_labels];
        for(int i = 0; i < number_of_labels; i++) {
            file.read((char*)&_dataset[i], 1);
        }
        return _dataset;
    } else {
        throw runtime_error("Unable to open file `" + full_path + "`!");
    }
}

int num_layers = 4;
int nl1 = 784;
int nl2 = 300;
int nl3 = 100;
int nl4 = 10;
int nl[4] = {784, 300, 100, 10};
int vn = nl1*nl2+nl2*nl3+nl3*nl4+nl2+nl3+nl4;		// total no of weights and biases
double eta = 3;

double sigmoid(double z) {
	return 1/(1+exp(-z));
	//return (z < 0) ? 0 : z;
}

void backProp(vector <vector<double> > &l, vector <vector<vector<double> > > &w, vector <vector<double> > & del, int y[], int level) {
	if (level == 1) return;
	if (level == num_layers) {
		for (int i=0; i < del[level-1].size(); i++) {
			double ajL = l[level-1][i];
			del[level-1][i] = ajL*(1 - ajL)*(ajL - y[i]);
		}
	}
	else {
		for (int i=0; i < del[level-1].size(); i++) {
			double sum = 0;
			for (int k=0; k < del[level].size(); k++) {
				sum += (w[level-1][k][i] * del[level][k]);
			}
			double ajL = l[level-1][i];
			del[level-1][i] = sum * ajL * (1 - ajL);
		}
	}
	backProp(l, w, del, y, level-1);
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
	srand(time(0));
	vector<double> l1(nl1, 0), l2(nl2, 0), l3(nl3, 0), l4(nl4, 0);		//neurons
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

	for (int i=0; i<3; i++) {
		for(int j = 0; j < weights[i].size(); j++) {
			for (int k =0; k < weights[i][j].size(); k++) {
				weights[i][j][k] = (rand()*6.0/RAND_MAX - 3) * sqrt(2/nl[i]);
			}
		}
	}

	double* all_wnb[vn];		// references to all weights and biases

	vector<vector<int> > ar = read_mnist_images("train-images-idx3-ubyte", 60000, 784);
	int* labels = read_mnist_labels("train-labels-idx1-ubyte", 60000);

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
	int n_of_mb = 30;				// no of mini batches
	int y[10];				// for storing correct answer

	for (int k = 0; k < n_of_mb; k++) {
		vector<double> gradSum(vn, 0);			// list of gradients for a particular mini batch
		for (int i=0; i < mini_batch; i++) {
			for (int j=0; j<nl1; j++) layers[0][j] = ar[mini_batch*k+i][j]/255.0;//cin >> layers[0][j];
			feedforward(layers, weights, biases);

			for (int j=0; j<10; j++) y[j] = 0;//cin >> y[j];
			y[labels[mini_batch*k+i]] = 1;
			backProp(layers, weights, deltas, y, num_layers);
			vector<double> gr = calcGradient(layers, deltas);
			for (int i2=0; i2 < vn; i2++) {
				gradSum[i2] += gr[i2];
			}
			//transform(gradSum.begin(), gradSum.end(), gr.begin(), gradSum.begin(), plus<double>());
		}
		for (int i=0; i < vn; i++)
			*all_wnb[i] -= gradSum[i] * eta / mini_batch;
	}

	fstream file;
	file.open("wnb.txt", ios::trunc | ios::out | ios::in);
	if (file) {
		for (int i=0; i<vn; i++) file << *all_wnb[i] << ' ';
	}
	else cout << "Error creating file" << endl;

	for (int c = 0; c < 10; c++) {
	for (int i=0; i<nl1; i++) layers[0][i] = ar[c+3000][i];
	feedforward(layers, weights, biases);
	for (int i=0; i<10; i++) cout << layers[3][i] << ' ';

	cout << endl << labels[c+3000] << endl;
}

	return 0;
}