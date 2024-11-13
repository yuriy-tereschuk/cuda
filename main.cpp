#include "kernel.h"
#include "tools.h"

int main(int argc, char** argv)
{
	int w, h;

	w = h = 16;

	int N = w * h;

	int* a = new int[N];
	int* b = new int[N];
	int* c = new int[N];

	initm(a, w, h);
	initm(b, w, h);

	printm(a, w, h);

	matrix_block_mask(c, a, b, w, h);

	printm(c, w, h);

	delete[] a;
	delete[] b;
	delete[] c;

	return 0;
}