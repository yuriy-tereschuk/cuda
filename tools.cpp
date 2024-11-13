#include "tools.h"

#include <iostream>

void printm(int* matrix, int weight, int height)
{
	for (int w = 0; w < weight; ++w)
	{
		for (int h = 0; h < height; ++h)
		{
			int value = *matrix++;
			std::cout << value << " ";
		}
		std::cout << std::endl;
	}
}

void initm(int* matrix, int width, int height)
{
	for (int w = 0; w < width; w++)
	{
		for (int h = 0; h < height; h++) {
			*matrix = h;
			matrix++;
		}
	}
}