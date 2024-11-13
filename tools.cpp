#include "tools.h"

#include <iostream>
#include <iomanip>

void printm(int* matrix, int weight, int height)
{
	for (int w = 0; w < weight; ++w)
	{
		for (int h = 0; h < height; ++h)
		{
			int value = *matrix++;
			std::cout << std::setw(4) << std::left << value;
		}
		std::cout << std::endl << std::endl;
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