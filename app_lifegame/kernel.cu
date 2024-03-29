
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <curand.h>

void writeToConsole(const int n, const int m, int* cells)
{
	std::cout << std::endl <<"******";
	int l = 0;
	for (int i=0;i<n;i++)
	{
		std::cout<<std::endl;
		for (int j=0;j<m;j++)
		{
			std::cout<<cells[l++]<<" ";
		}
	}
}

void readtestfile(std::ifstream& in, int* cells)
{
	char x;
	int j=0;
	
	while(in>>x)
	{
		if(x=='o')
		{
			cells[j]=0;
		}
		else if(x=='x')
		{
			cells[j]=1;
		}
		else
		{
			cells[j]=2;
		}
		j++;
	}
	
}

void setOneCellArray(int* cells0, int pos,const int n,const int m, int* cells_next)
{
	if (cells0[pos] != 2)
	{
		int count = 0;

		//count
		if (pos >= m)
		{
			if ((pos - m - 1) % m != m-1) 
				{ count += cells0[pos - m - 1]; }
			count += cells0[pos - m];
			if ((pos - m + 1) % m != 0) 
				{ count += cells0[pos - m + 1]; }
		}
		if ((pos - 1) % m != m-1) { count += cells0[pos - 1]; }
		if ((pos + 1) % m != 0) { count += cells0[pos + 1]; }
		if (pos + m < m * n)
		{
			if ((pos + m - 1) % m != m - 1)
				{ count += cells0[pos + m - 1]; }
			count += cells0[pos + m];
			if ((pos + m + 1) % m != 0)
				{ count += cells0[pos + m + 1]; }
		}
		//end count

		if ((count == 3) || ((cells0[pos] == 1) && (count == 2)))
		{
			cells_next[pos] = 1;
		}
		else
		{
			cells_next[pos] = 0;
		}
	}
}


void runCPU(int nrOfGeneration)
{
	char infile[100] = "testfiles\\I35x28o.txt";
	std::ifstream in(infile);
	int n, m;
	in >> m;
	in >> n;
	std::cout << n << " " << m << std::endl;
	int* cellsCPU0 = (int*)malloc((n * m) * sizeof(int));
	readtestfile(in, cellsCPU0);
	in.close();
	writeToConsole(n, m, cellsCPU0);
	int* cells_next = (int*)malloc((n * m) * sizeof(int));
	while (nrOfGeneration > 0)
	{
		for (int i = 0; i < n * m; i++)
		{
			setOneCellArray(cellsCPU0, i, n, m, cells_next);
		}
		nrOfGeneration--;
		int* temp = cells_next;
		cells_next = cellsCPU0;
		cellsCPU0 = temp;
		writeToConsole(n, m, cellsCPU0);
	}
	free(cells_next);
	free(cellsCPU0);
}

__device__ int dev_n, dev_m, dev_nrGeneration;

__global__
void dev_generateTestfile(int* cells)
{
	int pos = blockIdx.x*blockDim.y* blockDim.x + threadIdx.x * blockDim.y + threadIdx.y;
	if (pos< dev_n*dev_m)
		cells[pos] += pos;
}

__device__ int countNeighbours(int* dev_cells0, int pos)
{
	int count = 0;

	if (pos >= dev_m)
	{
		if ((pos - dev_m - 1) % dev_m != dev_m - 1)
		{
			count += dev_cells0[pos - dev_m - 1];
		}
		count += dev_cells0[pos - dev_m];
		if ((pos - dev_m + 1) % dev_m != 0)
		{
			count += dev_cells0[pos - dev_m + 1];
		}
	}
	if ((pos - 1) % dev_m != dev_m - 1) { count += dev_cells0[pos - 1]; }
	if ((pos + 1) % dev_m != 0) { count += dev_cells0[pos + 1]; }
	if (pos + dev_m < dev_m * dev_n)
	{
		if ((pos + dev_m - 1) % dev_m != dev_m - 1)
		{
			count += dev_cells0[pos + dev_m - 1];
		}
		count += dev_cells0[pos + dev_m];
		if ((pos + dev_m + 1) % dev_m != 0)
		{
			count += dev_cells0[pos + dev_m + 1];
		}
	}
	return count;
}

__global__ void dev_setOneCell(int* dev_cells0, int* dev_cells_next)
{
	__shared__ int dev_gen;
	dev_gen = dev_nrGeneration;

	/*extern __shared__ char dev_array[];
	int* dev_cells_next = (int*)dev_array;*/

	int pos = blockIdx.x * blockDim.y * blockDim.x + threadIdx.x * blockDim.y + threadIdx.y;
	__syncthreads();
	while (dev_gen > 0)
	{
		if ((pos<dev_n*dev_m)&&(dev_cells0[pos] != 2))
		{
			int count = countNeighbours(dev_cells0, pos);

			if ((count == 3) || ((dev_cells0[pos] == 1) && (count == 2)))
			{
				dev_cells_next[pos] = 1;
			}
			else
			{
				dev_cells_next[pos] = 0;
			}
			/*if (blockIdx.x == 1)
			{
				dev_cells_next[pos] = pos;
			}*/
		}
		__syncthreads();
		if (pos<dev_n*dev_m)
		{
			int temp = dev_cells0[pos];
			dev_cells0[pos] = dev_cells_next[pos];
			dev_cells_next[pos] = temp;
		}
		__syncthreads();
		if((threadIdx.x==0)&&(threadIdx.y==0))
		{
			dev_gen--;
		}
		__syncthreads();
	}
	//cudaFree(dev_cells_next);
}

void runGPU(int nrOfGeneration)
{
	int n, m;
	char infile[100] = "testfiles\\J35x28o_teszt.txt";
	//char infile[100] = "testfiles\\A5x5oBlinker.txt";
	std::ifstream in(infile);
	in >> m;
	in >> n;
	cudaMemcpyToSymbol(dev_m, &m, sizeof(int));
	cudaMemcpyToSymbol(dev_n, &n, sizeof(int));
	cudaMemcpyToSymbol(dev_nrGeneration, &nrOfGeneration, sizeof(int));
	std::cout << n << " " << m << std::endl;
	int* cells00 = (int*)malloc((n * m) * sizeof(int));
	readtestfile(in, cells00);
	in.close();
	writeToConsole(n, m, cells00);

	int blocksizex = 32, blocksizey = 30;
	int blockCount = (n * m) / (blocksizex * blocksizey) + 1;
	int sh_memoSize = ((blocksizex * blocksizey) + 2 * m)* sizeof(double);

	int* dev_cells0;
	cudaMalloc((void**)&dev_cells0, (n * m) * sizeof(int));
	cudaMemcpy(dev_cells0, cells00, (n * m) * sizeof(int), cudaMemcpyHostToDevice);

	int* dev_cells_next;
	cudaMalloc((void**)&dev_cells_next, (n * m) * sizeof(int));

	dim3 blocksize(blocksizex, blocksizey);
	dev_setOneCell << <blockCount, blocksize >> > (dev_cells0, dev_cells_next);
	//dev_setOneCell << <blockCount, blocksize, sh_memoSize >> > (dev_cells0);
	cudaMemcpy(cells00, dev_cells0, (n * m) * sizeof(int), cudaMemcpyDeviceToHost);
	//cudaMemcpyFromSymbol(&nrOfGeneration, dev_nrGeneration, sizeof(int));
	//std::cout<<std::endl<<"gen: " << nrOfGeneration;
	writeToConsole(n, m, cells00);


	free(cells00);
	cudaFree(dev_cells0);
	cudaFree(dev_cells_next);
}

int main()
{
	//runCPU(1);
	std::cout<<"\nGPU\n";
	runGPU(2);

    return 0;
}

