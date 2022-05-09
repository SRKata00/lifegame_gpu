
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <curand.h>

#define N 10
#define M 10

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

void setOneCellMatrix(const int cells[N][M], int cells2[N][M], int x, int y)
{
	/*int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;*/
	int startx, starty, endx, endy;
	startx = (x - 1 < 0) ? 0 : (x - 1);
	starty = (y - 1 < 0) ? 0 : (y - 1);
	endx = (x + 1 >= N) ? N : (x + 1);
	endy = (y + 1 >= M) ? M : (y + 1);
	if (cells[x][y] != 2)
	{
		int count = 0;
		for (size_t i = startx; i <= endx; i++)
		{
			for (size_t j = starty; j <= endy; j++)
			{
				if (cells[i][j] == 1)
				{
					count++;
				}
			}
		}
		if (((cells[x][y] == 1) && (count == 4)) || ((cells[x][y] != 2) && (count == 3)))
		{
			cells2[x][y] = 1;
		}
		else
		{
			cells2[x][y] = 0;
		}
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
			if ((pos - m - 1) % m >= 0) { count += cells0[pos - m - 1]; }
			count += cells0[pos - m];
			if ((pos - m + 1) % m < m) { count += cells0[pos - m + 1]; }
		}
		if ((pos - 1) % m >= 0) { count += cells0[pos - 1]; }
		if ((pos + 1) % m < m) { count += cells0[pos + 1]; }
		if (pos + m < m * n)
		{
			if ((pos + m - 1) % m >= 0) { count += cells0[pos + m - 1]; }
			count += cells0[pos + m];
			if ((pos + m + 1) % m < m) { count += cells0[pos + m + 1]; }
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

__device__ int dev_n, dev_m;

__global__
void generateTestfile(int* cells)
{
	int pos = blockIdx.x*blockDim.y* blockDim.x +threadIdx.x * blockDim.y + threadIdx.y;
	cells[pos] = pos;
}

__global__ void dev_setOneCell(int* dev_cells0)
{
	int* dev_cells_next;
	cudaMalloc((void**)&dev_cells_next, (dev_n * dev_m) * sizeof(int));
	int pos = blockIdx.x * blockDim.y * blockDim.x + threadIdx.x * blockDim.y + threadIdx.y;
	if (dev_cells0[pos] != 2)
	{
		int count = 0;

		//count
		if (pos >= dev_m)
		{
			if ((pos - dev_m - 1) % dev_m >= 0) { count += dev_cells0[pos - dev_m - 1]; }
			count += dev_cells0[pos - dev_m];
			if ((pos - dev_m + 1) % dev_m < dev_m) { count += dev_cells0[pos - dev_m + 1]; }
		}
		if ((pos - 1)%dev_m >= 0) { count += dev_cells0[pos-1]; }
		if ((pos + 1)%dev_m < dev_m) { count += dev_cells0[pos+1]; }
		if (pos + dev_m < dev_m * dev_n)
		{
			if ((pos + dev_m - 1) % dev_m >= 0) { count += dev_cells0[pos + dev_m - 1]; }
			count += dev_cells0[pos + dev_m];
			if ((pos + dev_m + 1) % dev_m < dev_m) { count += dev_cells0[pos + dev_m + 1]; }
		}
		//end count

		if ((count == 3) || ((dev_cells0[pos] == 1) && (count == 2)))
		{
			dev_cells_next[pos] = 1;
		}
		else
		{
			dev_cells_next[pos] = 0;
		}
	}
	__syncthreads();
	if (pos == 0)
	{
		int* temp = dev_cells0;
		dev_cells0 = dev_cells_next;
		dev_cells_next = temp;
	}
}


void runFromTestFile()
{
	char infile[100] = "testfiles\\E5x5g1.txt";
	int n, m;
	std::ifstream in(infile);
	in >> m;
	in >> n;
	cudaMemcpyToSymbol(dev_m, &m, sizeof(int));
	cudaMemcpyToSymbol(dev_n, &n, sizeof(int));
	std::cout << n << " " << m << std::endl;

	int* cells00 = (int*)malloc(n * m * sizeof(int));
	readtestfile(in, cells00);
	writeToConsole(n, m, cells00);
	in.close();

	int* dev_cells0;
	cudaMalloc((void**)&dev_cells0, n * m * sizeof(int));
	cudaMemcpy(dev_cells0, cells00, n * m * sizeof(int), cudaMemcpyHostToDevice);


	free(cells00);
	cudaFree(dev_cells0);
 }

void runCPU(int nrOfGeneration)
{
	char infile[100] = "testfiles\\C7x5oGlider.txt";
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

int main()
{
	runCPU(8);
	/*int n = 500, m = 40;
	//curandGenerator_t gen;

	char infile[100] = "testfiles\\A5x5oBlinker.txt";
	std::ifstream in(infile);
	in >> m;
	in >> n;
	std::cout<<n<<" "<<m<<std::endl;
	int* cells00 = (int*)malloc((n * m) * sizeof(int));
	readtestfile(in, n, m, cells00);
	in.close();
	writeToConsole(n, m, cells00);

	int blocksizex = 32, blocksizey = 30;
	//int blockCountx = n / blocksizex + 1, blockCounty = m / blocksizey + 1;
	int blockCount = (n * m) / (blocksizex * blocksizey) + 1;
	*/
	/*int* dev_cells0;
	cudaMalloc((void**)&dev_cells0, (n * m) * sizeof(int));
	cudaMemcpy(dev_cells0, cells00, (n * m) * sizeof(int), cudaMemcpyHostToDevice);

	//generateTestfile<<<dim3(blockCountx, blockCounty), dim3(blocksizex, blocksizey)>>>(dev_cells0);
	dim3 blocksize(blocksizex, blocksizey);
	dev_setOneCell<<<blockCount, blocksize>>>(dev_cells0);
	//generateTestfile<<<blockCount, blocksize>>>(dev_cells0);
	cudaMemcpy(cells00, dev_cells0, (n * m) * sizeof(int), cudaMemcpyDeviceToHost);
	writeToConsole(n,m,cells00);
	//curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	*/

	//free(cells00);
	//cudaFree(dev_cells0);
    return 0;
}

