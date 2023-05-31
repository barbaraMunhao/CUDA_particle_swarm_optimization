/*
 * Definição da Mochila Multidimensional
 */
 #ifndef __Knapsack__
 #define __Knapsack__

#define HOST __host__
#define DEVICE __device__
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>

class Knapsack
{
	protected:
		int optSol;  // Valor somados dos itens
		int numItems;        // Número de itens
		int numResources;  	 // Número de recursos
		int* values;       // Valores dos itens
		int** useResources;  // O quanto cada item consome de cada recurso (matriz em forma de vetor)
		int* capResources;   // Quantidade disponível de cada recurso
		
	public:
		//Construtor
		Knapsack(FILE*);
		HOST DEVICE
		Knapsack()= default;
		//Getters
		HOST DEVICE
		int getOptSol();
		HOST DEVICE
		int getNumItems();
		HOST DEVICE
		int getNumResources();
		HOST DEVICE
		void setPointers(int*, int*, int*);
		HOST DEVICE
		const auto intemsValues()
		{
			return values;
		}
		HOST DEVICE
		const auto itemsUseResources()
		{
			return useResources;
		}
		HOST DEVICE
		const auto resourcesCapacity()
		{
			return capResources;
		}
		
		//Destrutor
		~Knapsack();
		
		friend class Application;
    friend class Particle;
		friend class CUDAParticle;
};
//Construtor
Knapsack::Knapsack(FILE* fileIn)
{
	fscanf(fileIn, "%d", &numItems);
	fscanf(fileIn, "%d", &numResources);

	//printf("Num Objs  = %d Num Comparts = %d\n", m->qtdObjs, m->numConstraints);

	fscanf(fileIn, "%d", &optSol);

	//Alocação do vetor de values
	values = new int[numItems];//(int*) malloc(numItems * sizeof(int));

	//Leitura do vetor de values
	for (int j = 0; j < numItems; j++)
		fscanf(fileIn, "%d", &(values[j]));

	//Alocação da matriz de useResources como um vetor
	useResources = new int* [numResources];//(int**) malloc(numResources * sizeof(int*));

	for (int j = 0; j < numResources; j++)
		useResources[j] = new int[numItems];//(int*) malloc(numItems * sizeof(int));

	//Leitura da matriz de useResources
	for (int j = 0; j < numResources; j++)
		for (int k = 0; k < numItems; k++)
			fscanf(fileIn, "%d", &(useResources[j][k]));


	//Alocação do vetor de restrições
	capResources = new int[numResources];//(int*) malloc(numResources * sizeof(int));

	//Leitura do vetor de restrições
	for (int j = 0; j < numResources; j++)
		fscanf(fileIn, "%d", &(capResources[j]));
}

//Getters
HOST DEVICE
inline int
Knapsack::getOptSol()
{
	return optSol;
}
HOST DEVICE
inline int
Knapsack::getNumItems()
{
	return numItems;
}
HOST DEVICE
inline int
Knapsack::getNumResources()
{
	return numResources;
}

//Destrutor
inline 
Knapsack::~Knapsack()
{
	delete[] values;

	for (int j = 0; j < numResources; j++)
		delete[] useResources[j];

	delete[] useResources;

	delete[] capResources;
}
DEVICE
inline void
Knapsack::setPointers(int* valores, int * usoRec, int* capRec)
{
	values = valores;
	capResources = capRec;
	int* x = nullptr;
	useResources = new int* [numResources];
	for (int i = 0; i < numResources; i++) 
	{
		x = usoRec + (i * (int)numItems);
		useResources[i] = x;
	}
		
}


#endif// __Knapsack