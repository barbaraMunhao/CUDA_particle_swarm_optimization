#include "Particle.h"

__device__ 
int seq_max(CUDAParticle* p, int best)
{
	int desl = blockDim.x * blockIdx.x;
	int index = desl;
	for (int i = desl + 1; i < desl + blockDim.x; i++)
	{
		if (p[i].bestFitness() > p[index].bestFitness())
			index = i;
	}
	if (p[index].bestFitness() > p[best].bestFitness())
		best = index;
	return best;
}

__device__ 
int seq_max_p(CUDAParticle* p, int best, int* max)
{
	constexpr int d = 8;
	int n = blockDim.x / d;	
	int id = threadIdx.x;	
	int particle_desl = blockDim.x * blockIdx.x;

	int desl = id + d;

	int index_max = particle_desl + id;

	if (id % d == 0)
	{
		for (int i = id + 1; i < desl; i++)
		{
			if (p[particle_desl + i].bestFitness() > p[index_max].bestFitness())
				index_max = i;
		}
		max[id / d] = index_max;
	}
	while(n >= d)
	{
		__syncthreads();
	
		if ( id < n && id % d == 0)
		{		
			index_max = id;
			for (int i = id + 1; i < desl; i++)
			{
				if (p[particle_desl+max[i]].bestFitness() > p[particle_desl + index_max].bestFitness())
					index_max = i;
			}
			max[id / d] = index_max;			
		}
		n = n / 8;
	}
	if (id == 0) {
		if (p[particle_desl + index_max].bestFitness() > p[best].bestFitness())
			max[0] += particle_desl;
		else
			max[0] = best;
	}
	return max[0];
}


// -- PSO ---------------------------------------------------//
template<int E, bool P>
__global__ void
pso_kernel(param* parameters, Knapsack* k, int* values, int* useResources, int* capResources, float* bestFitness,
	int* solutions, CUDAParticle* particles,
	int* positions, float* velocities, curandState* devStates)  //implementa o laco do pso
{
	__shared__  int bestBlockParticle;

	//__shared__ int * indices;


	// W dinamico dentro do bloco
	__shared__ float w;
	//float w = parameters->w;
	__shared__ int last;
	//C1 e C2 dinamicos
	float c1, c2;
	c1 = parameters->c1;
	c2 = parameters->c2;
	float gap;
	bool changed = false;
	//Offsets
	int offset = threadIdx.x + blockIdx.x * blockDim.x;
	int offset_pos = (k->getNumItems() * blockIdx.x * blockDim.x * 2) + (2 * k->getNumItems()) * threadIdx.x;
	int offset_vel = (k->getNumItems() * blockIdx.x * blockDim.x) + (threadIdx.x * k->getNumItems());
	//Setting Knapsack
	if (offset == 0) {
		k->setPointers(values, useResources, capResources);
	}
	__syncthreads();
	//Create  particle	
	particles[offset].setParticleAttributes(k, parameters, positions + offset_pos, positions + offset_pos + k->getNumItems(), velocities + offset_vel, devStates + offset);
	//Initialize particle position and velocity
	particles[offset].setupParticle();
	__syncthreads();

	//Settings
	if (threadIdx.x == 0) {
		bestBlockParticle = offset;
		last = curand(devStates + offset);
		w = parameters->w;
		//indices = new int[64];
	}

	__syncthreads();
	//PSO
	for (int i = 0; i < E; i++)
	{
		//Finding th best 		
		__syncthreads();
		if (threadIdx.x == 0)
			bestBlockParticle = seq_max(particles, bestBlockParticle);
		//seq_max_p(particles, bestBlockParticle, indices);
		//if (threadIdx.x == 0)
			//bestBlockParticle = indices[0];
		__syncthreads();

		//Updating velocity and position
		particles[offset].updateVelocity(particles[bestBlockParticle].pBest(), w, c1, c2);
		particles[offset].updatePosition(P);

		//Updating pBest if needed 
		if (particles[offset].fitness() > particles[offset].bestFitness())
			particles[offset].updateBestPosition();

		//C2 dinamico
		gap = ((particles[offset].fitness() - particles[offset].bestFitness()) / (float)particles[offset].bestFitness()) * 100.0f;


		//Update W 

		if (threadIdx.x == 0)
		{
			auto c = curand(devStates + offset);
			if (c > last)
			{
				w = w * 0.9;				
			}
		}

	}// end for
	__syncthreads();
	//Results
	if (threadIdx.x == 0)
	{
		auto best = particles + bestBlockParticle;
		bestFitness[blockIdx.x * 2] = best->bestFitness();
		bestFitness[blockIdx.x * 2 + 1] = (float)best->objFunction();
		int sol_offset = blockIdx.x * k->getNumItems();
		int* sol = best->pBest();
		for (int i = 0; i < k->getNumItems(); i++)
		{
			solutions[sol_offset + i] = sol[i];
		}
	}

}// end PSO Kernel 

__global__ void setupKernel(curandState * state, unsigned long seed)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	curand_init(seed + idx, idx, 0, &state[idx]);
}
