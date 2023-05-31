#include "pso.cu"
//#include "Particle.h"
#include <stdio.h>
#include "Stopwatch.h"
#include "CUDAHelper.h"
//PSO MAIN//
#include <fstream>
#include <string>
#include <iostream>
#include <filesystem>
#include <chrono>
#include <random>
#include <sys/stat.h> 
#include <sys/types.h> 

namespace fs = std::experimental::filesystem;
//Declaracoes 
void teste_sort(int*, Knapsack*, int);

template<int, bool>
void execute_pso(const char*, const char*, param);
template<int, bool>
void run_all_instances(param, const char *, const char *);
template<int, bool>
void search(int execs, param_limits* limits, bool fc1 = false, bool fc2 = false, bool fw = true, bool fpenalty = false);

int main(){

	
	constexpr int epochs = 600;

	param p;
	// com penalidade - 0.87958	1.94253	1	4110.57	600
		// sem penalidade - 1.91862	0.794073	1	600
	//com penalidade e novo update posi.
	//0.601321,1.79865,1,329.594,600
	//ver esse com nova penalide --// depois ver com o de cima tbm
	//0.670175,0.935928,1,0,600
	p.c1 = 0.601321f;
	p.c2 = 1.79865f;
	/*p.c1 = 0.87958f;
	p.c2 = 1.94253f;*/
	p.w = 1.0f;
	//p.penalty = 329.594f;
//	execute_pso<epochs, true>("data/OR5x100-0.25_1.dat", "ajustePsoicao.teste", p);
	

	//run_all_instances<epochs, false>(p, "calibracao/", "cali_wp_results.data");
	//std::cout << "Done WP";
	run_all_instances<epochs, true>(p, "data/gk/", "resMO3gk.data");
	//std::cout << "Done!";
	//------
	/*int num_conf = 10;
	std::ofstream s_file;// media dos resultados

	cg::Stopwatch search_watch;

	param_limits* p = new param_limits();

	p->c1(0.5, 2.0f);
	p->c2(0.5f, 2.0f);
	p->w(0.2f, 1.0f);
	//p->penalty(100.0f, 500.0f);

	search_watch.start();
	//busca de parametros
	search<epochs, true>(num_conf, p);
	auto time = search_watch.lap();
	s_file.open("search_nova_funcao_penalidade.data", 'w');
	s_file << "c1, c1, c2, c2, w, w, penalty, penalty, epochs \n" << p->lc1[0] << ","
		<< p->lc1[1] << "," << p->lc2[0] << "," << p->lc2[1] << "," << p->lw[0] << "," << p->lw[1]
		<< "," << p->lpenalty[0] << "," << p->lpenalty[1] << "," << epochs;
	s_file << "\n num_conf " << num_conf << "tempo" << time;
	s_file << "\n w dinamico. busca de todos os parametros, exceto w. Nova funcao de penalidade.";*/

	return 0;
}
template<int D, bool P>
void 
run_all_instances(param p, const char* path, const char * file_out)
{
	cg::Stopwatch watch;
	std::ofstream results;// media dos resultados
	std::ofstream file_time;

	results.open(file_out, std::ofstream::out | std::ofstream::app);
	results << "c1, c2, w, penalty, epochs\n" << p.c1 << "," << p.c2 << "," << p.w << "," << p.penalty << "," << D << "\n";
	results << "intacia,nOpt,melhor_gap, melhor_fit, gap_medio, fit_medio, fobj_medio, dp, tempo(ms)\n";
	results.close();
	watch.start();
	for (const auto& entry : fs::directory_iterator(path))
	{
		execute_pso<D, P>(entry.path().string().c_str(), file_out, p);
	}
	auto time = watch.lap();

	file_time.open("configwp.data", 'w');
	file_time << "tempo" << time; 
	if(P)
		file_time << " \nPermite soluçoes inviaveis, usa penalidade." ;
	else
		file_time << "\n Não permite soluçoes inviaveis,  não usa penalidade.";
	file_time.close();
}

template <int epochs, bool P>
void 
search(int execs, param_limits* limits,	bool fc1 , bool fc2 , bool fw , bool fpenalty )
{
	std::string path = "calibracao/";

	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine generator(seed);
	srand(time(NULL));
	std::uniform_real_distribution<float> distC1(limits->lc1[0], limits->lc1[1]); 
	std::uniform_real_distribution<float> distC2(limits->lc2[0], limits->lc2[1]);
	std::uniform_real_distribution<float> distW(limits->lw[0], limits->lw[1]);
	std::uniform_real_distribution<float> distPenalty(limits->lpenalty[0], limits->lpenalty[1]); 
	param p;

	p.c1 = limits->lc1[0];
	p.c2 = limits->lc2[0];
	p.w = limits->lw[1];
	p.penalty = limits->lpenalty[0];
	char out_path[20];

	for (int it = 0; it < execs; it++)
	{
		if (!fc1)
			p.c1 = distC1(generator);
		if (!fc2)
			p.c2 = distC2(generator);
		if (!fw)
			p.w = distW(generator);
		if (!fpenalty)
			p.penalty = distPenalty(generator);

		sprintf_s(out_path, "%dresults.out", execs);
		std::ofstream results;// media dos resultados
		results.open(out_path, std::ofstream::out | std::ofstream::app);
		results << "c1, c2, w, penalty, epochs\n" << p.c1 << "," << p.c2 << "," << p.w << "," << p.penalty << "," << epochs <<"\n";
		results << "intacia, melhor_gap, melhor_fit, gap_medio, fit_medio, fobj_medio, dp, tempo(ms)\n";
		results.close();

		for (const auto& entry : fs::directory_iterator(path))
		{
			execute_pso<epochs, P>(entry.path().string().c_str(), out_path, p);
		}
	}	
}

template<int E, bool P>
void
execute_pso(const char* file, const char* out_path, param  h_param) 
{
	std::cout <<file<< std::endl;
	cg::Stopwatch watch;
	//Definição das dimensões do grid e dos blocos
	constexpr unsigned BLOCK_SIZE = 512;
	constexpr unsigned EXECS = 30;
	// host var.
	//char inst_name[50];
	//sprintf_s(inst_name, "%s", file);

	FILE* filein = fopen(file, "r");

	auto h_knapsack = new Knapsack(filein);// c2= 1.279862f 0.690363f
	//param h_param{ 0.5f, 0.5f,  1.f, 0.5f };
	auto dimension = h_knapsack->getNumItems();
	auto resources = h_knapsack->getNumResources();

	int* h_solutions = new int[EXECS * dimension];
	float* h_bestFitness = new float[EXECS * 2];

	printf_s("%d\n", h_knapsack->getOptSol());
	h_param.print();

	//kernel1 param.
	curandState* devStates;
	//kernel2 param.
	param* d_param;
	//Mochila------------
	Knapsack* d_knapsack;
	int* d_valores;
	int* d_useResources;
	int* d_capResources;
	//--------------------
	float* d_bestFitness;
	int* d_solutions;
	CUDAParticle* d_particles;
	//CUDAParticle** d_aux;
	int* d_positions;
	float* d_velocities;

	// dev allocation
	cg::cuda::allocate<param>(d_param, 1);
	cg::cuda::allocate<Knapsack>(d_knapsack, 1);//knapsack allocation
	cg::cuda::allocate<int>(d_valores, dimension);
	cg::cuda::allocate<int>(d_useResources, dimension * resources);
	cg::cuda::allocate<int>(d_capResources, resources); //end knapsack allocation
	cg::cuda::allocate<int>(d_solutions, EXECS * dimension);
	cg::cuda::allocate<float>(d_bestFitness, EXECS * 2);// adicao do fitness int
	cg::cuda::allocate<CUDAParticle>(d_particles, BLOCK_SIZE * EXECS);
	cg::cuda::allocate<int>(d_positions, EXECS * BLOCK_SIZE * dimension * 2); // numero de particulas* qtd de itens do problema
	cg::cuda::allocate<float>(d_velocities, EXECS * BLOCK_SIZE * dimension);

	cudaMalloc(&devStates, EXECS * BLOCK_SIZE * sizeof(curandState));
	//cudaThreadSynchronize();//(?)

	// copy input data from host to device
	cg::cuda::copyToDevice<param>(d_param, &h_param, 1);
	cg::cuda::copyToDevice<Knapsack>(d_knapsack, h_knapsack, 1);
	cg::cuda::copyToDevice<int>(d_valores, h_knapsack->intemsValues(), dimension);
	cg::cuda::copyToDevice<int>(d_capResources, h_knapsack->resourcesCapacity(), resources);
	const auto p = h_knapsack->itemsUseResources();
	for (int i = 0; i < resources; i++)
	{
		cg::cuda::copyToDevice<int>(d_useResources + (i * dimension), p[i], dimension);
	}


	// launch  kernels
	setupKernel << <EXECS, BLOCK_SIZE >> > (devStates, time(NULL));
	//checkLastCudaError("**setup failed");	
	cg::cuda::synchronize();

	//auto time_allocation_copy = watch.lap();
	//Start 
	watch.start();
	pso_kernel<E, P> << <EXECS, BLOCK_SIZE >> > (d_param, d_knapsack, d_valores, d_useResources, d_capResources, d_bestFitness, d_solutions, d_particles, d_positions, d_velocities, devStates);
	//	checkLastCudaError("**pso failed :(");
	cg::cuda::synchronize();

	auto kernel_exe_time = watch.lap();

	// copy output data from device to host
	size_t size{ EXECS * dimension };
	cg::cuda::copyToHost<int>(h_solutions, d_solutions, size);
	cg::cuda::copyToHost<float>(h_bestFitness, d_bestFitness, EXECS * 2);


	//RESULTADOS-----------------------------------------------------------------------------------
	std::ofstream results;// media dos resultados
	results.open(out_path, std::ofstream::out | std::ofstream::app);

	//std::ofstream blocks_results; // resultados de todos os blocos
	//blocks_results.open("wffitness.out");

	float fobj_m, gap_m, gap, fit_m, best;
	// check the result
	fobj_m = gap_m = fit_m = best = 0;
	//blocks_results <<"fitness, f.obj, gap\n";
	float gaps[30];
	float dpGaps = 0.0f;
	int j = 0;
	int optCounter{0};
	for (auto i = 0; i < EXECS * 2; i = i + 2)
	{
		gap = 0;
		printf_s("%f  ", h_bestFitness[i]);
		printf_s("%f\n", h_bestFitness[i + 1]);
		gap = ((h_knapsack->getOptSol() - h_bestFitness[i + 1]) / (float)h_knapsack->getOptSol()) * 100.0f;
		if (gap == 0.0f)
			optCounter++;
		//blocks_results << h_bestFitness[i] << ", " << h_bestFitness[i + 1] << ", " << gap <<"\n";
		gaps[j] = gap;
		j++;
		gap_m += gap;
		fit_m += h_bestFitness[i];
		fobj_m += h_bestFitness[i + 1];
		if (h_bestFitness[i + 1] > best)
			best = h_bestFitness[i + 1];
	}
	gap_m = gap_m / 30;
	fit_m = fit_m / 30;
	fobj_m = fobj_m / 30;
	//Desv.Padrao------------------------
	for (int i = 0; i < EXECS; i++)
		dpGaps += pow((gaps[i] - gap_m), 2);
	dpGaps /= EXECS;
	dpGaps = sqrt(dpGaps);
	//-----------------------------------

	gap = ((h_knapsack->getOptSol() - best) / (float)h_knapsack->getOptSol()) * 100.0f;

	std::cout << dpGaps << ", best gap = " << gap << ", opt = " << optCounter << "\n";
	results << file << ", " << optCounter<< ", " << gap << ", " << best << ", " << gap_m << ", " << fit_m << ", " << fobj_m << ", " << dpGaps << ", " << kernel_exe_time << "\n";

	printf_s("\n\n");
	printf_s("\nTempo total=%d ms\n", kernel_exe_time);

	//-------------------------------------------------------------------------------------------
	//Free memory
	//CPU
	delete h_knapsack;
	delete[] h_solutions;
	delete[] h_bestFitness;
	fclose(filein);
	//GPU
	cg::cuda::free(devStates);
	cg::cuda::free(d_param);
	cg::cuda::free(d_knapsack);
	cg::cuda::free(d_valores);
	cg::cuda::free(d_useResources);
	cg::cuda::free(d_capResources);
	cg::cuda::free(d_bestFitness);
	cg::cuda::free(d_solutions);
	cg::cuda::free(d_particles);
	cg::cuda::free(d_positions);
	cg::cuda::free(d_velocities);

}
void
teste_sort(int* pos, Knapsack* k, int n_particles)
{
	std::ofstream f;
	f.open("fitness.out");
	int sum{ 0 };
	int * values = k->intemsValues();
	int d = k->getNumItems();
	for (int i = 0; i < n_particles*2*d; i=i+2*d)
	{
		for (int v = 0; v <d; v++)
		{
			sum += pos[i + v] * values[v];
		}
		f << sum << " ";
		sum = 0;
		for (int v = 0; v < d; v++)
		{
			sum += pos[i+d + v] * values[v];
		}
		f << sum << " \n";
		sum = 0;
	}
}