#include <cstdlib>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <limits>
#ifndef __Particle__
#define __Particle__
#include "curand_kernel.h"
#include "Knapsack.h"


//#define penaltyFactor 5000//Fator de penalização, testar valores
//#define omega 0.3 // fator de inércia 
//#define c1 2.0
//#define c2 2.0
constexpr float bottomLimit = std::numeric_limits<float>::lowest();
struct param{    
    float c1; //constantes - intesificação e diversificação
    float c2;   
    float w; // fator de inércia
    float penalty; // penalidade função objetivo
		HOST DEVICE
    param(float c1, float c2, float w, float pf):
    c1{c1},c2{c2},w{w},penalty{pf}
    {
        //do nothing
    }
		HOST DEVICE
			param() {};
		HOST DEVICE
    int size()
    {
        return 4;
    }
		HOST
    void print()
    {
        printf("c1: %f\nc2: %f\nw: %f\np: %f\n", this->c1, this->c2, this->w, this->penalty);
    }
    
};
struct param_limits{
    float lc1[2];
    float lc2[2];
    float lw[2];
    float lpenalty[2];
    void c1(float i, float s)
    {
        this->lc1[0] = i;
        this->lc1[1] = s;
    }
    void c2(float i, float s)
    {
        this->lc2[0] = i;
        this->lc2[1] = s;
    }
    void w(float i, float s)
    {
        this->lw[0] = i;
        this->lw[1] = s;
    }
    void penalty(float i, float s)
    {
        this->lpenalty[0] = i;
        this->lpenalty[1] = s;
    }
    void print()
    {
        printf("c1 ;%f %f\n c2: %f %f\nw: %f %f\np: %f %f\n",this->lc1[0],
        this->lc1[1], this->lc2[0], this->lc2[1],this->lw[0], this->lw[1],
         this->lpenalty[0], this->lpenalty[1]);
    }
    
};


class Particle
{
	protected: 
		int _dimension;
		int* _pos; // vector[_dimension]
		float* _vel; // vector[_dimension]
		float _fitness;
		int* _pBest; // vector[_dimension]
		float _pBestFitness;    
    Knapsack* _knapsack;
    param * _parameters;
    int posLin(int);

		Particle() {}
		
	public:
		HOST 
		Particle(Knapsack* k, param* p)
		{
			_dimension = k->numItems;
			_knapsack = k;
			_parameters = p;
			//Aloca os vetores
			_pos = new int[_dimension];
			_vel = new float[_dimension];
			_pBest = new int[_dimension];

			_pBestFitness = bottomLimit;
			//printf_s("particle bestfitness = %f\n", _pBestFitness);
		}
		HOST DEVICE
		Particle(Knapsack* k, param* p, int* pos , int* best, float* vel):_dimension{ k->numItems}
		{
			//_dimension = k->numItems;
			_knapsack = k;
			_parameters = p;

			//Aloca os vetores
			_pos = pos;
			_vel = vel;
			_pBest = best;

			_pBestFitness = bottomLimit;			
		}
		HOST DEVICE
    void setupParticle();
		HOST DEVICE
		void computeFitness();
		HOST DEVICE
		float computePenalty();
		HOST DEVICE
		float computeDynamicPenalty();
		HOST DEVICE
    void updateBestPosition();
		HOST DEVICE
    void updateVelocity(int*);
		HOST DEVICE
    void updatePosition();
		HOST DEVICE
    float sigmoid(float);
		HOST DEVICE
		int objFunction();
		HOST DEVICE
    ~Particle();
		HOST DEVICE
		bool operator > (const Particle& other)
		{
			return _pBestFitness > other._pBestFitness;
		}

    HOST DEVICE
    auto dimension() const
    {
        return _dimension;
    }
		HOST DEVICE
			auto numResources() const
		{
			return _knapsack->getNumResources();
		}
	HOST DEVICE
    auto fitness() const
    {
        return _fitness;
    }
	HOST DEVICE
    auto pos(int i)const
    {
        return _pos[i];
    }
	HOST DEVICE
    auto bestFitness() const
    {
        return _pBestFitness;
    }
	HOST DEVICE
    int c1()const
    {
        return _parameters->c1;   
    }
	HOST DEVICE
    int c2()const{
        return _parameters->c2;   
    }
	HOST DEVICE
    float w()const{
        return _parameters->w;
    }
	HOST DEVICE
    float penalty()const{
        return _parameters->penalty;
    }
	HOST DEVICE
	const auto pBest()
	{
		return _pBest;
	}
};


inline void
Particle::setupParticle()//TODO
{
	//srand(time(NULL));
	//Inicialização das posições _MKP
	//Inicialização das velocidades MKP

	for (int i = 0; i < _dimension; i++)
	{
		_pos[i] = rand() % 2;
		_vel[i] = rand() % 2;
	}
	computeFitness();
}



//Método para o cálculo do fitness com penalização para soluções inviáveis
HOST DEVICE 
void
Particle::computeFitness()
{
	//DEIXEI COMO SUGERIDO NO ARTIGO, TENTEM ALTERAR OS CÁLCULOS DA PENALIZAÇÃO  TODO

	float value;

	value = 0.0;

	for (int i = 0; i < _knapsack->numItems; i++)
		value += _pos[i] * _knapsack->values[i];

	

	_fitness = value;// - this->penalty() * penalty;

}
HOST DEVICE 
float
Particle::computePenalty()
{
	float penalty{ 0.f };

	for (int i = 0; i < _knapsack->numResources; i++)
		penalty += posLin(i);
	_fitness -= this->penalty() * penalty;

	return penalty;
}
HOST DEVICE 
float
Particle::computeDynamicPenalty()
{
	auto resources = _knapsack->itemsUseResources();
	auto limitResources = _knapsack->resourcesCapacity();

	float penalty{0.f};
	float rAcc{ 0.f };
	float sPenalty{0.f};
	for (int r = 0; r <numResources(); r++) {
		rAcc = 0.0f;
		for (int i = 0; i < _dimension; i++) {
			rAcc += _pos[i] * resources[r][i];
		}
		if (rAcc > limitResources[r])
			sPenalty += rAcc;
	}
	if(sPenalty)
		_fitness =(_fitness / sPenalty);

	return sPenalty;
}
HOST DEVICE
inline int
Particle::posLin(int index)
{
	int spentResources, sqrSpentResources;
	spentResources = sqrSpentResources = 0;

	for (int i = 0; i < _knapsack->numItems; i++)
	{
		spentResources += _knapsack->useResources[index][i] * _pos[i];
		sqrSpentResources += (_knapsack->useResources[index][i] * _pos[i]);//^ 2;
	}
		

	spentResources -= _knapsack->capResources[index];

	if (spentResources < 0)//Não esgotou o recurso
		return 0;

	//return spentResources;
	return sqrSpentResources;
}
HOST DEVICE
inline void
Particle::updateBestPosition()
{
	for (int i = 0; i < _dimension; i++)
		_pBest[i] = _pos[i];

	_pBestFitness = _fitness;
}
HOST
inline void
Particle::updateVelocity(int* gBest)//TODO
{
	//Atualização de velocidades    
	for (int d = 0; d < _dimension; d++)
	{
		_vel[d] = w() * _vel[d] + c1() * rand() * (_pBest[d] - _pos[d]) + c2() * rand() * (gBest[d] - _pos[d]);
	}
}
HOST
inline void
Particle::updatePosition()//TODO
{
	//Atualização das posições
	for (int d = 0; d < _dimension; d++)
	{
		if (sigmoid(_vel[d]) >= (rand() % 2))
			_pos[d] = abs(_pos[d] - 1);
	}
	//Atualiza o fitness equivalente à solução.
	computeFitness();
}
HOST DEVICE
inline float
Particle::sigmoid(float vel)
{
	return ((float) 1.0 / (1.0 + exp(-1.0 * vel)));
}

inline 
Particle::~Particle()
{
	delete[] _pos;
	delete[] _vel;
	delete[] _pBest;
}
HOST DEVICE
inline int
Particle::objFunction()
{
	int sum{ 0 };
	int * v = _knapsack->intemsValues();
	for (auto i = 0; i < _dimension; i++)
	{
		sum += _pBest[i] * v[i];
	}
	return sum;
}

class CUDAParticle : public Particle
{
public:
	CUDAParticle() {}
	DEVICE
	CUDAParticle(Knapsack* k, param* p, int* pos, int* best, float* vel, curandState* randState) :
		Particle{ k, p, pos, best, vel}, _state{randState}
	{
		for (int i = 0; i < _dimension; i++)
		{
			_pBest[i] = 0;
			_pos[i] = 3;
		}
	}
 DEVICE
	void setParticleAttributes(Knapsack* k, param* p, int* pos, int* best, float* vel, curandState* randState);
DEVICE
  void setupParticle();//int*);
 DEVICE 
	void updatePosition(bool permitUnviableSol);
 DEVICE 
	 void updateVelocity(int *, float, float, float);
protected:
 DEVICE
	 void newSolution();
 DEVICE
	 void newViableSolution();


	
	~CUDAParticle() {};
private:
	curandState* _state;
};



DEVICE
void CUDAParticle::setParticleAttributes(Knapsack* k, param* p, int* pos, int* best, float* vel, curandState* randState)
{
	_dimension = k->numItems;
	_knapsack = k;
	_parameters = p;
	_state = randState;
	//Aloca os vetores
	_pos = pos;
	_vel = vel;
	_pBest = best;

	_pBestFitness = bottomLimit;		
}
DEVICE
void 
CUDAParticle::setupParticle()//int * pos)
{
	int v{0};
	/*if (threadIdx.x == 0)
		pos[0] = 3;
	int desl = 0/(threadIdx.x * _dimension * 2) + blockDim.x * blockIdx.x * 2;
	for (int i = desl; i < desl + _dimension; i++)
	{
		v = curand(_state) % 2;
		//		if (v < 0)
			//		v = -v;
		pos[i] = v;
		//v = curand(_state) % 2;
		//if (v < 0)
			//v = -v;
		//_vel[i] = v;

	}*/
	for (int i = 0; i < _dimension; i++)
	{
		v = curand(_state) % 2;
//		if (v < 0)
	//		v = -v;
		_pos[i] = v;
	
		//if (v < 0)
			//v = -v;
		_vel[i] = curand_uniform(_state);
		
	}
	computeFitness();
}
DEVICE
void
CUDAParticle::updatePosition(bool permitUnviableSol=true)
{
	if (permitUnviableSol) 
	{
		newSolution();
		//Atualiza o fitness equivalente à solução.
		computeFitness();
		//Atualiza o fitness com a penalidade associada à solução.
		//computePenalty();
		computeDynamicPenalty();
	}		
	else 
	{
		newViableSolution();
		//Atualiza o fitness equivalente à solução.
		computeFitness();
	}	
}
DEVICE
void
CUDAParticle::newSolution()
{
	float delta;
	//Atualização das posições
	for (int d = 0; d < _dimension; d++)
	{
		delta = curand_uniform(_state);
		if (delta < sigmoid(_vel[d]))
			_pos[d] = 1;// abs(_pos[d] - 1);
		else
			_pos[d] = 0;
	}
}
DEVICE
void
CUDAParticle::newViableSolution()
{
	auto rec = new int[numResources()];

	for (int i = 0; i < numResources(); i++)
		rec[i] = 0;

	int v;
	auto useResources = _knapsack->useResources;
	auto knapsackResources = _knapsack->capResources;

	for (int d = 0; d < dimension(); d++)
	{
		v = curand(_state) % 2;
		if (sigmoid(_vel[d]) <= (v))
			v = abs(_pos[d] - 1);
		if (v) {
			for (int i = 0; i < numResources(); i++)
			{				
				if ((rec[i] + useResources[i][d]) > knapsackResources[i])
				{
					v = 0;
					break;
				}				
				rec[i] += useResources[i][d];
			}
		}
		_pos[d] = v;
	}

	delete [] rec;
}

DEVICE
void 
CUDAParticle::updateVelocity( int * gBest, float w, float pc1, float pc2)
{
	int max{300};
	int min{ -4 };
	int rand1{ 0 }, rand2{0};
	//Atualização de velocidades    
	for (int d = 0; d < _dimension; d++)
	{
		rand1 = (curand(_state) %max);		
		rand2 = (curand(_state) %max);		
		_vel[d] = w * _vel[d] + pc1 * rand1 * (_pBest[d] - _pos[d]) + pc2 * rand2 * (gBest[d] - _pos[d]);
		/*if (_vel[d] > max)
			_vel[d] = max;
		if (_vel[d] < min)
			_vel[d] = min;*/
	}
}

#endif //__Particle__