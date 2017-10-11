#pragma once

#include "Solver.h"
#include "SolverStub.h"


#ifdef USE_CPU_XENONCAT
#include "xenoncat/xenoncat.hpp"
#else
CREATE_SOLVER_STUB(cpu_xenoncat, "cpu_xenoncat_STUB")
#endif
#ifdef USE_CUDA_DJEZO
#include "djezo/djezo.hpp"
#else
CREATE_SOLVER_STUB(cuda_djezo, "cuda_djezo_STUB")
#endif

class CPUSolverXenoncat : public Solver<cpu_xenoncat> {
public:
	CPUSolverXenoncat(int use_opt) : Solver<cpu_xenoncat>(new cpu_xenoncat(), SolverType::CPU) {
		_context->use_opt = use_opt;
	}
	virtual ~CPUSolverXenoncat() {}
};
// TODO remove platform id for cuda solvers
// CUDA solvers
class CUDASolverDjezo : public Solver<cuda_djezo> {
public:
	CUDASolverDjezo(int dev_id, int blocks, int threadsperblock) : Solver<cuda_djezo>(new cuda_djezo(0, dev_id), SolverType::CUDA) {
		if (blocks > 0) {
			_context->blocks = blocks;
		}
		if (threadsperblock > 0) {
			_context->threadsperblock = threadsperblock;
		}
	}
	virtual ~CUDASolverDjezo() {}
};
