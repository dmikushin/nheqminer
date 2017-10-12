#pragma once

#include "Solver.h"
#include "SolverStub.h"

#ifdef USE_CPU_XENONCAT
#include "xenoncat/xenoncat.hpp"
#endif
#ifdef USE_CUDA_DJEZO
#include "djezo/djezo.hpp"
#endif

class CPUSolverXenoncat : public Solver<cpu_xenoncat> {
public:
	CPUSolverXenoncat(int use_opt) : Solver<cpu_xenoncat>(new cpu_xenoncat(), SolverType::CPU) {
		_context->use_opt = use_opt;
	}
	virtual ~CPUSolverXenoncat() {}
};

