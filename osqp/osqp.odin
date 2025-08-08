package osqp
// A set of bindings for OSQP
// Apache-2.0 license
// v1.0.0 (236713ce) (2025-08-07)
// from https://github.com/osqp/osqp

when ODIN_OS == .Windows {
	// This source directory must contain osqp.lib and osqp.dll
	// Your executable must be able to find osqp.dll
	foreign import osqp "osqp.lib"
} else {
	// Linux
	foreign import osqp "libosqp.so"
}


// NOTE: these types are configurable when OSQP is compiled to be 32 or 64 bit
Float :: f64
Int :: i64


osqp_linsys_solver_type :: enum i32 {
	OSQP_UNKNOWN_SOLVER = 0, /* Start from 0 for unknown solver because we index an array*/
	OSQP_DIRECT_SOLVER,
	OSQP_INDIRECT_SOLVER,
}


osqp_precond_type :: enum i32 {
	OSQP_NO_PRECONDITIONER = 0, /* Don't use a preconditioner */
	OSQP_DIAGONAL_PRECONDITIONER, /* Diagonal (Jacobi) preconditioner */
}


osqp_error_type :: enum i32 {
	OSQP_NO_ERROR = 0,
	OSQP_DATA_VALIDATION_ERROR = 1, /* Start errors from 1 */
	OSQP_SETTINGS_VALIDATION_ERROR,
	OSQP_LINSYS_SOLVER_INIT_ERROR,
	OSQP_NONCVX_ERROR,
	OSQP_MEM_ALLOC_ERROR,
	OSQP_WORKSPACE_NOT_INIT_ERROR,
	OSQP_ALGEBRA_LOAD_ERROR,
	OSQP_FOPEN_ERROR,
	OSQP_CODEGEN_DEFINES_ERROR,
	OSQP_DATA_NOT_INITIALIZED,
	OSQP_FUNC_NOT_IMPLEMENTED, /**< Function not implemented in this library */
	OSQP_LAST_ERROR_PLACE, /* This must always be the last item in the enum */
}


Settings :: struct {
	/* Note: If this struct is updated, ensure update_settings is also updated */
	device:                 Int, ///< device identifier; currently used for CUDA devices
	linsys_solver:          osqp_linsys_solver_type, ///< linear system solver to use

	// Control settings
	allocate_solution:      Int, ///< boolean; allocate solution in OSQPSolver during osqp_setup
	verbose:                Int, ///< boolean; write out progress
	profiler_level:         Int, ///< integer; level of detail for profiler annotation
	warm_starting:          Int, ///< boolean; warm start
	scaling:                Int, ///< data scaling iterations; if 0, then disabled
	polishing:              Int, ///< boolean; polish ADMM solution

	// ADMM parameters
	rho:                    Float, ///< ADMM penalty parameter
	rho_is_vec:             Int, ///< boolean; is rho scalar or vector?
	sigma:                  Float, ///< ADMM penalty parameter
	alpha:                  Float, ///< ADMM relaxation parameter

	// CG settings
	cg_max_iter:            Int, ///< maximum number of CG iterations per solve
	cg_tol_reduction:       Int, ///< number of consecutive zero CG iterations before the tolerance gets halved
	cg_tol_fraction:        Float, ///< CG tolerance (fraction of ADMM residuals)
	cg_precond:             osqp_precond_type, ///< Preconditioner to use in the CG method

	// adaptive rho logic
	adaptive_rho:           Int, ///< boolean, is rho step size adaptive?
	adaptive_rho_interval:  Int, ///< number of iterations between rho adaptations; if 0, then it is timing-based
	adaptive_rho_fraction:  Float, ///< time interval for adapting rho (fraction of the setup time)
	adaptive_rho_tolerance: Float, ///< tolerance X for adapting rho; new rho must be X times larger or smaller than the current one to change it

	// TODO: allowing negative values for adaptive_rho_interval can eliminate the need for adaptive_rho

	// termination parameters
	max_iter:               Int, ///< maximum number of iterations
	eps_abs:                Float, ///< absolute solution tolerance
	eps_rel:                Float, ///< relative solution tolerance
	eps_prim_inf:           Float, ///< primal infeasibility tolerance
	eps_dual_inf:           Float, ///< dual infeasibility tolerance
	scaled_termination:     Int, ///< boolean; use scaled termination criteria
	check_termination:      Int, ///< integer, check termination interval; if 0, checking is disabled
	check_dualgap:          Int, ///< Boolean; use duality gap termination criteria
	time_limit:             Float, ///< maximum time to solve the problem (seconds)

	// polishing parameters
	delta:                  Float, ///< regularization parameter for polishing
	polish_refine_iter:     Int, ///< number of iterative refinement steps in polishing
}


Solution :: struct {
	x:             [^]Float, ///< Primal solution
	y:             [^]Float, ///< Lagrange multiplier associated with \f$l \le Ax \le u\f$
	prim_inf_cert: [^]Float, ///< Primal infeasibility certificate
	dual_inf_cert: [^]Float, ///< Dual infeasibility certificate
}


// The workspace is internal details for OSQP
Workspace :: struct {}


Info :: struct {
	// solver status
	status:        [32]byte, ///< Status string, e.g. 'solved'
	status_val:    Int, ///< Status as Int, defined in osqp_api_constants.h
	status_polish: Int, ///< Polishing status: successful (1), unperformed (0), unsuccessful (-1)

	// solution quality
	obj_val:       Float, ///< Primal objective value
	dual_obj_val:  Float, ///< Dual objective value
	prim_res:      Float, ///< Norm of primal residual
	dual_res:      Float, ///< Norm of dual residual
	duality_gap:   Float, ///< Duality gap (primal obj - dual obj)

	// algorithm information
	iter:          Int, ///< Number of iterations taken
	rho_updates:   Int, ///< Number of rho updates performned
	rho_estimate:  Float, ///< Best rho estimate so far from residuals

	// timing information
	setup_time:    Float, ///< Setup phase time (seconds)
	solve_time:    Float, ///< Solve phase time (seconds)
	update_time:   Float, ///< Update phase time (seconds)
	polish_time:   Float, ///< Polish phase time (seconds)
	run_time:      Float, ///< Total solve time (seconds)

	// Convergence information
	primdual_int:  Float, ///< Integral of duality gap over time (primal-dual integral), requires profiling
	rel_kkt_error: Float, ///< Relative KKT error
}


Solver :: struct {
	settings:  ^Settings,
	solution:  ^Solution,
	info:      ^Info,
	workspace: ^Workspace,
}


@(default_calling_convention = "c", link_prefix = "osqp_")
foreign osqp {
	set_default_settings :: proc(settings: ^Settings) ---
	setup :: proc(solver: ^^Solver, P: ^CscMatrix, q: [^]Float, A: ^CscMatrix, l, u: [^]Float, m, n: Int, settings: ^Settings) -> Int ---
	solve :: proc(solver: ^Solver) -> Int ---
	cleanup :: proc(solver: ^Solver) ---

	// New vectors, null if no change. Exitflag 0 if no errors
	update_data_vec :: proc(solver: ^Solver, q_new, l_new, u_new: [^]Float) -> Int ---

	// Update matrices while maintaining their sparsity structures
	// This is tricky, read the original docs
	update_data_mat :: proc(solver: ^Solver, Px_new: [^]Float, Px_new_idx: [^]Int, P_new_n: Int, Ax_new: [^]Float, Ax_new_idx: [^]Int, A_new_n: Int) -> Int ---
}
