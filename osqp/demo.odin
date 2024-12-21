package osqp

import "core:log"
import "core:testing"

@(test)
test_qp :: proc(t: ^testing.T) {

	// The quadratic cost terms
	P_x := [3]Float{4.0, 1.0, 2.0}
	P_nnz: Int = 3
	P_i := [3]Int{0, 0, 1}
	P_p := [3]Int{0, 1, 3}

	// Linear cost term
	q := [2]Float{1.0, 1.0}

	// Constraints
	A_x := [4]Float{1.0, 1.0, 1.0, 1.0}
	A_nnz: Int = 4
	A_i := [4]Int{0, 1, 0, 2}
	A_p := [3]Int{0, 2, 4}
	l := [3]Float{1.0, 0.0, 0.0}
	u := [3]Float{1.0, 0.7, 0.7}
	n: Int = 2 // number of variables
	m: Int = 3 // number of constraints

	solver: ^Solver // to be freed by osqp.cleanup()

	settings := new(Settings)
	defer free(settings)

	P := new(CscMatrix)
	defer free(P)
	A := new(CscMatrix)
	defer free(A)

	csc_set_data(A, m, n, A_nnz, raw_data(A_x[:]), raw_data(A_i[:]), raw_data(A_p[:]))
	csc_set_data(P, n, n, P_nnz, raw_data(P_x[:]), raw_data(P_i[:]), raw_data(P_p[:]))

	set_default_settings(settings)
	settings.alpha = 1.0 // ADMM relaxation parameter (???)

	setup_result := setup(&solver, P, raw_data(q[:]), A, raw_data(l[:]), raw_data(u[:]), m, n, settings)
	log.debugf("setup result: %v\n", setup_result)

	solver_result := solve(solver)
	log.debugf("solver result: %v\n", solver_result)

	log.debugf("Solution!!!!!!!!!!!!: %#v, %#v:\n", solver.solution.x[0], solver.solution.x[1])

	testing.expect(t, abs((solver.solution.x[0] - 0.3) / 0.3) < 0.01)
	testing.expect(t, abs((solver.solution.x[1] - 0.7) / 0.7) < 0.01)

	cleanup(solver)
}
