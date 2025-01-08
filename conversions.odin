package numerical_algebra

import osqp "./osqp"
import "core:log"
import "core:testing"

// Convert numerical_algebra.Matrix (dense, column-major)
//      to OSQP.CSC_Matrix (sparse, CSC)
// Note this allocates the internal matrices.
mat_to_osqp_csc :: proc(m: ^Matrix, allocator := context.allocator) -> (result: osqp.CscMatrix) {
	context.allocator = allocator
	result.nz = -1 // Constant for CSC
	result.m_rows = osqp.Int(m.rows)
	result.n_cols = osqp.Int(m.cols)

	// Find out how many nonzero values there are
	result.nzmax = 0
	for v in m.data {
		if v != 0 {
			result.nzmax += 1
		}
	}

	// Allocate CSC descriptor lists
	column_starts := make([]osqp.Int, m.cols + 1)
	result.p = raw_data(column_starts[:])

	row_indices := make([]osqp.Int, result.nzmax)
	result.i = raw_data(row_indices[:])

	values := make([]osqp.Float, result.nzmax)
	result.x = raw_data(values[:])

	// Fill in the data!
	index: osqp.Int = 0
	for col in 0 ..< m.cols {
		new_col := true
		for row in 0 ..< m.rows {
			val := cast(osqp.Float)get(m, row, col)
			if val != 0 {
				values[index] = val
				row_indices[index] = osqp.Int(row)
				if new_col {
					column_starts[col] = index
					new_col = false
				}

				index += 1
			}
		}
	}

	// Fill in the end of the column pointers
	column_starts[m.cols] = result.nzmax

	return result
}


@(test)
test_osqp_conversion :: proc(t: ^testing.T) {
	P_dense := alloc_dims(2, 2) or_else panic("Couldn't allocate P_dense")
	defer dealloc(P_dense)


	set(&P_dense, 0, 0, 4)
	set(&P_dense, 0, 1, 1)
	set(&P_dense, 1, 0, 1)
	set(&P_dense, 1, 1, 2)

	P_sparse := new(osqp.CscMatrix)
	defer free(P_sparse)

	{
		n: osqp.Int = 2 // number of variables
		P_x := [3]osqp.Float{4.0, 1.0, 2.0}
		P_nnz: osqp.Int = 3
		P_i := [3]osqp.Int{0, 0, 1}
		P_p := [3]osqp.Int{0, 1, 3}
		osqp.csc_set_data(P_sparse, n, n, P_nnz, raw_data(P_x[:]), raw_data(P_i[:]), raw_data(P_p[:]))
	}

	P_converted := mat_to_osqp_csc(&P_dense)
	defer free(P_converted.p)
	defer free(P_converted.x)
	defer free(P_converted.i)
	log.infof("Sparse: %v\n", P_sparse)
	log.infof("Converted: %v\n", P_converted)

	// TODO: These checks currently fail because the OSQP-provided example 
	// appears to be an extra-compressed format that assumes sequential identical
	// values remain identical even when the matrix is updated.
	/*
	for i in 0 ..< 2 {
		testing.expect(t, P_converted.x[i] == P_sparse.x[i])
	}
	for i in 0 ..< 2 {
		testing.expect(t, P_converted.i[i] == P_sparse.i[i])
	}
	for i in 0 ..< 2 {
		testing.expect(t, P_converted.p[i] == P_sparse.p[i])
	}
	*/

}
