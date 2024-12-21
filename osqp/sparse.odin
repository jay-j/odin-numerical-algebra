package osqp

CscMatrix :: struct {
	m_rows: Int, // number of rows
	n_cols: Int, // number of columns
	p:      [^]Int, // column pointers, size n+1
	i:      [^]Int, // row indices, size nzmax starting from 0
	x:      [^]Float, // the numerical values!
	nzmax:  Int, // maximum number of entries
	nz:     Int, // number of entries in triplet matrix, -1 for CSC
}


csc_init :: proc(#any_int rows, cols: Int, allocator := context.allocator) -> (M: ^CscMatrix) {
	context.allocator = allocator

	M = new(CscMatrix)
	M.m_rows = rows
	M.n_cols = cols
	M.nz = -1
	return M
}


// This procedure is provided by OSQP but it doesn't seem to be compiled
// into the Windows binaries, so just write it out manually for all OS.
csc_set_data :: proc(M: ^CscMatrix, m, n, nzmax: Int, x: [^]Float, i, p: [^]Int) {
	M.m_rows = m
	M.n_cols = n
	M.nz = -1
	M.nzmax = nzmax
	M.x = x
	M.i = i
	M.p = p
}


// csc_set :: proc(M: ^CscMatrix, row, col: Int, val: Float) {

// 	// Figure out what index this is supposed to be in the list
// 	// Look through M.p to find the start of the column that goes here
// 	// then check P

// }
