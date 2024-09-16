package numerical_algebra
// import "core:fmt"
import "core:log"
import "core:math"
import "core:math/rand"
import "core:testing"

Lapack_Index :: int
Lapack_Layout :: enum i32 {
	Row_Major = 101,
	Col_Major = 102,
}
Lapack_Transpose :: enum i32 {
	No_Trans   = 111,
	Trans      = 112,
	Conj_Trans = 113,
}
Lapack_UpLo :: enum i32 {
	Upper = 121,
	Lower = 122,
}
Lapack_Diag :: enum i32 {
	Non_Unit = 131,
	Unit     = 132,
}
Lapack_Side :: enum i32 {
	Left  = 141,
	Right = 142,
}

// Call the system copy of lapacke rather than needing to bundle one with this library.. for now
@(require)
foreign import lapack "system:lapacke"

@(default_calling_convention = "c", link_prefix = "LAPACKE_")
foreign lapack {
	sgelss :: proc(layout: Lapack_Layout, M, N, nrhs: i32, A: [^]f32, lda: i32, B: [^]f32, ldb: i32, S: [^]f32, rcond: f32, rank: ^i32) -> (info: i32) ---

	// A = U * SIGMA * transpose(V)
	// jobu specifies options for computing all or part of the matrix U
	// jobvt specifies options for computing all or part of the array VT
	// CAUTION: modifies data at pointers given to it!
	sgesvd :: proc(layout: Lapack_Layout, jobu, jobv: u8, M, N: i32, A: [^]f32, lda: i32, S: [^]f32, U: [^]f32, ldu: i32, VT: [^]f32, ldvt: i32, superb: [^]f32) -> (info: i32) ---

}


// Call the system blas library..
@(require)
foreign import blas "system:blas"

@(default_calling_convention = "c", link_prefix = "cblas_")
foreign blas {
	// sgecon - computes the recriprocal of the condition number of the given general matrix

	// stbsv - may be a very fast (BLAS lvl 2) solve for a well conditioned and square system.

	// C := alpha*A*B + beta*C
	sgemm :: proc(layout: Lapack_Layout, transa, transb: Lapack_Transpose, a_rows, b_cols, a_cols: i32, alpha: f32 = 1.0, A: [^]f32, lda: i32, B: [^]f32, ldb: i32, beta: f32 = 0, C: [^]f32, ldc: i32) -> (info: i32) ---

}

// NOTE: internally uses the temp allocator for work matrices. Allocates the result on the given allocator.
matrix_leastsquares :: proc(
	matrix_a: Matrix,
	matrix_b: Matrix,
	allocator := context.allocator,
) -> (
	result: Matrix,
	err: Matrix_Error,
) {
	// compute matrix pseudo-inverse using the LAPACK function dgelss
	// http://www.netlib.org/lapack/explore-html/d7/d3b/group__double_g_esolve_gaa6ed601d0622edcecb90de08d7a218ec.html#gaa6ed601d0622edcecb90de08d7a218ec
	// the resulting inverse is stored in the B matrix
	// matrix_layout = LAPACK_COL_MAJOR or LAPACK_ROW_MAJOR


	// TODO: check that the dimensions are compatible

	// nr := (M) The number of rows of matrix A. Equal to number of equations, the number of rows of B.
	// nc := (N) The number of columns of matrix A. Equal to number of unknowns in (single column) X.
	nrhs: i32 = 1 // number of columns of matrices B and X; only solve one
	lda: i32 = i32(matrix_a.rows)
	ldb: i32 = i32(matrix_a.rows > matrix_a.cols ? matrix_a.rows : matrix_a.cols)

	A := make([]f32, matrix_a.rows * matrix_a.cols, context.temp_allocator)
	B := make([]f32, matrix_a.rows, context.temp_allocator)

	// TODO: will these fail if A size is reserved for work much larger than matrix_a?
	// PERFORMANCE: at what point does the overhead of these copy operations become larger than the lapack speedup?
	copy(A, matrix_a.data)
	copy(B, matrix_b.data)

	S := make([]f32, matrix_a.rows < matrix_a.cols ? matrix_a.rows : matrix_a.cols, context.temp_allocator)
	rcond: f32 = -1
	rank: i32

	info := sgelss(
		.Col_Major,
		i32(matrix_a.rows),
		i32(matrix_a.cols),
		nrhs,
		&A[0],
		lda,
		&B[0],
		ldb,
		&S[0],
		rcond,
		&rank,
	)

	if (info < 0) {
		log.fatalf("ERROR! Illegal value in LAPACKE_sgelss() %v-th argument.\n", info)
		assert(false)
	}
	if info > 0 {
		log.fatalf(
			"ERROR! The algorithm for computing the SVD failed to converge; if INFO = i, i off-diagonal element of an intermediate bidirectional form did not converge to zero. INFO = %v\n",
			info,
		)
		assert(false)
	}

	result = alloc(matrix_a.rows, 1, allocator) or_return
	copy(result.data, B)
	return result, nil
}


@(test)
test_3x3 :: proc(t: ^testing.T) {
	//  1 4 8
	//  2 5 8
	//  3 6 9
	// A := []f32{1, 2, 3, 4, 5, 6, 8, 8, 9}
	A := alloc(3, 3) or_else panic("Couldn't allocate matrix.")
	defer dealloc(A)
	set_col(&A, 0, []f32{1, 2, 3})
	set_col(&A, 1, []f32{4, 5, 6})
	set_col(&A, 2, []f32{8, 8, 9})
	// B := []f32{4, 2, 1}
	B := alloc(3, 1) or_else panic("Couldn't allocate matrix.")
	defer dealloc(B)
	set_col(&B, 0, []f32{4, 2, 1})

	x, err := matrix_leastsquares(A, B)
	defer dealloc(x)
	testing.expect_value(t, err, nil)
	free_all(context.temp_allocator)

	testing.expect(t, math.abs(x.data[0] - (-1.3333)) < 0.01)
	testing.expect(t, math.abs(x.data[1] - (-0.666667)) < 0.01)
	testing.expect(t, math.abs(x.data[2] - 1.0) < 0.01)

	// fmt.printf("x=%v\n", x)
}


@(test)
test_5x3 :: proc(t: ^testing.T) {
	// 3 unknowns
	// 5 constraints
	// A [5, 3] @ x [3 x 1] = B [5 x 1 ]
	A := alloc(5, 3) or_else panic("Couldn't allocate matrix.")
	defer dealloc(A)
	set_col(&A, 0, []f32{5, 4, 3, 2, 1})
	set_col(&A, 1, []f32{8, 9, 0, 4, 5})
	set_col(&A, 2, []f32{2, 1, 9, 7, 5})

	B := alloc(5, 1) or_else panic("Couldn't allocate matrix.")
	defer dealloc(B)
	set_col(&B, 0, []f32{8, 7, 6, -2, -3})

	x, err := matrix_leastsquares(A, B)
	defer dealloc(x)
	testing.expect_value(t, err, nil)
	free_all(context.temp_allocator)

	testing.expect(t, math.abs(x.data[0] - 3.39806315) < 0.01)
	testing.expect(t, math.abs(x.data[1] - (-0.82739837)) < 0.01)
	testing.expect(t, math.abs(x.data[2] - (-0.57091698)) < 0.01)
}


///////////////////////////////////////////////////////////////////////////////
// Singular Value Decomposition


// Compute the singular values by doing the SVD of the given matrix
// Factorizes the matrix A(row x cols) into U(row x i) @ S(i x i) @ Vh(i x cols)
// U  := always form an orthonormal set
// S  := 1D array of singular values on the diagonal
// Vh := always form an orthonormal set
// Allocates work on context.temp_allocator and return on the given allocator
svd :: proc(A: Matrix, allocator := context.allocator) -> (result: []f32, err: Matrix_Error) {
	rows := i32(A.rows)
	cols := i32(A.cols)
	context.allocator = allocator
	matrix_a := make([]f32, cols * rows, allocator = context.temp_allocator) or_return
	copy(matrix_a[:], A.data[:])

	matrix_s := make([]f32, min(cols, rows)) or_return
	superb := make([]f32, cols * rows * 10, allocator = context.temp_allocator) or_return // TODO wtf this guess

	// The U and VT interfaces are not interacted with for the 'N' modes.

	info := sgesvd(
		.Col_Major, // NOTE: A must be column major!
		jobu = 'N',
		jobv = 'N',
		M = rows,
		N = cols,
		A = &matrix_a[0],
		lda = rows,
		S = &matrix_s[0],
		U = nil,
		ldu = 1,
		VT = nil,
		ldvt = 1,
		superb = &superb[0],
	)


	// TODO error handling with the SVD singular values!
	log.debugf("svd info = %v\n", info)

	return matrix_s[:], nil
}


@(test)
svd_basic_test :: proc(t: ^testing.T) {
	rows := 3
	cols := 4

	A := alloc(rows, cols) or_else panic("Couldn't allocate matrix.")
	defer dealloc(A)
	set_row(&A, 0, []f32{5, 2, 8, 4})
	set_row(&A, 1, []f32{3, 3, 9, 5})
	set_row(&A, 2, []f32{6, 5, 12, 9})

	s, err := svd(A)
	defer delete(s)
	testing.expect_value(t, err, nil)
	free_all(context.temp_allocator)

	approx :: proc(a, b: f32) -> (result: bool = false) {
		if math.abs(a - b) < 1e-5 {
			return true
		}
		return false
	}

	testing.expect_value(t, len(s), 3)

	testing.expect(t, approx(s[0], 22.64434879), "first singular value")
	testing.expect(t, approx(s[1], 2.0323747), "second singular value")
	testing.expect(t, approx(s[2], 1.45014521), "third singular value")
}

// https://stackoverflow.com/questions/5889142/python-numpy-scipy-finding-the-null-space-of-a-matrix
// def null_space(A, rcond=None):
//     u, s, vh = svd(A, full_matrices=True)
//     M, N = u.shape[0], vh.shape[1]
//     if rcond is None:
//         rcond = numpy.finfo(s.dtype).eps * max(M, N)
//     tol = numpy.amax(s) * rcond
//     num = numpy.sum(s > tol, dtype=int)
//     Q = vh[num:,:].T.conj()
//     return Q
// null_space :: proc(A: []f32, rows, cols: i32, allocator := context.allocator) -> (result: []f32, ok: bool = false) {
// 	context.allocator = allocator
// 	u, s, vh := svd(A, rows, cols)


// }


// Inverse:
// U, S, VH = svd(A)
// inv(A) = V @ inv(D) @ transpose(U)


///////////////////////////////////////////////////////////////////////////////
// Matmul

// TODO want more variation of these depending on whether or not the should allocate or increment C
mul :: proc {
	mul_mat_mat,
}

mul_mat_mat :: proc(a, b: Matrix, allocator := context.allocator) -> (c: Matrix, err: Matrix_Error) {

	if a.cols != b.rows {
		return Matrix{}, Dimension_Mismatch{}
	}

	c = alloc(a.rows, b.cols, allocator) or_return

	info := sgemm(
		.Col_Major,
		.No_Trans,
		.No_Trans,
		i32(a.rows),
		i32(b.cols),
		i32(a.cols),
		1.0,
		&a.data[0],
		i32(a.rows),
		&b.data[0],
		i32(b.rows),
		0,
		&c.data[0],
		i32(c.rows),
	)

	log.debugf("matmul info = %v\n", info)
	// TODO use the info return value for error reporting

	return c, nil
}


@(test)
test_matmul :: proc(t: ^testing.T) {
	a := alloc(10, 10) or_else panic("alloc failed")
	defer dealloc(a)
	b := alloc(10, 10) or_else panic("alloc failed")
	defer dealloc(b)

	rand.reset(1)
	fill_random_range(&a, -1, 1)
	fill_random_range(&b, -1, 1)

	c, err := mul(a, b)
	defer dealloc(c)
	testing.expect_value(t, err, nil)

	print(c, "AxB=C")
}
