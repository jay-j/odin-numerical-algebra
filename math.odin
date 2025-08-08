package numerical_algebra
import "core:fmt"
import "core:log"
import "core:math/rand"
import "core:mem"
import "core:slice"
import "core:testing"

// Column-major
// TODO: allow matrix slicing? Output into a chunk of a bigger matrix rather than needing to copy. Would need to also record stride.
// Column-Major Matrix
//  0   4   8
//  1   5   9
//  2   6  10
//  3   7  11
// Matrix is a bundle containing a pointer to the data and have immutable size
Matrix :: struct {
	data: []f64, // pointer to data + length
	rows: int,
	cols: int,
}

index :: #force_inline proc(m: Matrix, row, col: int) -> (i: int) {
	i = col * m.rows + row
	return i
}

Matrix_Error :: union {
	mem.Allocator_Error,
	Dimension_Mismatch,
	Out_of_Bounds,
}

Dimension_Mismatch :: struct {}
Allocation_Failed :: struct {}
Out_of_Bounds :: struct {}

alloc_dims :: proc(rows, cols: int, allocator := context.allocator) -> (res: Matrix, err: Matrix_Error) {
	capacity := rows * cols
	res.data = make([]f64, capacity, allocator = allocator) or_return
	res.rows = rows
	res.cols = cols
	return res, nil
}

alloc_like :: proc(src: Matrix, allocator := context.allocator) -> (dst: Matrix, err: Matrix_Error) {
	dst, err = alloc_dims(src.rows, src.cols, allocator)
	return
}

alloc :: proc {
	alloc_dims,
	alloc_like,
}


dealloc :: proc(m: Matrix) {
	delete(m.data)
}


/////////////////////////////////////////////////////////////////////////////////////////
// Set Utilities
// FEATURE: reshape? via storing & changing stride


// Set an individual element of the matrix
set :: #force_inline proc(m: Matrix, row, col: int, value: f64) {
	i := col * m.rows + row
	m.data[i] = value
}


// Set an column of the matrix to a slice
// TODO what is a convenient pattern for checking the error result? How can this be squashed for optimized compiles?
set_col :: proc(m: Matrix, col: int, value: []f64) -> (err: Matrix_Error) {
	if col >= m.cols {
		return Out_of_Bounds{}
	}
	if col < 0 {
		return Out_of_Bounds{}
	}
	if len(value) != m.rows {
		return Dimension_Mismatch{}
	}
	index_start := m.rows * col
	index_end := index_start + m.rows
	copy(m.data[index_start:index_end], value)
	return nil
}

// Set a row of the matrix to a slice
set_row :: proc(m: Matrix, row: int, value: []f64) -> (err: Matrix_Error) {
	if row >= m.rows {
		return Out_of_Bounds{}
	}
	if row < 0 {
		return Out_of_Bounds{}
	}
	if len(value) != m.cols {
		return Dimension_Mismatch{}
	}
	for col in 0 ..< m.cols {
		set(m, row, col, value[col])
	}
	return nil
}


// Copy a matrix into a 2D chunk of a destination matrix
set_submatrix :: proc(m: Matrix, row, col: int, submatrix: Matrix) -> (err: Matrix_Error) {
	// Make sure the submatrix can fit
	if row + submatrix.rows > m.rows {
		return Out_of_Bounds{}
	}
	if col + submatrix.cols > m.cols {
		return Out_of_Bounds{}
	}

	// ASSUME: Column-major matrix; copy over one column of the submatrix at a time
	for c: int = 0; c < submatrix.cols; c += 1 {
		col_dst := c + col
		copy(
			m.data[m.rows * col_dst + row:m.rows * col_dst + submatrix.rows + row],
			submatrix.data[submatrix.rows * c:submatrix.rows * (c + 1)],
		)
	}

	return nil
}

/////////////////////////////////////////////////////////////////////////////////////////
// Get Utilities

get :: #force_inline proc(m: Matrix, #any_int row, col: int) -> (value: f64) {
	i := col * m.rows + row
	value = m.data[i]
	return
}

// Get a multi-pointer to the raw matrix data. 
raw :: #force_inline proc(m: Matrix) -> (data: [^]f64) {
	data = raw_data(m.data[:])
	return data
}


// Allocates a new slice to return the requested column of data.
get_col_alloc :: proc(
	m: Matrix,
	#any_int col: int,
	allocator := context.allocator,
) -> (
	values: []f64,
	err: Matrix_Error,
) {
	if col < 0 {
		return nil, Out_of_Bounds{}
	}
	if col >= m.cols {
		return nil, Out_of_Bounds{}
	}
	values = make([]f64, m.rows, allocator = allocator) or_return
	copy(values[:], m.data[col * m.rows:(col + 1) * m.rows])
	return values, nil
}


// Fill the provided slice with a column of the given matrix.
// This procedure is intended for usage when you want the values on the stack.
get_col_fill :: proc(m: Matrix, #any_int col: int, values: []f64) -> (err: Matrix_Error) {
	if col < 0 {
		return Out_of_Bounds{}
	}
	if col >= m.cols {
		return Out_of_Bounds{}
	}

	copy(values[:], m.data[col * m.rows:(col + 1) * m.rows])
	return nil
}


get_col :: proc {
	get_col_alloc,
	get_col_fill,
}


get_row :: proc(m: Matrix, #any_int row: int, allocator := context.allocator) -> (values: []f64, err: Matrix_Error) {
	if row < 0 {
		return nil, Out_of_Bounds{}
	}
	if row >= m.rows {
		return nil, Out_of_Bounds{}
	}
	values = make([]f64, m.cols, allocator = allocator) or_return
	for col in 0 ..< m.cols {
		values[col] = get(m, row, col)
	}
	return values, nil
}


// Rangtes use the slice convention of [min, max)
get_submatrix :: proc(
	m: Matrix,
	row_range, col_range: [2]int,
	allocator := context.allocator,
) -> (
	s: Matrix,
	err: Matrix_Error,
) {
	when ODIN_DEBUG {
		if row_range[1] < row_range[0] {
			return Matrix{}, Out_of_Bounds{}
		}
		if row_range[0] < 0 {
			return Matrix{}, Out_of_Bounds{}
		}
		if row_range[1] > m.rows {
			return Matrix{}, Out_of_Bounds{}
		}

		if col_range[1] < col_range[0] {
			return Matrix{}, Out_of_Bounds{}
		}
		if col_range[0] < 0 {
			return Matrix{}, Out_of_Bounds{}
		}
		if col_range[1] > m.cols {
			return Matrix{}, Out_of_Bounds{}
		}
	}

	context.allocator = allocator

	s, err = alloc_dims(row_range[1] - row_range[0], col_range[1] - col_range[0])
	if err != nil {
		return Matrix{}, err
	}

	for col_og, col_new in col_range[0] ..< col_range[1] {
		// Data fills entire columns of the output matrix
		// Data may be sourced from partial column of the input matrix
		copy(
			s.data[col_new * s.rows:(col_new + 1) * s.rows],
			m.data[col_og * m.rows + row_range[0]:col_og * m.rows + row_range[1]],
		)
	}
	return s, nil
}


@(test)
test_submatrix :: proc(t: ^testing.T) {
	src := alloc_dims(4, 5) or_else panic("Test alloc error")
	defer dealloc(src)

	copy(src.data, []f64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20})
	// col:      0   1    2    3    4
	//         -------------------------
	// row: 0 |  1   5    9   13   17
	//      1 |  2   6  *10   14*  18
	//      2 |  3   7   11   15   19
	//      3 |  4   8  *12   16*  20

	dst := get_submatrix(src, row_range = {1, 4}, col_range = {2, 4}) or_else panic("Test alloc error")
	defer dealloc(dst)

	testing.expect_value(t, get(src, 1, 2), get(dst, 0, 0))
	testing.expect_value(t, get(src, 3, 2), get(dst, 2, 0))
	testing.expect_value(t, get(src, 1, 3), get(dst, 0, 1))
	testing.expect_value(t, get(src, 3, 3), get(dst, 2, 1))

	// BUG: This test is reporting a leak and bad free.. how?!
}


// Allocate a new matrix which is the transpose of the original
// PERFORMANCE: This becomes free and could allow in-pace operation if Matrix supports stride
transpose :: proc(m: Matrix, allocator := context.allocator) -> (mt: Matrix, err: Matrix_Error) {
	mt = alloc_dims(rows = m.cols, cols = m.rows, allocator = allocator) or_return
	for col in 0 ..< m.cols {
		for row in 0 ..< m.rows {
			index_old := col * m.rows + row
			index_new := row * mt.rows + col
			mt.data[index_new] = m.data[index_old]
		}
	}
	return mt, nil
}


@(test)
test_transpose :: proc(t: ^testing.T) {
	orig := alloc_dims(4, 3) or_else panic("Matrix allocation error")
	defer dealloc(orig)
	for i in 0 ..< orig.rows * orig.cols {
		orig.data[i] = f64(i)
	}

	transp := transpose(orig) or_else panic("Matrix allocation error")
	defer dealloc(transp)
	testing.expect_value(t, get(transp, 1, 1), get(orig, 1, 1))
	testing.expect_value(t, get(transp, 2, 1), get(orig, 1, 2))
	testing.expect_value(t, get(transp, 2, 3), get(orig, 3, 2))
}


/////////////////////////////////////////////////////////////////////////////////////////
// Fill Utilities

fill_value :: proc(m: Matrix, val: f64) {
	slice.fill(m.data, val)
}

fill_zero :: proc(m: Matrix) {
	slice.zero(m.data)
}

fill_random_range :: proc(m: Matrix, low, high: f64, rng := context.random_generator) {
	for i in 0 ..< len(m.data) {
		m.data[i] = rand.float64_uniform(low, high, rng)
	}
}


linspace :: proc {
	linspace_alloc,
	linspace_fill,
}


// Allocate to make a linspaced column vector
linspace_alloc :: proc(low, high: f64, N: int, allocator := context.allocator) -> (m: Matrix, err: Matrix_Error) {
	dx: f64 = (high - low) / f64(N - 1)
	m = alloc(N, 1, allocator) or_return
	for i in 0 ..< N {
		m.data[i] = dx * f64(i) + low
	}
	return m, nil
}


// Fill a given Nx1 or 1xN matrix using linspace
linspace_fill :: proc(m: Matrix, low, high: f64) -> (err: Matrix_Error) {
	if m.rows > 1 && m.cols > 1 {
		err = Dimension_Mismatch{}
		return
	}

	N := max(m.rows, m.cols)
	dx: f64 = (high - low) / f64(N - 1)
	for i in 0 ..< N {
		m.data[i] = dx * f64(i) + low
	}
	return nil
}


identity :: proc {
	identity_alloc,
	identity_fill,
}

identity_alloc :: proc(dim: int, allocator := context.allocator) -> (m: Matrix, err: Matrix_Error) {
	if dim < 1 {
		return Matrix{}, Out_of_Bounds{}
	}
	m, err = alloc_dims(dim, dim, allocator)
	if err != nil {
		return Matrix{}, err
	}

	for i in 0 ..< dim {
		set(m, i, i, 1)
	}
	return m, nil
}


// Turn a matrix into an identity matrix
// If the input matrix is non-square, still fills in each [i,i] element
identity_fill :: proc(m: Matrix) -> (err: Matrix_Error) {
	if raw_data(m.data) == nil {
		return Out_of_Bounds{}
	}
	fill_zero(m)
	N := min(m.rows, m.cols)
	for i in 0 ..< N {
		set(m, i, i, 1)
	}
	return nil
}

/////////////////////////////////////////////////////////////////////////////////////////
// Misc.

apply_scalar_operation :: proc {
	apply_scalar_operation_alloc,
	apply_scalar_operation_fill,
}


// Apply scalar operation to each element in the matrix
apply_scalar_operation_alloc :: proc(
	m: Matrix,
	f: proc(_: f64) -> f64,
	allocator := context.allocator,
) -> (
	out: Matrix,
	err: Matrix_Error,
) {
	out = alloc_like(m, allocator) or_return
	apply_scalar_operation_fill(out, m, f) or_return
	return out, nil
}


// Apply scalar operation to each element in the matrix
apply_scalar_operation_fill :: proc(out, m: Matrix, f: proc(_: f64) -> f64) -> (err: Matrix_Error) {
	if out.rows != m.rows {
		return Dimension_Mismatch{}
	}
	if out.cols != m.cols {
		return Dimension_Mismatch{}
	}

	for i in 0 ..< len(m.data) {
		out.data[i] = f(m.data[i])
	}
	return nil
}

/////////////////////////////////////////////////////////////////////////////////////////
// Printing

// TODO: write into extensions for tprintf, aprintf, etc.
print :: proc(m: Matrix, name: string = "matrix") {
	fmt.printf("%v (%v x %v):\n", name, m.rows, m.cols)
	for row in 0 ..< m.rows {
		for col in 0 ..< m.cols {
			i := index(m, row, col)
			fmt.printf("%v  ", m.data[i])
		}
		fmt.printf("\n")
	}
	fmt.printf("\n")
}

/////////////////////////////////////////////////////////////////////////////////////////
// Tests

@(test)
test_mul_vec :: proc(t: ^testing.T) {
	rand.reset(1)

	A, _ := alloc(3, 3)
	defer dealloc(A)
	B, _ := alloc(3, 1)
	defer dealloc(B)

	fill_random_range(A, -1, 1)
	fill_random_range(B, -1, 1)

	C, err := mul(A, B)
	defer dealloc(C)

	testing.expect_value(t, err, nil)
	testing.expect_value(t, C.rows, 3)
	testing.expect_value(t, C.cols, 1)

	// print(C, "C")
}

@(test)
test_set_get :: proc(t: ^testing.T) {
	rand.reset(2)

	A, _ := alloc(9, 12)
	defer dealloc(A)
	fill_random_range(A, -1, 1)

	new_col := make([]f64, 9)
	defer delete(new_col)
	slice.fill(new_col, 8)

	new_row := make([]f64, 12)
	defer delete(new_row)
	slice.fill(new_row, 33)

	set_col(A, 3, new_col)

	set_row(A, 6, new_row)

	testing.expect_value(t, A.data[6], 33)
	testing.expect_value(t, A.data[27], 8)

	// print(A, "A")

	mycol := get_col(A, 4) or_else panic("Couldn't allocate")
	defer delete(mycol)
	testing.expect_value(t, len(mycol), 9)

	myrow := get_row(A, 2) or_else panic("Couldn't get row")
	defer delete(myrow)
	testing.expect_value(t, len(myrow), 12)
}


@(test)
test_linspace :: proc(t: ^testing.T) {

	A, err := linspace(-5, 12, 32)
	defer dealloc(A)
	testing.expect_value(t, err, nil)
	testing.expect_value(t, len(A.data), 32)

	testing.expect_value(t, A.data[0], -5)
	testing.expect_value(t, A.data[31], 12)
}


@(test)
test_set_submatrix :: proc(t: ^testing.T) {
	rand.reset(3)

	A, _ := alloc(6, 8)
	defer dealloc(A)
	fill_random_range(A, 100, 200)

	B, _ := alloc(6, 8)
	defer dealloc(B)
	fill_random_range(B, 1, 2)

	C, _ := alloc(3, 4)
	defer dealloc(C)
	fill_random_range(C, -2, -1)

	set_submatrix(A, 0, 0, B)
	set_submatrix(A, 1, 1, C)

	testing.expect(t, get(A, 0, 0) > 0)
	testing.expect(t, get(A, 1, 1) < 0)

	for v in A.data {
		testing.expect(t, v < 100)
	}
}

@(test)
test_identity :: proc(t: ^testing.T) {
	m1, a1 := alloc(5, 5)
	defer dealloc(m1)
	e1 := identity_fill(m1)
	testing.expect_value(t, a1, nil)
	testing.expect_value(t, e1, nil)

	m2, e2 := identity_alloc(5)
	defer dealloc(m2)
	testing.expect_value(t, e2, nil)

}
