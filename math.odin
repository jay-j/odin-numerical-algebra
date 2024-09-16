package numerical_algebra
import "core:fmt"
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
Matrix :: struct {
	data: []f32, // pointer to data + length
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
	res.data = make([]f32, capacity, allocator = allocator) or_return
	res.rows = rows
	res.cols = cols
	return res, nil
}

alloc :: proc {
	alloc_dims,
}


dealloc :: proc(m: Matrix) {
	delete(m.data)
}


/////////////////////////////////////////////////////////////////////////////////////////
// Utilities
// FEATURE: reshape? via storing & changing stride

fill_zero :: proc(m: ^Matrix) {
	slice.zero(m.data)
}

fill_random_range :: proc(m: ^Matrix, low, high: f32, rng := context.random_generator) {
	for i in 0 ..< len(m.data) {
		m.data[i] = rand.float32_uniform(low, high, rng)
	}
}

// Set an individual element of the matrix
set :: #force_inline proc(m: ^Matrix, row, col: int, value: f32) {
	i := col * m.rows + row
	m.data[i] = value
}


// Set an column of the matrix to a slice
// TODO what is a convenient pattern for checking the error result? How can this be squashed for optimized compiles?
set_col :: proc(m: ^Matrix, col: int, value: []f32) -> (err: Matrix_Error) {
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
set_row :: proc(m: ^Matrix, row: int, value: []f32) -> (err: Matrix_Error) {
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

// TODO set columns/rows/sections to a given other Matrix?

get :: #force_inline proc(m: ^Matrix, row, col: int) -> (value: f32) {
	i := col * m.rows + row
	value = m.data[i]
	return
}


get_col :: proc(m: ^Matrix, col: int, allocator := context.allocator) -> (values: []f32, err: Matrix_Error) {
	if col < 0 {
		return nil, Out_of_Bounds{}
	}
	if col >= m.cols {
		return nil, Out_of_Bounds{}
	}
	values = make([]f32, m.rows, allocator = allocator) or_return
	copy(values[:], m.data[col * m.rows:(col + 1) * m.rows])
	return values, nil
}


get_row :: proc(m: ^Matrix, row: int, allocator := context.allocator) -> (values: []f32, err: Matrix_Error) {
	if row < 0 {
		return nil, Out_of_Bounds{}
	}
	if row >= m.rows {
		return nil, Out_of_Bounds{}
	}
	values = make([]f32, m.cols, allocator = allocator) or_return
	for col in 0 ..< m.cols {
		values[col] = get(m, row, col)
	}
	return values, nil
}


linspace :: proc {
	linspace_alloc,
}


// Makes a linspaced column vector
linspace_alloc :: proc(low, high: f32, N: int, allocator := context.allocator) -> (m: Matrix, err: Matrix_Error) {
	dx: f32 = (high - low) / f32(N - 1)
	m = alloc(N, 1, allocator) or_return
	for i in 0 ..< N {
		m.data[i] = dx * f32(i) + low
	}
	return m, nil
}

// transpose :: proc(m: Matrix, allocator := context.allocator) -> (mt: Matrix, err: Matrix_Error) {
// TODO: if strides are available, this is essentially free
// }

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

	fill_random_range(&A, -1, 1)
	fill_random_range(&B, -1, 1)

	C, err := mul(A, B)
	defer dealloc(C)

	testing.expect_value(t, err, nil)
	testing.expect_value(t, C.rows, 3)
	testing.expect_value(t, C.cols, 1)

	print(C, "C")
}

@(test)
test_set_get :: proc(t: ^testing.T) {
	rand.reset(2)

	A, _ := alloc(9, 12)
	defer dealloc(A)
	fill_random_range(&A, -1, 1)

	new_col := make([]f32, 9)
	defer delete(new_col)
	slice.fill(new_col, 8)

	new_row := make([]f32, 12)
	defer delete(new_row)
	slice.fill(new_row, 33)

	set_col(&A, 3, new_col)

	set_row(&A, 6, new_row)

	testing.expect_value(t, A.data[6], 33)
	testing.expect_value(t, A.data[27], 8)

	print(A, "A")

	mycol := get_col(&A, 4) or_else panic("Couldn't allocate")
	defer delete(mycol)
	testing.expect_value(t, len(mycol), 9)

	myrow := get_row(&A, 2) or_else panic("Couldn't get row")
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
