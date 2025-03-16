package numerical_algebra
import "core:math/rand"
import "core:testing"

add :: proc {
	add_mat_alloc,
	add_mat_fill,
	add_scalar_alloc,
	add_scalar_fill,
}


// Element-wise addition of two matrices, allocate a new matrix to output
add_mat_alloc :: proc(a, b: Matrix, allocator := context.allocator) -> (res: Matrix, err: Matrix_Error) {
	if a.rows != b.rows {
		return Matrix{}, Dimension_Mismatch{}
	}
	if a.cols != b.cols {
		return Matrix{}, Dimension_Mismatch{}
	}

	res = alloc(a.rows, a.cols, allocator) or_return


	for i in 0 ..< len(a.data) {
		res.data[i] = a.data[i] + b.data[i]
	}
	return res, nil
}


// Element-wise addition of two matrices, output into a given matrix (no internal allocations)
// The output matrix may be one of the input matrices
add_mat_fill :: proc(out, a, b: Matrix) -> (err: Matrix_Error) {
	if a.rows != b.rows {
		return Dimension_Mismatch{}
	}
	if a.cols != b.cols {
		return Dimension_Mismatch{}
	}
	if a.rows != out.rows {
		return Dimension_Mismatch{}
	}
	if a.cols != out.cols {
		return Dimension_Mismatch{}
	}
	
	for i in 0 ..< len(a.data) {
		out.data[i] = a.data[i] + b.data[i]
	}

	return nil
}


// Add the given scalar to every element of the given matrix
// Allocate a matrix for the output
add_scalar_alloc :: proc(a: Matrix, s: f64, allocator := context.allocator) -> (res: Matrix, err: Matrix_Error) {
	res = alloc(a.rows, a.cols, allocator) or_return

	for i in 0 ..< len(a.data) {
		res.data[i] = a.data[i] + s
	}

	return res, nil
}


// Add the given scalar to every element of the given matrix
// Output in a user-provided matrix (which may be the input matrix)
add_scalar_fill :: proc(out, a: Matrix, s: f64) -> (err: Matrix_Error) {
	if a.rows != out.rows {
		return Dimension_Mismatch{}
	}
	if a.cols != out.cols {
		return Dimension_Mismatch{}
	}

	for i in 0 ..< len(a.data) {
		out.data[i] = a.data[i] + s
	}

	return nil
}


@(test)
test_add :: proc(t: ^testing.T) {
	rand.reset(0)

	A, _ := alloc(3, 4)
	defer dealloc(A)
	B, _ := alloc(3, 4)
	defer dealloc(B)

	fill_random_range(&A, -1, 1, context.random_generator)
	fill_random_range(&B, -1, 1, context.random_generator)

	set(&A, 0, 3, 100)
	set(&B, 2, 1, -100)

	C, err := add(A, B)
	defer dealloc(C)
	testing.expect_value(t, err, nil)

	testing.expect(t, C.data[5] < -90)
	testing.expect(t, C.data[9] > 90)
	testing.expect_value(t, C.data[0], A.data[0] + B.data[0])
}
