package numerical_algebra

sub :: proc {
	sub_mat_alloc,
	sub_mat_fill,
	sub_scalar_alloc,
	sub_scalar_fill,
}


// Element-wise subtraction of two matrices
sub_mat_alloc :: proc(a, b: Matrix, allocator := context.allocator) -> (res: Matrix, err: Matrix_Error) {
	if a.rows != b.rows {
		return Matrix{}, Dimension_Mismatch{}
	}
	if a.cols != b.cols {
		return Matrix{}, Dimension_Mismatch{}
	}

	res = alloc(a.rows, a.cols, allocator) or_return


	for i in 0 ..< len(a.data) {
		res.data[i] = a.data[i] - b.data[i]
	}
	return res, nil
}


// Element-wise subtraction of two matrices, output into a given matrix (no internal allocations)
// The output matrix may be one of the input matrices
sub_mat_fill :: proc(out, a, b: Matrix) -> (err: Matrix_Error) {
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
		out.data[i] = a.data[i] - b.data[i]
	}

	return nil
}


// Subtract the given scar from every element of the given matrix
// Allocate a matrix for the output
sub_scalar_alloc :: proc(a: Matrix, s: f64, allocator := context.allocator) -> (res: Matrix, err: Matrix_Error) {
	res = alloc(a.rows, a.cols, allocator) or_return

	for i in 0 ..< len(a.data) {
		res.data[i] = a.data[i] - s
	}

	return res, nil
}


// Subtract the given scalar to every element of the given matrix
// Output in a user-provided matrix (which may be the input matrix)
sub_scalar_fill :: proc(out, a: Matrix, s: f64) -> (err: Matrix_Error) {
	if a.rows != out.rows {
		return Dimension_Mismatch{}
	}
	if a.cols != out.cols {
		return Dimension_Mismatch{}
	}

	for i in 0 ..< len(a.data) {
		out.data[i] = a.data[i] - s
	}

	return nil
}
