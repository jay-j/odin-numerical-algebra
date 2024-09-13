package performance
import na ".."
import "core:fmt"
import "core:math/rand"
import "core:time"

// odin run . -o:speed
// though speed is not totally necessary since the hard work is on the BLAS-side

main :: proc() {
	n := 4096
	A := na.alloc(n, n) or_else panic("Could not allocate A")
	B := na.alloc(n, n) or_else panic("Could not allocate B")

	rand.reset(1)

	for _ in 0 ..< 10 {
		na.fill_random_range(&A, -1, 1)
		na.fill_random_range(&B, -1, 1)

		timer: time.Stopwatch
		time.stopwatch_start(&timer)

		// start timer
		C := na.mul(A, B) or_else panic("matmul failed")

		// end timer
		time.stopwatch_stop(&timer)
		fmt.printf("Time: %v\n", time.stopwatch_duration(timer))

		// Something so C doesn't go unused
		assert(C.data[0] != 0)
	}
}
