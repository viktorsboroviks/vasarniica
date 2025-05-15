#include <stddef.h>
#include <cstdint>
#include <cstdio>
#include <unistd.h>
#include <string.h>


inline uint64_t nanotime(void) {uint64_t ret;__asm__ __volatile__("rdtsc" : "=A" (ret) : );return ret;}
template<typename T> inline void zero(T& A){bzero(&A, sizeof(T));}
template<typename T> inline void zero(T* A, size_t count){bzero(A, sizeof(T) * count);}


template<size_t max>
class prof
{
	uint64_t m[max][max]; //indexing as [TO][FROM], as FROM is dynamic, but TO is a CTC from template parameter
	uint64_t ot; // Old Timestamp
	size_t   fr; // FRom state

public:
	prof(): fr(0) {zero(m); ot = nanotime();}

	template<size_t to> void check(void)
	{
		static_assert(to > 0);   // 0 is reserved for init/exit state
		static_assert(to < max); // increase PROF_SLOTS value

		//It's questionable if it makes sense to run on OMP master thread only
		//but it certainly does not make sense to run in parallel,
		//as there is no defined previous state then

		#pragma omp master
		{
			uint64_t const nt = nanotime(); // New Timestamp
			m[to][fr] += nt - ot; //accumulate the diff in specific slot
			ot = nt;
			fr = to;
		}
	}

	~prof()
	{
		m[0][fr] += nanotime() - ot; //complete

		//find total - could be tracked
		uint64_t tot = 0;
		for(size_t to = 0; to < max; ++to)
			for(size_t fr = 0; fr < max; ++fr)
				tot += m[to][fr];

		for(size_t fr = 0; fr < max; ++fr) {
			for(size_t to = 0; to < max; ++to) {
				if (m[to][fr] > 0) {
					size_t const pc = size_t(m[to][fr] * 100 / tot);
					if (pc > 0)
						fprintf(stderr, "profiler: %2d -> %2d: %2d%%\t\tPROF_CHECK(%d)\n", fr, to, pc, fr);
				}
			}
		}
	}
};


//PROF_SLOTS defined to some number enables functionality
//space complexity is O(PROF_SLOTS * PROF_SLOTS), so do not overdo
#ifdef PROF_SLOTS
using prof_t = prof<PROF_SLOTS>;
extern prof_t PROF;
#define PROF_CHECK(i) do { PROF.check<i>(); } while (false)
#else
#define PROF_CHECK(i)
#endif//PROF_SLOTS


//=========== source =============


#ifdef PROF_SLOTS
prof_t PROF;
#endif//PROF_SLOTS

int main(int argc, char* argv[])
{
	PROF_CHECK(1);
	sleep(1);
	PROF_CHECK(2);
	return 0;
}


//====== sample output (not this program) ========


//   profiler:  5 ->  4: 54%		PROF_CHECK(5)
//   profiler:  6 ->  7:  2%		PROF_CHECK(6)
//   profiler:  7 ->  8:  6%		PROF_CHECK(7)
//   profiler:  8 ->  9: 15%		PROF_CHECK(8)
//   profiler:  9 -> 10:  1%		PROF_CHECK(9)
//   profiler: 11 -> 12:  6%		PROF_CHECK(11)
//   profiler: 15 -> 16:  1%		PROF_CHECK(15)
//   profiler: 19 -> 20: 10%		PROF_CHECK(19)
