#include<Kokkos_Core.hpp>

int main() {
  Kokkos::initialize();
  {
    int result = 0;
    Kokkos::parallel_reduce(1,KOKKOS_LAMBDA (const int& i, int& upd) {
      upd += 1;
    },result);
    printf("Result: %i\n",result);
  }
  Kokkos::finalize();
}
