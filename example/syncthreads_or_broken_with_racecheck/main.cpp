
#define SCANTEST
__device__ ptrdiff_t values[256];
#include<Kokkos_Core.hpp>



int main(int argc, char* argv[]) {
  Kokkos::initialize(argc,argv);
  {
     int N = (argc>1) ? atoi(argv[1]) : 257;
    
     Kokkos::View<ptrdiff_t*> scan_a("ScanA",N);
     auto scan_h = Kokkos::create_mirror_view(scan_a);

  
     Kokkos::parallel_scan(N,KOKKOS_LAMBDA(const int i, ptrdiff_t& val, bool is_final) {
       val+=i;
       if(is_final) scan_a(i) = val;
       //printf("Hello %i %li\n",i,val);
     });
     Kokkos::fence();
     Kokkos::deep_copy(scan_h,scan_a);

     ptrdiff_t h_values[256];
     cudaMemcpyFromSymbol(h_values,values,256*sizeof(ptrdiff_t),0);
     for(int i=0; i<5; i++) printf("Values: %i %li\n",i,h_values[i]);
     ptrdiff_t val = 0;
     for(int i=0; i<N; i++) {
       val += i;
       if(val != scan_h(i))
         std::cout << "Error: " << i << " val: " << val << " scan_h " << scan_h(i) << std::endl;
     }
// This doesn't work as reproducer ...
/*
     Kokkos::View<unsigned int> flag("FLAG");
     Kokkos::deep_copy(flag,0);
     unsigned int* global_flags = flag.data();
     Kokkos::parallel_for(N,KOKKOS_LAMBDA(const int i) {
#ifdef __CUDA_ARCH__
       int block_count = gridDim.x;
       __threadfence();
       int current_count=0;
       if(threadIdx.y==0)
         current_count = atomicInc(global_flags, block_count - 1);
       int syncthreadsorinput = int(threadIdx.y? 0: (1 + current_count < block_count));
       const bool is_last_block = !__syncthreads_or(syncthreadsorinput);

       printf("ParForTest: %i %i %i %i %i %i %i\n",int(blockIdx.x),int(threadIdx.y),is_last_block?1:0,int(block_count),int(gridDim.x),current_count,syncthreadsorinput);
#else
  *global_flags += 1;
#endif
     });
*/
     Kokkos::fence();
  }
  Kokkos::finalize();
}
