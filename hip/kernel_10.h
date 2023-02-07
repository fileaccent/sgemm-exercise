__global__ void sgemm_10(float alpha, float *A, float *B, float beta, float *C, int N)
{

	const int Ns = 128, Ks = 8; 

	__shared__ float As[2][Ns*Ks]; 
    __shared__ float Bs[2][Ns*Ks]; 

    int bx = blockIdx.x, by = blockIdx.y; 
    int tx = threadIdx.x; 
    
    int rowa = (tx%32)*4, cola = tx/32;
    int rowb = (tx%2)*4, colb = tx/2;
    // int rowc = (tx % 16)*8, colc = (tx/16)*8; 
	
    int warp_id = tx>>5;
    int lane_id = tx&31;
    int warp_row = warp_id & 3, warp_col = warp_id >> 2;
    int row_w = lane_id&3, col_w = lane_id>>2;
    int rowc = (warp_row<<5) + (row_w<<3), colc = (warp_col<<6) + (col_w<<3);


	int abegin = Ns * bx; 
	int astep = Ks * N; 

	int bbegin = by * Ns * N; 
	int bstep = Ks; 
	int bend = bbegin + N;


    //for computatiom
    float B_reg[2][8];
    float4 A_reg1[2], A_reg2[2]; 

    float4 A_nxtile, B_nxtile; 


    // float4 Av1[2], Bv1[2]; //load from global memory to shared memory
    // float4 Bv2;  
    
    float4 Cv[16]; 
    float4 Csum[16]; 
    memset(Csum, 0, sizeof(Csum)); 

    int write_id = 1; 


    //load first tile to shared memory
    int a = abegin, b = bbegin; 
    //load tile in A
    A_nxtile = *( (float4 *)(&A[a + N * cola + rowa]) );

    *((float4 *)(&As[0][rowa + cola * Ns])) = A_nxtile;

    //load tile in B
    B_nxtile = *( (float4 *)(&B[b + N*colb + rowb]) );

    Bs[0][colb + rowb * Ns] = B_nxtile.x; 
    Bs[0][colb + (rowb + 1) * Ns] = B_nxtile.y; 
    Bs[0][colb + (rowb + 2) * Ns] = B_nxtile.z; 
    Bs[0][colb + (rowb + 3) * Ns] = B_nxtile.w;

    __syncthreads(); 


    // load from shared memory to register
    // load matrix B into register
    A_reg1[0] = *( (float4 *)(&As[0][rowc + 0*Ns]) );
    A_reg2[0] = *( (float4 *)(&As[0][rowc + 4 + 0*Ns]) );

    // load matrix B into register
    #pragma unroll 
    for(int i=0;i<8;i++)
        B_reg[0][i] = Bs[0][colc + i + 0*Ns]; 

    

 
    for(; b < bend;  a += astep, b += bstep)
    {

        //load next tile into shared memory
        //load matrix A into shared memory
        if(b + bstep < bend)
        {
            A_nxtile = *( (float4 *)(&A[a + astep + N * cola + rowa]) );
            
            //load tile in B
            B_nxtile = *( (float4 *)(&B[b + bstep  + N*colb + rowb]) );

        }
        
        int load_id = write_id^1; 
        int next_id, cur_id; 

        #pragma unroll
        for(int k=0;k<Ks;k++)
        {

            cur_id = k%2; 
            next_id = (k+1)%2; 

            // Load next register tile
            if(k < Ks - 1)
            {
                //load matrix A into register from shared memory
                A_reg1[next_id] = *( (float4 *)(&As[load_id][rowc + (k+1)*Ns]) );
                A_reg2[next_id] = *( (float4 *)(&As[load_id][rowc + 4 + (k+1)*Ns]) );

            // load matrix B into register from shared memory
                #pragma unroll 
                for(int i=0;i<8;i++)
                    B_reg[next_id][i] = Bs[load_id][colc + i + (k+1)*Ns]; 
            }


            //compute C element

            #pragma unroll 
            for(int i=0;i<8;i++)
            {
                Csum[i*2 ] += A_reg1[cur_id] * B_reg[cur_id][i]; 
                Csum[i*2 + 1] += A_reg2[cur_id] * B_reg[cur_id][i]; 
            }

        }

        if(b + bstep < bend)
        {
           *((float4 *)(&As[write_id][rowa + cola * Ns])) = A_nxtile; 
            Bs[write_id][colb + rowb * Ns] = B_nxtile.x; 
            Bs[write_id][colb + (rowb + 1) * Ns] = B_nxtile.y; 
            Bs[write_id][colb + (rowb + 2) * Ns] = B_nxtile.z; 
            Bs[write_id][colb + (rowb + 3) * Ns] = B_nxtile.w;
        }
     
        __syncthreads();
 

        A_reg1[0] = *( (float4 *)(&As[write_id][rowc + 0*Ns]) );
        A_reg2[0] = *( (float4 *)(&As[write_id][rowc + 4 + 0*Ns]) );

        // load matrix B into register
        #pragma unroll 
        for(int i=0;i<8;i++)
            B_reg[0][i] = Bs[write_id][colc + i + 0*Ns];

        write_id ^= 1;

    }

    
    #pragma unroll 
    for(int i=0;i<8;i++)
    {
        Cv[i*2] = *( (float4 *)(&C[abegin + bbegin + N * (colc + i) + rowc ]) );
        Cv[i*2+1] = *( (float4 *)(&C[abegin + bbegin + N * (colc + i) + rowc +4]) );
    }


    #pragma unroll 
    for(int i=0;i<16;i++)
    {
        Cv[i] = Csum[i] * alpha + Cv[i] * beta; 
    }


    #pragma unroll 
    for(int i=0;i<8;i++)
    {
        *( (float4 *)(&C[abegin + bbegin + N * (colc + i) + rowc ]) ) = Cv[i*2];
        *( (float4 *)(&C[abegin + bbegin + N * (colc + i) + rowc +4]) ) = Cv[i*2+1];
    }

  
}
float test10 () {
    const int m = 64;
    const int n = 64;
    const int k = 16;
    const int TM = 4;
    const int TN = 4;
    const int TK = 2;
    const int padding = 8;
    // int thread_size = (m * n + reg_size * reg_size - 1) / (reg_size * reg_size);
    test_start();
    dim3 blocks(M/128, N/128);
    int threads = 256;

    sgemm_10<<<blocks, threads>>>(1, d_A, d_B, 0, d_C, N);
    KernelErrChk();
    ErrChk(hipEventRecord(start, 0));
    for (int i = 0; i < iteration; i++) {
        sgemm_10<<<blocks, threads>>>(1, d_A, d_B, 0, d_C, N);
    }
    test_end();
    return elapsedTime / iteration;
}

