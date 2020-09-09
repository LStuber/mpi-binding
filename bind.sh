#!/bin/bash

# This is a binding script that will restrict number of cores and GPUS to each MPI.
# Usage:  mpirun -n 10 ./bind.sh ./my_app

# Use SLURM variables if detected
if [[ -z $OMPI_COMM_WORLD_LOCAL_RANK ]]; then
   if [[ -z $PMI_RANK ]]; then
      export OMPI_COMM_WORLD_RANK=$SLURM_PROCID
      export OMPI_COMM_WORLD_LOCAL_RANK=$SLURM_LOCALID
      export OMPI_COMM_WORLD_SIZE=$SLURM_NTASKS
      export OMPI_COMM_WORLD_LOCAL_SIZE=$((OMPI_COMM_WORLD_SIZE/SLURM_NNODES))
   else
      export OMPI_COMM_WORLD_RANK=$PMI_RANK
      export OMPI_COMM_WORLD_LOCAL_RANK=$PMI_RANK
      export OMPI_COMM_WORLD_SIZE=$SLURM_NTASKS
      export OMPI_COMM_WORLD_LOCAL_SIZE=$((OMPI_COMM_WORLD_SIZE/SLURM_NNODES))
   fi
fi

if [[ -z $OMPI_COMM_WORLD_LOCAL_SIZE ]]; then
   echo "Error OMPI_COMM_WORLD_LOCAL_SIZE not defined. This script only works with Slurm or OpenMPI."
   exit -1
fi

if [[ -z $OMPI_COMM_WORLD_LOCAL_RANK ]]; then
   echo "Error: OMPI_COMM_WORLD_LOCAL_RANK not defined"
   exit -1
fi

#hardware
ncores_per_socket=$(lscpu | grep "Core(s)" | awk '{print $4}')
nsockets=$(lscpu | grep Sock | awk '{print $2}')

if [[ -z $NGPUS ]]; then
   NGPUS=$(nvidia-smi --query-gpu=count --format=csv -i 0 | head -2 | tail -1)
fi
if [[ -n $NCPU_SOCKETS ]]; then
   nsockets=$NCPU_SOCKETS
fi

# calculate cores/sockets per mpi
nmpi_per_socket=$(((OMPI_COMM_WORLD_LOCAL_SIZE + nsockets - 1)/ nsockets))
if [[ $nmpi_per_socket == 0 ]]; then
   nmpi_per_socket=1
fi
isocket=$((OMPI_COMM_WORLD_LOCAL_RANK / nmpi_per_socket))

if [[ -z $NCORES_PER_MPI ]];then
   if [[ -z $OMP_NUM_THREADS ]]; then
      ncores_per_mpi=$((ncores_per_socket/nmpi_per_socket))
   else
      ncores_per_mpi=$OMP_NUM_THREADS
   fi
else
   ncores_per_mpi=$NCORES_PER_MPI
fi

if [[ $LS_DEBUG == 1 ]]; then
   echo "Debug ncores_per_socket=$ncores_per_socket nsockets=$nsockets NCORES_PER_MPI=$NCORES_PER_MPI nmpi_per_socket=$nmpi_per_socket OMP_NUM_THREADS=$OMP_NUM_THREADS ncores_per_mpi=$ncores_per_mpi"
fi

if [[ -n $OMP_NUM_THREADS ]] && [[ -z $OMP_PROC_BIND ]]; then
   export OMP_PROC_BIND=true
fi


#establish list of cores
intrasocket_rank=$((OMPI_COMM_WORLD_LOCAL_RANK % nmpi_per_socket))
first=$((isocket*ncores_per_socket + intrasocket_rank*ncores_per_mpi))
cores=$first
for i in $(seq 1 $((ncores_per_mpi-1))); do
   cores=$cores,$((first+i))
done

if [[ $HYPERTHREAD == 1 ]]; then
   hyperthread_start=$((ncores_per_socket*nsockets))
   for c in $(echo $cores | tr ',' ' '); do
      cores=$cores,$((c+hyperthread_start))
   done
fi

# GPU binding
if [[ $DISABLE_GPU_BINDING != 1 ]]; then
   export ACC_DEVICE_NUM=$OMPI_COMM_WORLD_LOCAL_RANK
   export ACC_DEVICE_NUM=0
   export CUDA_VISIBLE_DEVICES=$(((NGPUS/nsockets )*isocket + intrasocket_rank))
   if [[ $GPU_BINDING_ALL_VISIBLE == 1 ]]; then
      export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
      CUDA_VISIBLE_DEVICES=$(echo $CUDA_VISIBLE_DEVICES | sed "s/$OMPI_COMM_WORLD_LOCAL_RANK,//g")
      CUDA_VISIBLE_DEVICES=$(echo $CUDA_VISIBLE_DEVICES | sed "s/,$OMPI_COMM_WORLD_LOCAL_RANK//g")
      CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK,$CUDA_VISIBLE_DEVICES
   fi
fi
if [[ -n $MPI_PER_GPU ]]; then
export LS_MPS=$MPI_PER_GPU
fi
if [[ -n $LS_MPS ]] && [[ $LS_MPS -gt 0 ]]; then
   export CUDA_VISIBLE_DEVICES=$((OMPI_COMM_WORLD_LOCAL_RANK/LS_MPS))
fi
if [[ -n $CVD ]]; then
   export CUDA_VISIBLE_DEVICES=${CVD:$OMPI_COMM_WORLD_LOCAL_RANK:1}
fi

if [[ -n $FORCE_UCX_NET_DEVICES ]]; then
   export UCX_NET_DEVICES="$FORCE_UCX_NET_DEVICES"
fi


echo "BINDER: Rank $OMPI_COMM_WORLD_RANK host $(hostname) GPU $CUDA_VISIBLE_DEVICES cores $cores $OMPI_MCA_btl_openib_if_include $UCX_NET_DEVICES"
taskset -c $cores $@
