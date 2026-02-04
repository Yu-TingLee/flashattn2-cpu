#!/usr/bin/env bash
set -e

T_values=(
  256
  512
  1024 
  2048 
  4096 
  8192
)

d_values=(
  64 
  128
)

num_testsets=(
  50
)


M_bytes_values=($(python3 -c "
base = 131072
multipliers = [0.25, 0.5, 1.0, 1.25, 1.5, 2.0, 2.25, 2.5, 
               3.0, 3.25, 3.5, 4.0, 4.25, 4.5, 5.0, 5.25, 
               5.5, 6.0, 6.25, 6.5, 7.0, 7.25, 7.5, 8.0]
print(' '.join(str(int(m * base)) for m in multipliers))
"))

python data_generation.py --T 8192 --d 128 --num_testsets "$num_testsets"

for t in "${T_values[@]}"; do
  for d in "${d_values[@]}"; do
      echo "Running Python naive_attn: T=$t, d=$d"
      python naive_attn.py --T "$t" --d "$d" --num_testsets "$num_testsets"
  done
done

for t in "${T_values[@]}"; do
  for d in "${d_values[@]}"; do
    for m in "${M_bytes_values[@]}"; do
      echo "Running FA-2: T=$t, d=$d, M=$(($m/1024))KiB"
      python flash_attn2.py --T "$t" --d "$d" --M_bytes "$m" --num_testsets "$num_testsets"
    done
  done
done

for t in "${T_values[@]}"; do
  for d in "${d_values[@]}"; do
    for m in "${M_bytes_values[@]}"; do
      echo "Running JIT FA-2: T=$t, d=$d, M=$(($m/1024))KiB"
      python flash_attn2_jit.py --T "$t" --d "$d" --M_bytes "$m" --num_testsets "$num_testsets"
    done
  done
done

OPT_FLAGS=(O3)

# for opt in "${OPT_FLAGS[@]}"; do
#   g++ -std=c++17 -${opt} -march=native -I /usr/include/eigen3 -o bin/naive_attn_${opt} naive_attn.cpp
#   g++ -std=c++17 -${opt} -march=native -I /usr/include/eigen3 -o bin/flash_attn2_${opt} flash_attn2.cpp
#   g++ -std=c++17 -${opt} -march=native -I /usr/include/eigen3 -o bin/flash_attn2_profile_${opt} ./profile/flash_attn2_profile.cpp
# done

for opt in "${OPT_FLAGS[@]}"; do
  for t in "${T_values[@]}"; do
    for d in "${d_values[@]}"; do
      echo "Running C++ naive_attn_${opt}: T=$t, d=$d, opt=$opt"
      ./build/bin/naive_attn_${opt} --T "$t" --d "$d" --num_testsets "$num_testsets" --opt_flag "$opt"
    done
  done
done

for opt in "${OPT_FLAGS[@]}"; do
  for t in "${T_values[@]}"; do
    for d in "${d_values[@]}"; do
      for m in "${M_bytes_values[@]}"; do
        echo "Running C++ flash_attn2_${opt}: T=$t, d=$d, M=$(($m/1024))KiB, opt=$opt"
        ./build/bin/flash_attn2_${opt} --T "$t" --d "$d" --M_bytes "$m" --num_testsets "$num_testsets" --opt_flag "$opt"
      done
    done
  done
done

M_bytes_values_profile=($(python3 -c "
base = 131072
multipliers = [0.125, 2.0, 8.0]
print(' '.join(str(int(m * base)) for m in multipliers))
"))

T_values_profile=(
  512
  2048
  8192
)

for t in "${T_values_profile[@]}"; do
  for d in "${d_values[@]}"; do
    for m in "${M_bytes_values_profile[@]}"; do
      echo "Running FA-2 profile: T=$t, d=$d, M=$(($m/1024))KiB"
      python ./profile/flash_attn2_profile.py --T "$t" --d "$d" --M_bytes "$m" --num_testsets "$num_testsets"
    done
  done
done

for opt in "${OPT_FLAGS[@]}"; do
  for t in "${T_values_profile[@]}"; do
    for d in "${d_values[@]}"; do
      for m in "${M_bytes_values_profile[@]}"; do
        echo "Running C++ FA-2 profile_${opt}: T=$t, d=$d, M=$(($m/1024))KiB, opt=$opt"
        ./bin/flash_attn2_profile_${opt} --T "$t" --d "$d" --M_bytes "$m" --num_testsets "$num_testsets" --opt_flag "$opt"
      done
    done
  done
done

python plot.py