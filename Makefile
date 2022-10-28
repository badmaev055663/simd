CC = gcc
LIBS_OMP = -fopenmp
CC_FLAGS1 = -O2 -mavx
CC_FLAGS2 = -O3 -mavx -march=native
SRC = main.c

omp1: $(SRC)
	$(CC) $(SRC) -o omp $(CC_FLAGS1) $(LIBS_OMP)

omp2: $(SRC)
	$(CC) $(SRC) -o omp $(CC_FLAGS2) $(LIBS_OMP)

asm_omp: omp
	objdump -D -M intel omp > asm_dump

clean:
	rm -f omp
	rm -f asm_dump