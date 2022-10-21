CC = gcc
LIBS_OMP = -fopenmp
CC_FLAGS = -O2
SRC = main.c 

omp: $(SRC)
	$(CC) $(SRC) -o omp $(CC_FLAGS) $(LIBS_OMP)

intrin: $(SRC)
	$(CC) $(SRC) -o intrin $(CC_FLAGS)

clean:
	rm -f omp
	rm -f intrin