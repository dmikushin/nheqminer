OBJECTS = equihash_avx1.o equihash_avx2.o

all: $(OBJECTS)

equihash_avx1.o: equihash_avx1.asm
	fasm -m 1280000 $<

equihash_avx2.o: equihash_avx2.asm
	fasm -m 1280000 $<

clean:
	rm -rf $(OBJECTS)
