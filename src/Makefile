libsrc = gridder.c ang2q.c
libobj = $(patsubst %.c,%.o,$(libsrc))
lib = libxutils.so

CFLAGS=-fPIC -O2 -Wall
LDFALGS = -lpthread -lm
CC = gcc

all: $(lib)

$(lib):	$(libobj)
	$(CC) -shared -o $(lib) $(libobj) $(LDFLAGS)

$(libobj):%.o:%.c
	$(CC) $(CFLAGS) -o $@ -c $< 

clean:
	rm *.o
	rm $(lib)

