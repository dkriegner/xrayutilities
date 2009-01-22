##
## Makefile
##  
## Made by Eugen Wintersberger
## Login   <eugenramses.lan>
##
## Started on  Tue Aug 21 21:05:27 2007 Eugen Wintersberger
## Last update Tue Aug 21 21:05:27 2007 Eugen Wintersberger
## 
libsrc = gridder.c ang2q.c
libobj = $(patsubst %.c,%.o,$(libsrc))
lib = libxrayutils.so

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

