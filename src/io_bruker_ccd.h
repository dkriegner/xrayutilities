#include<stdlib.h>
#include<stdio.h>

#define KEY_LINE_LENGTH 8
#define BASE_LINE_LENGTH 72
#define HEADER_BLK_SIZE 512
#define NOF_HEADER_LINES 94

/* 
 Bruker stores the header entries in list of key - value pairs 
 where the first 8 bytes are the key and the second 72 hold the value
 of the entry. This datatype describes a single entry.
*/
typedef struct structBruker_CCD_Header_Entry{
  char key[KEY_LINE_LENGTH];
  char value[BASE_LINE_LENGTH];	
}Bruker_CCD_Header_Entry;

/*data structure describing the entire Bruker header*/
typedef struct structBruker_CCD_Header{
  Bruker_CCD_Header_Entry entry[NOF_HEADER_LINES];
}Bruker_CCD_Header;

/*
 The data is stored in frames. We need a datatype that 
 describes a single frame. Overflow and underflow 
 correction act on this data structure.
*/

typedef struct structBruker_CCD_Frame{
  int nof_x_channels;
  int nof_y_channels;
  int *data;
}Bruker_CCD_Frame;

/*type for setting a roi*/
typedef struct structBruker_CCD_ROI{
  int lower_x_channel;
  int upper_x_channel;
  int lower_y_channel;
  int upper_y_channel;
}Bruker_CCD_ROI;

/*
 Prototypes for linking external applications against the API
*/

Bruker_CCD_Frame *create_frame(int,int);
Bruker_CCD_Frame *create_frame_fill(int ,int ,int );
int del_frame(Bruker_CCD_Frame *);
Bruker_CCD_Frame *get_ROI(Bruker_CCD_Frame *,Bruker_CCD_ROI *);
Bruker_CCD_Frame *merge_frames(int,Bruker_CCD_Frame *);
Bruker_CCD_Frame *read_frame(char *);
Bruker_CCD_Frame *read_frames(char *,int,int,int,int); 
Bruker_CCD_Header *read_frame_header(FILE *);



/*
 The key value pairs are stored in the header are not really usefull, 
 therefore, the information is converted to usefull datatypes 
 and finally stored in a table.
 
*/
/*
typedef struct structBruker_CCD_Header{
	int   FORMAT;
	int   VERSION;
	int   HDRBLKS;
	char  TYPE[BASE_LINE_LENGTH];
	char  SITE[BASE_LINE_LENGTH];
	char  MODEL[BASE_LINE_LENGTH];
	char  USER[BASE_LINE_LENGTH];
	char  SAMPLE[BASE_LINE_LENGTH];
	char  SETNAME[BASE_LINE_LENGTH];
	int   RUN;
	int   SAMPNUM;
	char  TITLE[BASE_LINE_LENGTH*8];
	int   NCOUNTS[2];
	int   NOVERFL[3];
	int   MINIMUM;
	int   MAXIMUM;
	int   NONTIME;
	int   NLATE;
	int   FILENAM[BASE_LINE_LENGTH];
	int   CREATED[BASE_LINE_LENGTH];
	float CUMULAT[BASE_LINE_LENGTH];
	float ELAPSDR;
	float ELAPSDA;
	int   OSCILLA;
	int   NSTEPS;
	float RANGE;
	float START;
	float INCREME;
	int   NUMBER;
	int   NFRAMES;
	int   ANGLES[4];
	int   NOVER64[3];
	int   NPIXELB[2];;
	int   NROWS;
	int   NCOLS;
	int   WORDORD;
	int   LONGORD;
	char  TARGET[BASE_LINE_LENGTH];
	float SOURCEK;
	float SOURCEM;
	char  FILTER[BASE_LINE_LENGTH];
	float CELL[2][6];
	float MATRIX[2,9];
	int   LOWTEMP[3];
	float ZOOM[3];
	float CENTER[4];
	float DISTANC[2]; //has to be observed
	int   TRAILER;
	char  COMPRES[BASE_LINE_LENGTH];
	char  LINEAR[BASE_LINE_LENGTH];
	float PHD[2];
	float PREAMP[2]; //has to be observed
	char  CORRECT[BASE_LINE_LENGTH];
	char  WARPFIL[BASE_LINE_LENGTH];
	float WAVELEN[4];
	float MAXXY[2];
	int   AXIS;
	float ENDING[4];
	float DETPAR[2][6];
	char  LUT[BASE_LINE_LENGTH];
	float DISPLIM[2];
	char  PROGRAM[BASE_LINE_LENGTH];
	int   ROTATE;
	char  BITMASK[BASE_LINE_LENGTH];
	int   OCTMASK[2][8];
	float ESDCELL[2][5];
	char  DETTYPE[BASE_LINE_LENGTH];
	int   NEXP[5];
	float CCDPARM[5];
	char  CHEM[BASE_LINE_LENGTH];
	char  MORPH[BASE_LINE_LENGTH];
	char  CCOLOR[BASE_LINE_LENGTH];
	char  CSIZE[BASE_LINE_LENGTH];
	char  DNSMET[BASE_LINE_LENGTH];
	char  DARK[BASE_LINE_LENGTH];
	float AUTORNG[5];
	float ZEROADJ[4];
	float XTRANS[3];
	float HKL_XY[5];
	float AXES2[4];
	float ENDINGS2[4];
	float FILTER2[2];
}Bruker_CCD_Header; */
