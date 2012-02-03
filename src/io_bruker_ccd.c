/*
 * This file is part of xrutils.
 * 
 * xrutils is free software; you can redistribute it and/or modify 
 * it under the terms of the GNU General Public License as published by 
 * the Free Software Foundation; either version 2 of the License, or 
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, see <http://www.gnu.org/licenses/>.
 * 
 * Copyright (C) 2009 Eugen Wintersberger <eugen.wintersberger@desy.de>
*/

#include<stdio.h>
#include<stdlib.h>

#include"bruker_ccd.h"

/*create a new frame structure without predefined data content*/
Bruker_CCD_Frame *create_frame(int nofxchan,int nofychan)
{
  Bruker_CCD_Frame *frame;
  
  frame->nof_x_channels = nofxchan;
  frame->nof_y_channels = nofychan;
  frame->data = malloc(sizeof(int)*nofxchan*nofychan);
  if(frame->data == NULL)
    {
	  printf("create_frame: Error allocating frame memory!\n");
	  return(NULL);	
    }
  
  return(frame);
}

/*create a new frame structure and initialize the data section 
  with an user given value*/
Bruker_CCD_Frame *create_frame_fill(int nofxchan,int nofychan,int fill_value)
{
  int i;
  Bruker_CCD_Frame *frame;
  
  frame = create_frame(nofxchan,nofychan);
  if(frame==NULL) return(NULL);
  
  /*fill the frame with the requested values*/
  for(i=0;i<nofxchan*nofychan;i++)
    {
      frame->data[i] = fill_value;
    }
  
  return(frame);
	
}

/*destroy a frame structure*/
int del_frame(Bruker_CCD_Frame *frame)
{
  free(frame->data);
}

/*select a region of interest from the frame structure and return 
  a new frame containing only the data described by the ROI*/
Bruker_CCD_Frame *get_ROI(Bruker_CCD_Frame *frame,Bruker_CCD_ROI *roi)
{
  Bruker_CCD_Frame *new_frame;
  
  new_frame->nof_x_channels = (roi->upper_x_channel - roi->lower_x_channel);
  new_frame->nof_y_channels = (roi->upper_x_channel - roi->lower_x_channel);
  
  new_frame->data = malloc(sizeof(int)*new_frame->nof_x_channels*new_frame->nof_y_channels);
  if(new_frame->data==NULL)
    {
      printf("get_ROI: error allocating frame memory!\n");
      return(NULL);	
    }
  
  return(new_frame);
}

/***************************************************************************
Bruker_CCD_Frame *merge_frames(int nframes,Bruker_CCD_Frame *frames):
	Merging several frames into one single frame. 
	
	input arguments:
	nframes(int) ............... number of frames to merge
	frames ..................... pointer to a list of frames which will 
	                             be merged.
	                             
	return value:
	A single Bruker CCD frame. 
****************************************************************************/
Bruker_CCD_Frame *merge_frames(int nframes,Bruker_CCD_Frame *frames)
{
  int totnofp = 0; /*total number of points*/
  int nofcols = 0; /*number of columns*/
  int i,j;         /*index counter for looping over the data*/	
  Bruker_CCD_Frame *new_frame;
  
  /*calculate the total number of points per frame*/
  totnofp = frames->nof_x_channels*frames->nof_y_channels;
  
  /*create a new frame initialized to zero*/
  new_frame = create_frame_fill(frames->nof_x_channels,frames->nof_y_channels,0);
  
  /*loop over all frames*/
  for(i=0;i<nframes;i++)
    {      
      /*summ over all frames*/
      for(j=0;j<totnofp;j++)
	{
	  new_frame->data[j] = new_frame->data[j]+(frames+j)->data[j];
	}
    }
  
  return(new_frame);
}

/*reads a frame from its file*/
Bruker_CCD_Frame *read_frame(char *full_file_name)
{
  FILE *fp; /*file pointer*/
  Bruker_CCD_Frame *data_frame; /*frame holding the data*/
  Bruker_CCD_Header *header;	
  int nofxchans;
  int nofychans;
  int nof_underflow;      /*number of pixel underflows*/
  int nof_1byte_overflow; /*number of 1byte pixel overflows*/
  int nof_2byte_overflow; /*number of 2byte pixel overflows*/
  int nof_header_blocks = 0; /*total number of header blocks*/
  int nofb_per_pixel = 0;    /*number of bytes per pixel*/
  unsigned char data_buffer;    /*data buffer for 1byte per pixel reading*/
  unsigned short data_buffer_2; /*data buffer for 2byte per pixel reading*/
  int i;
  
  /*ope file*/
  fp = fopen(full_file_name,"rb");
  if (fp==NULL)
    {
      printf("error opening file %s!\n",full_file_name);
      return(NULL);
    }
  
  /*read the header block from the data file*/
  header = malloc(sizeof(Bruker_CCD_Header));
  fread(header,sizeof(Bruker_CCD_Header),1,fp);
  nof_header_blocks = atoi(header->entry[2].value);
  nofb_per_pixel = atoi(header->entry[39].value);

  fseek(fp,(long)(nofb_per_pixel*512)*sizeof(char),0);
#ifdef DEBUG
  printf("read header structure\n");
  printf("header size %i\n",sizeof(Bruker_CCD_Header));
  printf("calculated header size %i\n",nof_header_blocks*HEADER_BLK_SIZE);
  printf("number of bytes per pixes: %i\n",nofb_per_pixel);
#endif

  nofychans = atoi(header->entry[40].value);
  nofxchans = atoi(header->entry[41].value);
#ifdef DEBUG
  printf("nof x channels: %i\n",nofxchans);
  printf("nof y channels: %i\n",nofychans);
#endif
  sscanf(header->entry[20].value,"%i %i %i",&nof_underflow,&nof_1byte_overflow,&nof_2byte_overflow);

#ifdef DEBUG
  printf("number of underflows: %i\n",nof_underflow);
  printf("number of 1byte overflows: %i\n",nof_1byte_overflow);
  printf("number of 2byte overflows: %i\n",nof_2byte_overflow);
#endif

  /*create a new data frame*/
  data_frame = (Bruker_CCD_Frame *)create_frame(nofxchans,nofychans);
#ifdef DEBUG
  printf("data frame created\n");
#endif
  
  /*read the data from the file*/
  for(i=0;i<nofxchans*nofychans-100;i++)
    {
      fread(&data_buffer,1,1,fp);	
      data_frame->data[i] = (int)(data_buffer); 

      if(i>nofxchans*nofychans-40)
	{
	  printf("value: %i\n",(int)(data_buffer));
	}
    }
#ifdef DEBUG
  printf("read data from file\n");
#endif
  /*free the header structure of the file since it is no longer
    needed*/
  free(header);  

  /*after everything is done - close the file and free all memory*/
  fclose(fp);

  return(data_frame);
}



