/*
** libgrid.c
** 
** Made by (Eugen Wintersberger)
** Login   <eugen@ohm.lan>
** 
** Started on  Tue Nov 21 21:30:41 2006 Eugen Wintersberger
** Last update Sun May 12 01:17:25 2002 Speed Blue
*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "libgrid.h"


double getmax(double *list,int nofp)
{
  double maxval=0.0;
  int i;

  for(i=0;i<nofp;i++)
    {
      if(list[i]>=maxval)
	{
	  maxval = list[i];
	}
    }

  return(maxval);
}

double getmin(double *list,int nofp)
{
  double minval = 0.0;
  int i;

  for(i=0;i<nofp;i++)
    {
      if(list[i]<=minval)
	{
	  minval = list[i];
	}
    }

  return(minval);
}



int gridder1d(double *xaxis,double *data,int nofdp,double *xgrid,double *datagrid,int nofpg)
{
  int i=0;
  double dx;
  double xmin,xmax;
  double *data_norm;
  double x,delta,b;
  int x_index;
  

  /*calculate minimum and maximum of the axis, stepsize and so on */
  xmin = getmin(xaxis,nofdp);
  xmax = getmax(xaxis,nofdp);
  dx = abs(xmax-xmin)/(double)(nofpg-1);

  /*allocate memory*/
  xgrid = malloc(nofpg*sizeof(double));
  datagrid = malloc(nofpg*sizeof(double));
  if((xgrid==NULL)||(datagrid==NULL))
    {
      printf("error allocating memeory for grid buffer\n");
      return(-1);
    }

  data_norm = malloc(nofpg*sizeof(double));

  /*build the axis and initialize the memory*/
  for(i=0;i<nofpg;i++)
    {
      data_norm[i] = 0.0;
      datagrid[i] = 0.0;
      xgrid[i] = xmin+(double)(i)*dx;
    }

  /*main loop to grid the data*/
  for(i=0;i<nofpg;i++)
    {
      x = xaxis[i];
      
      /*check if the point is within the border of the grid*/
      if((x>=xmin)&&(x<=xmax))
	{
	  b = floor((x-xmin)/dx);
	  delta = x-xmin-b*dx;
            
	  if((delta==0.0)||(delta <= dx/2.0))
	    {
	      x_index = (int)(b);
	    }
	  else
	    {
	      x_index = (int)(b)+1;
	    }

            datagrid[x_index] = datagrid[x_index] + data[i];
            data_norm[x_index] = data_norm[x_index]+1;
	}
    }

  /*calculate normalization*/
  for(i=0;i<nofpg;i++)
    {
      if(data_norm[i]!=0.0)
	{
	  datagrid[i] = datagrid[i]/(data_norm[i]-1.0);
	}
    }
  
  
  /*free all temporary used memory*/
  free(data_norm);
  
}

int gridder2d(double *xaxis,double *yaxis,double *data,int nofdp,
	      double *xgrid,double *ygrid,double *datagrid,int nofgpx,int nofgpy)
{
  double dx,dy,xmin,xmax,ymin,ymax;
  double *data_norm;
  double bx,by,x,y,delta_x,delta_y;
  int x_index,y_index;
  int i,ntot,j;

  //determine the total number of grid points
  ntot = nofgpx*nofgpy;

  xmin = getmin(xaxis,nofdp);
  xmax = getmax(xaxis,nofdp);
  ymin = getmin(yaxis,nofdp);
  ymax = getmin(yaxis,nofdp);
  dx = abs(xmax-xmin)/(double)(nofgpx-1.0);
  dy = abs(ymax-ymin)/(double)(nofgpy-1.0);

  /*allocate memory*/
  xgrid = malloc(nofgpx*nofgpy*sizeof(double));
  ygrid = malloc(nofgpy*nofgpx*sizeof(double));
  datagrid = malloc(nofgpx*nofgpy*sizeof(double));
  data_norm = malloc(nofgpx*nofgpy*sizeof(double));
  if((xgrid==NULL)||(ygrid==NULL)||(datagrid==NULL)||(data_norm==NULL))
    {
      printf("error allocating memeory for grid data\n");
      return(-1);
    }

  /*initialize the data and generate the axes*/
  for(i=0;i<ntot;i++)
    {
      datagrid[i] = 0.0; 
      data_norm[i] = 0.0;
    }
  for(i=0;i<nofgpx;i++)
    {
      xgrid[i] = xmin+(double)(i)*dx;
    }
  for(i=0;i<nofgpy;i++)
    {
      ygrid[i] = ymin+(double)(i)*dy;
    }


  for(i=0;i<nofdp;i++)
    {
      x = xaxis[i];
      if((x>=xmin)&&(x<=xmax))
	{
	  bx = floor((x-xmin)/dx);
	  delta_x = x-xmin-bx*dx;

	  if((delta_x == 0.0)||(delta_x <= 0.5*dx))
	    {
	      x_index = (int)(bx);
	    }
	  else
	    {
	      x_index = (int)(bx)+1;
	    }
	}
      else
	{
	  continue;
	}

      y = yaxis[i];

      if((y>=ymin)||(y<=ymax))
	{
	  //if y is within the grid range
	  by = floor((y-ymin)/dy);
	  delta_y = y-ymin - by*dy;
        
	  //determine the y-index of the point
	  if((delta_y == 0.0) ||( delta_y <= dy/2.0))
	    {
	      y_index = (int)(by);
	    }
	  else
	    {
	      y_index = (int)(by)+1;
	    }

	  datagrid[y_index,x_index] = datagrid[y_index,x_index] + data[i];
	  data_norm[y_index,x_index] = data_norm[y_index,x_index]+1.0;
	}
      else
	{
	  continue;
	}
      
    }

  /*caculate the normalization of the gridded data*/
  for(i=0;i<nofgpy;i++)
    {
      for(j=0;j<nofgpx;j++)
	{
	  if(data_norm[i,j]!=0.0)
	    {	      
                datagrid[i,j] = datagrid[i,j]/(data_norm[i,j]-1.0);
	    }
	}
    }
}
