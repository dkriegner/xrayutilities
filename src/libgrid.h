/*
** libgrid.h
** 
** Made by Eugen Wintersberger
** Login   <eugen@ohm.lan>
** 
** Started on  Tue Nov 21 21:24:20 2006 Eugen Wintersberger
** Last update Tue Nov 21 21:24:20 2006 Eugen Wintersberger
*/

#ifndef   	LIBGRID_H_
# define   	LIBGRID_H_

int gridder1d(double *xaxis,double *data,int nofdp,double *xgrid,double *datagrid,int nofpg);
int gridder2d(double *xaxis,double *yaxis,double *data,int nofdp,
	      double *xgrid,double *ygrid,double *datagrid,int nofgpx,int nofgpy);
/*int gridder3d(double *xaxis,double *yaxis,double *zdata,double *data,int nofdp,
	      double *xgrid,double *ygrid,double *zgrid,double *datagrid,
	      int nofpgx,int nofpgy,int nofpgz);*/

double getmax(double *list,int nofp);
double getmin(double *list,int nofp);


#endif 	    /* !LIBGRID_H_ */
