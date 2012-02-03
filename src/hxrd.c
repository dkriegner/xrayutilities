/*
 * This file is part of xrayutilities.
 * 
 * xrayutilities is free software; you can redistribute it and/or modify 
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

/*
** hxrd.c
** 
** Made by (Eugen Wintersberger)
** Login   <eugen@ramses.lan>
** 
** Started on  Tue Aug 21 19:58:48 2007 Eugen Wintersberger
** Last update Sun May 12 01:17:25 2002 Speed Blue
*/

#include<stdio.h>
#include<stdlib.h>
#include<math.h>

#include "xrutils.h"
#include "constants.h"

int hxrd_ang2q(double *omega,double *tth,double lambda,int n,double *qx,double *qz,int aunit)
{
  double k;
  double p;
  double angcorr;
  int i;

  k=2.0*PI/lambda;

  switch(aunit){
  case ANG_RAD_UNIT:
    angcorr = 1.0;
    break;
  case ANG_DEG_UNIT:
    angcorr = DEG2RAD;
    break;
  default:
    angcorr = DEG2RAD;
  }
  
  for(i=0;i<n;i++){
    p = 2.0*k*sin(angcorr*tth[i]/2.0);
    qx[i] = p*sin(angcorr*(omega[i]-tth[i]/2.0));
    qz[i] = p*cos(angcorr*(omega[i]-tth[i]/2.0));     
  }
}

int gid_ang2q(double *ai,double *af,double *tth,int n,double lambda,double *qx,double *qy,double *qz,int aunit)
{
  double k;
  double angcorr;
  int i;
  /*local angular variables - we use them to do angle correction only once*/
  double ai_loc;
  double af_loc;
  double tth_loc;

  k = 2.0*PI/lambda;

  switch(aunit){
  case ANG_RAD_UNIT:
    angcorr = 1.0;
    break;
  case ANG_DEG_UNIT:
    angcorr = DEG2RAD;
    break;
  default:
    angcorr = DEG2RAD;
  }

  for(i=0;i<n;i++){
    /*calculate local angular variables*/
    ai_loc = angcorr*ai[i];
    af_loc = angcorr*af[i];
    tth_loc = angcorr*tth[i];

    /*calculate the q-space variables*/
    qx[i] = k*(cos(ai_loc)-cos(af_loc)*cos(tth_loc));
    qy[i] = k*cos(af_loc)*sin(tth_loc);
    qz[i] = k*(sin(ai_loc)+sin(af_loc));
    
  }
}







