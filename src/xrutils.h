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
** xrutils.h
** 
** Made by Eugen Wintersberger
** Login   <eugen@ramses.lan>
** 
** Started on  Tue Aug 21 19:50:08 2007 Eugen Wintersberger
** Last update Tue Aug 21 19:50:08 2007 Eugen Wintersberger
*/

#ifndef   	XRUTILS_H_
# define   	XRUTILS_H_

/*functions for q-space conversion*/
int hxrd_ang2q(double *,double *,double ,int,double *,double *,int);
int gid_ang2q(double *,double *,double *,int ,double ,double *,double *,double *,int);

#endif 	    /* !XRUTILS_H_ */




