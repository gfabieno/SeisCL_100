/*------------------------------------------------------------------------
 * Copyright (C) 2016 For the list of authors, see file AUTHORS.
 *
 * This file is part of SeisCL.
 *
 * SeisCL is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.0 of the License only.
 *
 * SeisCL is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with SeisCL. See file COPYING and/or
 * <http://www.gnu.org/licenses/gpl-3.0.html>.
 --------------------------------------------------------------------------*/

/*Adjoint sources */

/*Define useful macros to be able to write a matrix formulation in 2D with OpenCl */
#if ND==3
#define sxx(z,y,x) sxx[(x)*NY*NZ+(y)*NZ+(z)]
#define syy(z,y,x) syy[(x)*NY*NZ+(y)*NZ+(z)]
#define szz(z,y,x) szz[(x)*NY*NZ+(y)*NZ+(z)]
#define sxy(z,y,x) sxy[(x)*NY*NZ+(y)*NZ+(z)]
#define syz(z,y,x) syz[(x)*NY*NZ+(y)*NZ+(z)]
#define sxz(z,y,x) sxz[(x)*NY*NZ+(y)*NZ+(z)]

#define sxx_buf1(z,y,x) sxx_buf1[(x)*(NY-2*fdoh)*(NZ-2*fdoh)+(y)*(NZ-2*fdoh)+(z)]
#define syy_buf1(z,y,x) syy_buf1[(x)*(NY-2*fdoh)*(NZ-2*fdoh)+(y)*(NZ-2*fdoh)+(z)]
#define szz_buf1(z,y,x) szz_buf1[(x)*(NY-2*fdoh)*(NZ-2*fdoh)+(y)*(NZ-2*fdoh)+(z)]
#define sxy_buf1(z,y,x) sxy_buf1[(x)*(NY-2*fdoh)*(NZ-2*fdoh)+(y)*(NZ-2*fdoh)+(z)]
#define syz_buf1(z,y,x) syz_buf1[(x)*(NY-2*fdoh)*(NZ-2*fdoh)+(y)*(NZ-2*fdoh)+(z)]
#define sxz_buf1(z,y,x) sxz_buf1[(x)*(NY-2*fdoh)*(NZ-2*fdoh)+(y)*(NZ-2*fdoh)+(z)]

#define sxx_buf2(z,y,x) sxx_buf2[(x)*(NY-2*fdoh)*(NZ-2*fdoh)+(y)*(NZ-2*fdoh)+(z)]
#define syy_buf2(z,y,x) syy_buf2[(x)*(NY-2*fdoh)*(NZ-2*fdoh)+(y)*(NZ-2*fdoh)+(z)]
#define szz_buf2(z,y,x) szz_buf2[(x)*(NY-2*fdoh)*(NZ-2*fdoh)+(y)*(NZ-2*fdoh)+(z)]
#define sxy_buf2(z,y,x) sxy_buf2[(x)*(NY-2*fdoh)*(NZ-2*fdoh)+(y)*(NZ-2*fdoh)+(z)]
#define syz_buf2(z,y,x) syz_buf2[(x)*(NY-2*fdoh)*(NZ-2*fdoh)+(y)*(NZ-2*fdoh)+(z)]
#define sxz_buf2(z,y,x) sxz_buf2[(x)*(NY-2*fdoh)*(NZ-2*fdoh)+(y)*(NZ-2*fdoh)+(z)]
#endif

#if ND==2 || ND==21

#define sxx(z,y,x) sxx[(x)*NZ+(z)]
#define syy(z,y,x) syy[(x)*NZ+(z)]
#define szz(z,y,x) szz[(x)*NZ+(z)]
#define sxy(z,y,x) sxy[(x)*NZ+(z)]
#define syz(z,y,x) syz[(x)*NZ+(z)]
#define sxz(z,y,x) sxz[(x)*NZ+(z)]

#define sxx_buf1(z,y,x) sxx_buf1[(x)*(NZ-2*fdoh)+(z)]
#define syy_buf1(z,y,x) syy_buf1[(x)*(NZ-2*fdoh)+(z)]
#define szz_buf1(z,y,x) szz_buf1[(x)*(NZ-2*fdoh)+(z)]
#define sxy_buf1(z,y,x) sxy_buf1[(x)*(NZ-2*fdoh)+(z)]
#define syz_buf1(z,y,x) syz_buf1[(x)*(NZ-2*fdoh)+(z)]
#define sxz_buf1(z,y,x) sxz_buf1[(x)*(NZ-2*fdoh)+(z)]

#define sxx_buf2(z,y,x) sxx_buf2[(x)*(NZ-2*fdoh)+(z)]
#define syy_buf2(z,y,x) syy_buf2[(x)*(NZ-2*fdoh)+(z)]
#define szz_buf2(z,y,x) szz_buf2[(x)*(NZ-2*fdoh)+(z)]
#define sxy_buf2(z,y,x) sxy_buf2[(x)*(NZ-2*fdoh)+(z)]
#define syz_buf2(z,y,x) syz_buf2[(x)*(NZ-2*fdoh)+(z)]
#define sxz_buf2(z,y,x) sxz_buf2[(x)*(NZ-2*fdoh)+(z)]

#endif

__kernel void fill_transfer_buff_s_out(__global float *sxx,        __global float *syy,          __global float *szz,
                                       __global float *sxy,        __global float *syz,          __global float *sxz,
                                       __global float *sxx_buf1,        __global float *syy_buf1,          __global float *szz_buf1,
                                       __global float *sxy_buf1,        __global float *syz_buf1,          __global float *sxz_buf1,
                                       __global float *sxx_buf2,        __global float *syy_buf2,          __global float *szz_buf2,
                                       __global float *sxy_buf2,        __global float *syz_buf2,          __global float *sxz_buf2)
{
#if ND==3
    // If we use local memory
#if local_off==0
    
    int gidz = get_global_id(0)+fdoh;
    int gidy = get_global_id(1)+fdoh;
    int gidx = get_global_id(2);
    
    // If local memory is turned off
#elif local_off==1
    
    int gid = get_global_id(0);
    int gidz = gid%glsizez+fdoh;
    int gidy = (gid/glsizez)%glsizey+fdoh;
    int gidx = gid/(glsizez*glsizey);
    
#endif
    
#else
    
    // If we use local memory
#if local_off==0
    int gidz = get_global_id(0)+fdoh;
    int gidx = get_global_id(1);
    int gidy = 0;
    
    // If local memory is turned off
#elif local_off==1
    int gid = get_global_id(0);
    int gidz = gid%glsizez+fdoh;
    int gidx = (gid/glsizez);
    int gidy = 0;
#endif
#endif


#if !(dev==0 & MYLOCALID==0)
    sxx_buf1(gidz-fdoh,gidy,gidx)=sxx(gidz,gidy,gidx+fdoh);
    szz_buf1(gidz-fdoh,gidy,gidx)=szz(gidz,gidy,gidx+fdoh);
    sxz_buf1(gidz-fdoh,gidy,gidx)=sxz(gidz,gidy,gidx+fdoh);
#endif
#if !(dev==num_devices-1 & MYLOCALID==NLOCALP-1)
    sxx_buf2(gidz-fdoh,gidy,gidx)=sxx(gidz,gidy,gidx+NX-2*fdoh);
    szz_buf2(gidz-fdoh,gidy,gidx)=szz(gidz,gidy,gidx+NX-2*fdoh);
    sxz_buf2(gidz-fdoh,gidy,gidx)=sxz(gidz,gidy,gidx+NX-2*fdoh);
#endif
    
    
#if ND==3
#if !(dev==0 & MYLOCALID==0)
    syy_buf1(gidz-fdoh,gidy,gidx)=syy(gidz,gidy,gidx+fdoh);
    sxy_buf1(gidz-fdoh,gidy,gidx)=sxy(gidz,gidy,gidx+fdoh);
    syz_buf1(gidz-fdoh,gidy,gidx)=syz(gidz,gidy,gidx+fdoh);
#endif
#if !(dev==num_devices-1 & MYLOCALID==NLOCALP-1)
    syy_buf2(gidz-fdoh,gidy,gidx)=syy(gidz,gidy,gidx+NX-2*fdoh);
    sxy_buf2(gidz-fdoh,gidy,gidx)=sxy(gidz,gidy,gidx+NX-2*fdoh);
    syz_buf2(gidz-fdoh,gidy,gidx)=syz(gidz,gidy,gidx+NX-2*fdoh);
#endif
#endif
    

 
}

__kernel void fill_transfer_buff_s_in(__global float *sxx,        __global float *syy,          __global float *szz,
                                       __global float *sxy,        __global float *syz,          __global float *sxz,
                                       __global float *sxx_buf1,        __global float *syy_buf1,          __global float *szz_buf1,
                                       __global float *sxy_buf1,        __global float *syz_buf1,          __global float *sxz_buf1,
                                       __global float *sxx_buf2,        __global float *syy_buf2,          __global float *szz_buf2,
                                       __global float *sxy_buf2,        __global float *syz_buf2,          __global float *sxz_buf2)
{
#if ND==3
    // If we use local memory
#if local_off==0
    
    int gidz = get_global_id(0)+fdoh;
    int gidy = get_global_id(1)+fdoh;
    int gidx = get_global_id(2);
    
    // If local memory is turned off
#elif local_off==1
    
    int gid = get_global_id(0);
    int gidz = gid%glsizez+fdoh;
    int gidy = (gid/glsizez)%glsizey+fdoh;
    int gidx = gid/(glsizez*glsizey);
    
#endif
    
#else
    
    // If we use local memory
#if local_off==0
    int gidz = get_global_id(0)+fdoh;
    int gidx = get_global_id(1);
    int gidy = 0;
    
    // If local memory is turned off
#elif local_off==1
    int gid = get_global_id(0);
    int gidz = gid%glsizez+fdoh;
    int gidx = (gid/glsizez);
    int gidy = 0;
#endif
#endif
    
#if !(dev==0 & MYLOCALID==0)
    sxx(gidz,gidy,gidx)=sxx_buf1(gidz-fdoh,gidy,gidx);
    szz(gidz,gidy,gidx)=szz_buf1(gidz-fdoh,gidy,gidx);
    sxz(gidz,gidy,gidx)=sxz_buf1(gidz-fdoh,gidy,gidx);
#endif
#if !(dev==num_devices-1 & MYLOCALID==NLOCALP-1)
    sxx(gidz,gidy,gidx+NX-fdoh)=sxx_buf2(gidz-fdoh,gidy,gidx);
    szz(gidz,gidy,gidx+NX-fdoh)=szz_buf2(gidz-fdoh,gidy,gidx);
    sxz(gidz,gidy,gidx+NX-fdoh)=sxz_buf2(gidz-fdoh,gidy,gidx);
#endif


#if ND==3
#if !(dev==0 & MYLOCALID==0)
    syy(gidz,gidy,gidx)=syy_buf1(gidz-fdoh,gidy,gidx);
    sxy(gidz,gidy,gidx)=sxy_buf1(gidz-fdoh,gidy,gidx);
    syz(gidz,gidy,gidx)=syz_buf1(gidz-fdoh,gidy,gidx);
#endif
#if !(dev==num_devices-1 & MYLOCALID==NLOCALP-1)
    syy(gidz,gidy,gidx+NX-fdoh)=syy_buf2(gidz-fdoh,gidy,gidx);
    sxy(gidz,gidy,gidx+NX-fdoh)=sxy_buf2(gidz-fdoh,gidy,gidx);
    syz(gidz,gidy,gidx+NX-fdoh)=syz_buf2(gidz-fdoh,gidy,gidx);
#endif
#endif
    
}
