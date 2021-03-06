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

/*Adjoint update of the stresses in 2D SV*/


FUNDEF void update_adjs(int offcomm,
                        GLOBARG __pprec *muipkp, GLOBARG __pprec *M,
                        GLOBARG __pprec *mu,
                        GLOBARG __prec2 *sxx, GLOBARG __prec2 *sxz,
                        GLOBARG __prec2 *szz, GLOBARG __prec2 *vx,
                        GLOBARG __prec2 *vz, GLOBARG __prec2 *sxxbnd,
                        GLOBARG __prec2 *sxzbnd, GLOBARG __prec2 *szzbnd,
                        GLOBARG __prec2 *vxbnd, GLOBARG __prec2 *vzbnd,
                        GLOBARG __prec2 *sxxr,GLOBARG __prec2 *sxzr,
                        GLOBARG __prec2 *szzr, GLOBARG __prec2 *vxr,
                        GLOBARG __prec2 *vzr, GLOBARG float *taper,
                        GLOBARG float *K_x,         GLOBARG float *a_x,
                        GLOBARG float *b_x,
                        GLOBARG float *K_x_half,    GLOBARG float *a_x_half,
                        GLOBARG float *b_x_half,
                        GLOBARG float *K_z,         GLOBARG float *a_z,
                        GLOBARG float *b_z,
                        GLOBARG float *K_z_half,    GLOBARG float *a_z_half,
                        GLOBARG float *b_z_half,
                        GLOBARG __prec2 *psi_vx_x,  GLOBARG __prec2 *psi_vx_z,
                        GLOBARG __prec2 *psi_vz_x,  GLOBARG __prec2 *psi_vz_z,
                        GLOBARG __gprec *gradrho, GLOBARG __gprec *gradM,
                        GLOBARG __gprec *gradmu,  GLOBARG __gprec *Hrho,
                        GLOBARG __gprec *HM,      GLOBARG  __gprec *Hmu,
                        int res_scale, int src_scale, int par_scale, LOCARG2)
{

    //Local memory
    #ifdef __OPENCL_VERSION__
    __local __prec * lvar=lvar2;
    #else
    extern __shared__ __prec2 lvar2[];
    __prec * lvar=(__prec *)lvar2;
    #endif
    
    //Grid position
    // If we use local memory
    #if LOCAL_OFF==0
        #ifdef __OPENCL_VERSION__
        int lsizez = get_local_size(0)+2*FDOH/DIV;
        int lsizex = get_local_size(1)+2*FDOH;
        int lidz = get_local_id(0)+FDOH/DIV;
        int lidx = get_local_id(1)+FDOH;
        int gidz = get_global_id(0)+FDOH/DIV;
        int gidx = get_global_id(1)+FDOH+offcomm;
        #else
        int lsizez = blockDim.x+2*FDOH/DIV;
        int lsizex = blockDim.y+2*FDOH;
        int lidz = threadIdx.x+FDOH/DIV;
        int lidx = threadIdx.y+FDOH;
        int gidz = blockIdx.x*blockDim.x+threadIdx.x+FDOH/DIV;
        int gidx = blockIdx.y*blockDim.y+threadIdx.y+FDOH+offcomm;
        #endif

    // If local memory is turned off
    #elif LOCAL_OFF==1
        #ifdef __OPENCL_VERSION__
        int gid = get_global_id(0);
        int glsizez = (NZ-2*FDOH/DIV);
        int gidz = gid%glsizez+FDOH/DIV;
        int gidx = (gid/glsizez)+FDOH+offcomm;
        #else
        int lsizez = blockDim.x+2*FDOH/DIV;
        int lsizex = blockDim.y+2*FDOH;
        int lidz = threadIdx.x+FDOH/DIV;
        int lidx = threadIdx.y+FDOH;
        int gidz = blockIdx.x*blockDim.x+threadIdx.x+FDOH/DIV;
        int gidx = blockIdx.y*blockDim.y+threadIdx.y+FDOH+offcomm;
        #endif

    #endif

    int indp = ((gidx)-FDOH)*(NZ-2*FDOH/DIV)+((gidz)-FDOH/DIV);
    int indv = gidx*NZ+gidz;
    
    //Define private derivatives
    __cprec vx_x2;
    __cprec vx_z1;
    __cprec vz_x1;
    __cprec vz_z2;
    __cprec vxr_x2;
    __cprec vxr_z1;
    __cprec vzr_x1;
    __cprec vzr_z2;



    // If we use local memory
    #if LOCAL_OFF==0
        #define lvx lvar
        #define lvz lvar
        #define lvx2 lvar2
        #define lvz2 lvar2

        #define lvxr lvar
        #define lvzr lvar
        #define lvxr2 lvar2
        #define lvzr2 lvar2

    // If local memory is turned off
    #elif LOCAL_OFF==1

        #define lvxr vxr
        #define lvzr vzr
        #define lvx vx
        #define lvz vz
        #define lidx gidx
        #define lidz gidz

    #endif

    /* Calculation of the velocity spatial derivatives of the forward wavefield
    if backpropagation is used */
    #if BACK_PROP_TYPE==1
        {
        #if LOCAL_OFF==0
            BARRIER
            load_local_in(vx);
            load_local_haloz(vx);
            load_local_halox(vx);
            BARRIER
        #endif
            vx_x2 = Dxm(lvx2);
            vx_z1 = Dzp(lvx);

        #if LOCAL_OFF==0
            BARRIER
            load_local_in(vz);
            load_local_haloz(vz);
            load_local_halox(vz);
            BARRIER
        #endif
            vz_x1 = Dxp(lvz2);
            vz_z2 = Dzm(lvz);
            BARRIER
        }
    #endif

    // Calculation of the velocity spatial derivatives of the adjoint wavefield
    {
    #if LOCAL_OFF==0
        BARRIER
        load_local_in(vxr);
        load_local_haloz(vxr);
        load_local_halox(vxr);
        BARRIER
    #endif
        vxr_x2 = Dxm(lvxr2);
        vxr_z1 = Dzp(lvxr);

    #if LOCAL_OFF==0
        BARRIER
        load_local_in(vzr);
        load_local_haloz(vzr);
        load_local_halox(vzr);
        BARRIER
    #endif
        vzr_x1 = Dxp(lvzr2);
        vzr_z2 = Dzm(lvzr);
    }

    // To stop updating if we are outside the model (global id must be a
    //multiple of local id in OpenCL, hence we stop if we have a global id
    //outside the grid)
    #if  LOCAL_OFF==0
    #if COMM12==0
    if ( gidz>(NZ-FDOH/DIV-1) ||  (gidx-offcomm)>(NX-FDOH-1-LCOMM) )
        return;
    #else
    if ( gidz>(NZ-FDOH/DIV-1)  )
        return;
    #endif
    #endif

    //Define and load private parameters and variables
    __cprec lsxx = __h22f2(sxx[indv]);
    __cprec lsxz = __h22f2(sxz[indv]);
    __cprec lszz = __h22f2(szz[indv]);
    __cprec lsxxr = __h22f2(sxxr[indv]);
    __cprec lsxzr = __h22f2(sxzr[indv]);
    __cprec lszzr = __h22f2(szzr[indv]);
    __cprec lM = __pconv(M[indp]);
    __cprec lmu = __pconv(mu[indp]);
    __cprec lmuipkp = __pconv(muipkp[indp]);

    // Backpropagate the forward stresses
    #if BACK_PROP_TYPE==1
    {
        // Update the variables
        // Update the variables
        lsxz=lsxz - lmuipkp * (vx_z1+vz_x1);
        lsxx=lsxx - lM*(vx_x2+vz_z2) + 2.0f * lmu * vz_z2;
        lszz=lszz - lM*(vx_x2+vz_z2) + 2.0f * lmu * vx_x2;

        int m=inject_ind(gidz, gidx);
        if (m!=-1){
            lsxx= __h22f2(sxxbnd[m]);
            lszz= __h22f2(szzbnd[m]);
            lsxz= __h22f2(sxzbnd[m]);
        }

        //Write updated values to global memory
        sxx[indv] = __f22h2(lsxx);
        sxz[indv] = __f22h2(lsxz);
        szz[indv] = __f22h2(lszz);

    }
    #endif
    
    // Correct spatial derivatives to implement CPML
    #if ABS_TYPE==1
    {
        int i,k,indm,indn;
        if (DIV*gidz>DIV*NZ-NAB-FDOH-1){
            i =gidx - FDOH;
            k =gidz - NZ + 2*NAB/DIV + FDOH/DIV;
            indm=2*NAB - 1 - k*DIV;
            indn = (i)*(2*NAB/DIV)+(k);

            psi_vx_z[indn] = __f22h2(__hpgi(&b_z_half[indm]) * psi_vx_z[indn]
                                     + __hpgi(&a_z_half[indm]) * vxr_z1);
            vxr_z1 = vxr_z1 / __hpgi(&K_z_half[indm]) + psi_vx_z[indn];
            psi_vz_z[indn] = __f22h2(__hpgi(&b_z[indm+1]) * psi_vz_z[indn]
                                     + __hpgi(&a_z[indm+1]) * vzr_z2);
            vzr_z2 = vzr_z2 / __hpgi(&K_z[indm+1]) + psi_vz_z[indn];
        }

        #if FREESURF==0
        if (DIV*gidz-FDOH<NAB){
            i =gidx-FDOH;
            k =gidz*DIV-FDOH;
            indn = (i)*(2*NAB/DIV)+(k/DIV);

            psi_vx_z[indn] = __f22h2(__hpg(&b_z_half[k]) * psi_vx_z[indn]
                                     + __hpg(&a_z_half[k]) * vxr_z1);
            vxr_z1 = vxr_z1 / __hpg(&K_z_half[k]) + psi_vx_z[indn];
            psi_vz_z[indn] = __f22h2(__hpg(&b_z[k]) * psi_vz_z[indn]
                                     + __hpg(&a_z[k]) * vzr_z2);
            vzr_z2 = vzr_z2 / __hpg(&K_z[k]) + psi_vz_z[indn];
        }
        #endif

        #if DEVID==0 & MYLOCALID==0
        if (gidx-FDOH<NAB){
            i =gidx-FDOH;
            k =gidz-FDOH/DIV;
            indn = (i)*(NZ-2*FDOH/DIV)+(k);

            psi_vx_x[indn] = __f22h2(b_x[i] * psi_vx_x[indn]
                                     + a_x[i] * vxr_x2);
            vxr_x2 = vxr_x2 / K_x[i] + psi_vx_x[indn];
            psi_vz_x[indn] = __f22h2(b_x_half[i] * psi_vz_x[indn]
                                     + a_x_half[i] * vzr_x1);
            vzr_x1 = vzr_x1 / K_x_half[i] + psi_vz_x[indn];
        }
        #endif

        #if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
        if (gidx>NX-NAB-FDOH-1){
            i =gidx - NX+NAB+FDOH+NAB;
            k =gidz-FDOH/DIV;
            indm=2*NAB-1-i;
            indn = (i)*(NZ-2*FDOH/DIV)+(k);

            psi_vx_x[indn] = __f22h2(b_x[indm+1] * psi_vx_x[indn]
                                     + a_x[indm+1] * vxr_x2);
            vxr_x2 = vxr_x2 /K_x[indm+1] + psi_vx_x[indn];
            psi_vz_x[indn] = __f22h2(b_x_half[indm] * psi_vz_x[indn]
                                     + a_x_half[indm] * vzr_x1);
            vzr_x1 = vzr_x1 / K_x_half[indm]  +psi_vz_x[indn];
        }
        #endif
       }
    #endif
    
    // Update adjoint stresses
    {
        lsxzr=lsxzr + lmuipkp * (vxr_z1+vzr_x1);
        lsxxr=lsxxr + lM*(vxr_x2+vzr_z2) - 2.0f * lmu * vzr_z2;
        lszzr=lszzr + lM*(vxr_x2+vzr_z2) - 2.0f * lmu * vxr_x2;


        #if ABS_TYPE==2
        {
        #if FREESURF==0
            if (DIV*gidz-FDOH<NAB){
                lsxxr = lsxxr * __hpg(&taper[DIV*gidz-FDOH]);
                lszzr = lszzr * __hpg(&taper[DIV*gidz-FDOH]);
                lsxzr = lsxzr * __hpg(&taper[DIV*gidz-FDOH]);
            }
        #endif
            if (DIV*gidz>DIV*NZ-NAB-FDOH-1){
                lsxxr =lsxxr * __hpgi(&taper[DIV*NZ-FDOH-DIV*gidz-1]);
                lszzr =lszzr * __hpgi(&taper[DIV*NZ-FDOH-DIV*gidz-1]);
                lsxzr =lsxzr * __hpgi(&taper[DIV*NZ-FDOH-DIV*gidz-1]);
            }

        #if DEVID==0 & MYLOCALID==0
            if (gidx-FDOH<NAB){
                lsxxr = lsxxr * taper[gidx-FDOH];
                lszzr = lszzr * taper[gidx-FDOH];
                lsxzr = lsxzr * taper[gidx-FDOH];
            }
        #endif

        #if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
            if (gidx>NX-NAB-FDOH-1){
                lsxxr = lsxxr * taper[NX-FDOH-gidx-1];
                lszzr = lszzr * taper[NX-FDOH-gidx-1];
                lsxzr = lsxzr * taper[NX-FDOH-gidx-1];
            }
        #endif
        }
        #endif

        //Write updated values to global memory
        sxxr[indv] = __f22h2(lsxxr);
        sxzr[indv] = __f22h2(lsxzr);
        szzr[indv] = __f22h2(lszzr);
    }

    // Shear wave modulus and P-wave modulus gradient calculation on the fly
    #if BACK_PROP_TYPE==1
        #if RESTYPE==0
        //TODO review scaling
        __gprec c1=__h22f2c(1.0f/((2.0f*lM-2.0f*lmu)*(2.0f*lM-2.0f*lmu)));
        __gprec c3=__h22f2c(1.0f/(lmu*lmu));
        __gprec c5=0.25f*c3;

        lsxzr=lmuipkp * (vxr_z1+vzr_x1);
        lsxxr=lM*(vxr_x2+vzr_z2) - 2.0f * lmu * vzr_z2;
        lszzr=lM*(vxr_x2+vzr_z2) - 2.0f * lmu * vxr_x2;

        __gprec dM=c1*( (lsxx+lszz )*( lsxxr+lszzr ));
        gradM[indp]=gradM[indp]-scalefun(dM, 2*par_scale-src_scale - res_scale);
        gradmu[indp]=gradmu[indp] \
                        + scalefun(-c3*(lsxz*lsxzr)
                                   +dM
                                   -c5*(((lsxx-lszz)*(lsxxr-lszzr))),
                                   2*par_scale-src_scale - res_scale);


        #if HOUT==1
        __gprec dMH=c1*((lsxx+lszz )*( lsxx+lszz ));
        HM[indp]=HM[indp] + scalefun(dMH, -2*src_scale);
        Hmu[indp]=Hmu[indp] + scalefun(c3*(lsxz*lsxz)
                                       -dMH
                                       +c5*((lsxx-lszz)*(lsxx-lszz)),
                                       2*par_scale-src_scale - res_scale);
        #endif
        #endif

        #if RESTYPE==1
        __gprec dM=__h22f2c((lsxx+lszz )*( lsxxr+lszzr ));

        gradM[indp]=gradM[indp]-scalefun(dM, -src_scale - res_scale);

        #if HOUT==1
        __gprec dMH=__h22f2c((lsxx+lszz )*( lsxx+lszz));
        HM[indp]=HM[indp] + scalefun(dMH, -2.0*src_scale);

        #endif
        #endif

    #endif


}

