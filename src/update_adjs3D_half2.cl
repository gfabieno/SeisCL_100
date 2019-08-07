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

/*Adjoint update of the stresses in 3D*/

/*Define useful macros to be able to write a matrix formulation in 2D with OpenCl */

FUNDEF void update_adjs(int offcomm,
                        GLOBARG __pprec *muipjp, GLOBARG __pprec *mujpkp,
                        GLOBARG __pprec *muipkp, GLOBARG __pprec *M,
                        GLOBARG __pprec *mu,
                        GLOBARG __prec2 *sxx, GLOBARG __prec2 *sxy,
                        GLOBARG __prec2 *sxz, GLOBARG __prec2 *syy,
                        GLOBARG __prec2 *syz, GLOBARG __prec2 *szz,
                        GLOBARG __prec2 *vx,  GLOBARG __prec2 *vy,
                        GLOBARG __prec2 *vz,
                        GLOBARG __prec2 *sxxbnd, GLOBARG __prec2 *sxybnd,
                        GLOBARG __prec2 *sxzbnd, GLOBARG __prec2 *syybnd,
                        GLOBARG __prec2 *syzbnd, GLOBARG __prec2 *szzbnd,
                        GLOBARG __prec2 *sxxr, GLOBARG __prec2 *sxyr,
                        GLOBARG __prec2 *sxzr, GLOBARG __prec2 *syyr,
                        GLOBARG __prec2 *syzr, GLOBARG __prec2 *szzr,
                        GLOBARG __prec2 *vxr,  GLOBARG __prec2 *vyr,
                        GLOBARG __prec2 *vzr,
                        GLOBARG float *taper,
                        GLOBARG __gprec *gradM,  GLOBARG __gprec *gradmu,
                        GLOBARG __gprec *HM,     GLOBARG __gprec *Hmu,
                        int res_scale, int src_scale, int par_scale)
{

    //Local memory
    extern __shared__ __prec2 lvar2[];
    __prec * lvar=(__prec *)lvar2;

    //Grid position
    int lsizez = blockDim.x+2*FDOH/DIV;
    int lsizey = blockDim.y+2*FDOH;
    int lsizex = blockDim.z+2*FDOH;
    int lidz = threadIdx.x+FDOH/DIV;
    int lidy = threadIdx.y+FDOH;
    int lidx = threadIdx.z+FDOH;
    int gidz = blockIdx.x*blockDim.x+threadIdx.x+FDOH/DIV;
    int gidy = blockIdx.y*blockDim.y+threadIdx.y+FDOH;
    int gidx = blockIdx.z*blockDim.z+threadIdx.z+FDOH+offcomm;
    
    int indp = ((gidx)-FDOH)*(NZ-2*FDOH/DIV)*(NY-2*FDOH)+((gidy)-FDOH)*(NZ-2*FDOH/DIV)+((gidz)-FDOH/DIV);
    int indv = (gidx)*NZ*NY+(gidy)*NZ+(gidz);

    //Define private derivatives
    __cprec vx_x2;
    __cprec vx_y1;
    __cprec vx_z1;
    __cprec vy_x1;
    __cprec vy_y2;
    __cprec vy_z1;
    __cprec vz_x1;
    __cprec vz_y1;
    __cprec vz_z2;

    __cprec vxr_x2;
    __cprec vxr_y1;
    __cprec vxr_z1;
    __cprec vyr_x1;
    __cprec vyr_y2;
    __cprec vyr_z1;
    __cprec vzr_x1;
    __cprec vzr_y1;
    __cprec vzr_z2;

    //Local memory definitions if local is used
    #if LOCAL_OFF==0
        #define lvz lvar
        #define lvy lvar
        #define lvx lvar
        #define lvz2 lvar2
        #define lvy2 lvar2
        #define lvx2 lvar2

        #define lvzr lvar
        #define lvyr lvar
        #define lvxr lvar
        #define lvzr2 lvar2
        #define lvyr2 lvar2
        #define lvxr2 lvar2

        //Local memory definitions if local is not used
    #elif LOCAL_OFF==1
        #define lvz vz
        #define lvy vy
        #define lvx vx
        #define lidz gidz
        #define lidy gidy
        #define lidx gidx

    #endif

#if BACK_PROP_TYPE==1
    //Calculation of the spatial derivatives
    {
    #if LOCAL_OFF==0
        load_local_in(vz);
        load_local_haloz(vz);
        load_local_haloy(vz);
        load_local_halox(vz);
        BARRIER
    #endif
    vz_x1 = Dxp(lvz2);
    vz_y1 = Dyp(lvz2);
    vz_z2 = Dzm(lvz);
        
    #if LOCAL_OFF==0
        BARRIER
        load_local_in(vy);
        load_local_haloz(vy);
        load_local_haloy(vy);
        load_local_halox(vy);
        BARRIER
    #endif
    vy_x1 = Dxp(lvy2);
    vy_y2 = Dym(lvy2);
    vy_z1 = Dzp(lvy);
        
    #if LOCAL_OFF==0
        BARRIER
        load_local_in(vx);
        load_local_haloz(vx);
        load_local_haloy(vx);
        load_local_halox(vx);
        BARRIER
    #endif
    vx_x2 = Dxm(lvx2);
    vx_y1 = Dyp(lvx2);
    vx_z1 = Dzp(lvx);

    BARRIER
        }
#endif

    //Calculation of the spatial derivatives
    {
    #if LOCAL_OFF==0
        load_local_in(vzr);
        load_local_haloz(vzr);
        load_local_haloy(vzr);
        load_local_halox(vzr);
        BARRIER
    #endif
    vzr_x1 = Dxp(lvzr2);
    vzr_y1 = Dyp(lvzr2);
    vzr_z2 = Dzm(lvzr);
        
    #if LOCAL_OFF==0
        BARRIER
        load_local_in(vyr);
        load_local_haloz(vyr);
        load_local_haloy(vyr);
        load_local_halox(vyr);
        BARRIER
    #endif
    vyr_x1 = Dxp(lvyr2);
    vyr_y2 = Dym(lvyr2);
    vyr_z1 = Dzp(lvyr);
        
    #if LOCAL_OFF==0
        BARRIER
        load_local_in(vxr);
        load_local_haloz(vxr);
        load_local_haloy(vxr);
        load_local_halox(vxr);
        BARRIER
    #endif
    vxr_x2 = Dxm(lvxr2);
    vxr_y1 = Dyp(lvxr2);
    vxr_z1 = Dzp(lvxr);
        }
    // To stop updating if we are outside the model (global id must be amultiple
    // of local id in OpenCL, hence we stop if we have a global idoutside the grid)
    #if  LOCAL_OFF==0
    #if COMM12==0
    if ( gidz>(NZ-FDOH/DIV-1) ||  gidy>(NY-FDOH-1) ||  (gidx-offcomm)>(NX-FDOH-1-LCOMM) )
        return;
    #else
    if ( gidz>(NZ-FDOH/DIV-1) ||  gidy>(NY-FDOH-1)  )
        return;
    #endif
    #endif
    
    //Define and load private parameters and variables
    __cprec lsxxr = __h22f2(sxxr[indv]);
    __cprec lsxyr = __h22f2(sxyr[indv]);
    __cprec lsxzr = __h22f2(sxzr[indv]);
    __cprec lsyyr = __h22f2(syyr[indv]);
    __cprec lsyzr = __h22f2(syzr[indv]);
    __cprec lszzr = __h22f2(szzr[indv]);
    __cprec lM = __pconv(M[indp]);
    __cprec lmu = __pconv(mu[indp]);
    __cprec lmuipjp = __pconv(muipjp[indp]);
    __cprec lmuipkp = __pconv(muipkp[indp]);
    __cprec lmujpkp = __pconv(mujpkp[indp]);
    
    // Backpropagate the forward stresses
    #if BACK_PROP_TYPE==1
    __cprec lsxx = __h22f2(sxx[indv]);
    __cprec lsxy = __h22f2(sxy[indv]);
    __cprec lsxz = __h22f2(sxz[indv]);
    __cprec lsyy = __h22f2(syy[indv]);
    __cprec lsyz = __h22f2(syz[indv]);
    __cprec lszz = __h22f2(szz[indv]);
    {
        
        lsxy=lsxy - lmuipjp*(vx_y1+vy_x1);
        lsyz=lsyz - lmujpkp*(vy_z1+vz_y1);
        lsxz=lsxz - lmuipkp*(vx_z1+vz_x1);
        lsxx=lsxx - lM*(vx_x2+vy_y2+vz_z2) + 2.0 * lmu*(vy_y2+vz_z2);
        lsyy=lsyy - lM*(vx_x2+vy_y2+vz_z2) + 2.0 * lmu*(vx_x2+vz_z2);
        lszz=lszz - lM*(vx_x2+vy_y2+vz_z2) + 2.0 * lmu*(vx_x2+vy_y2);
        
        int m=inject_ind(gidz, gidy, gidx);
        if (m!=-1){
            lsxx= __h22f2(sxxbnd[m]);
            lsyy= __h22f2(syybnd[m]);
            lszz= __h22f2(szzbnd[m]);
            lsxy= __h22f2(sxybnd[m]);
            lsxz= __h22f2(sxzbnd[m]);
            lsyz= __h22f2(syzbnd[m]);
        }

        //Write updated values to global memory
        sxx[indv] = __f22h2(lsxx);
        sxy[indv] = __f22h2(lsxy);
        sxz[indv] = __f22h2(lsxz);
        syy[indv] = __f22h2(lsyy);
        syz[indv] = __f22h2(lsyz);
        szz[indv] = __f22h2(lszz);
    }
    #endif
    
    
    // Update adjoint stresses
    {
        // Update the variables
        lsxyr=lsxyr + lmuipjp*(vxr_y1+vyr_x1);
        lsyzr=lsyzr + lmujpkp*(vyr_z1+vzr_y1);
        lsxzr=lsxzr + lmuipkp*(vxr_z1+vzr_x1);
        lsxxr=lsxxr + lM*(vxr_x2+vyr_y2+vzr_z2) - 2.0 * lmu*(vyr_y2+vzr_z2);
        lsyyr=lsyyr + lM*(vxr_x2+vyr_y2+vzr_z2) - 2.0 * lmu*(vxr_x2+vzr_z2);
        lszzr=lszzr + lM*(vxr_x2+vyr_y2+vzr_z2) - 2.0 * lmu*(vxr_x2+vyr_y2);
    
    // Absorbing boundary
    #if ABS_TYPE==2
        {
        #if FREESURF==0
        if (DIV*gidz-FDOH<NAB){
            lsxyr = lsxyr * __hp(&taper[DIV*gidz-FDOH]);
            lsyzr = lsyzr * __hp(&taper[DIV*gidz-FDOH]);
            lsxzr = lsxzr * __hp(&taper[DIV*gidz-FDOH]);
            lsxxr = lsxxr * __hp(&taper[DIV*gidz-FDOH]);
            lszzr = lszzr * __hp(&taper[DIV*gidz-FDOH]);
            lsxzr = lsxzr * __hp(&taper[DIV*gidz-FDOH]);
        }
        #endif

        if (DIV*gidz>DIV*NZ-NAB-FDOH-1){
            lsxyr = lsxyr * __hpi(&taper[DIV*NZ-FDOH-DIV*gidz-1]);
            lsyzr = lsyzr * __hpi(&taper[DIV*NZ-FDOH-DIV*gidz-1]);
            lsxzr = lsxzr * __hpi(&taper[DIV*NZ-FDOH-DIV*gidz-1]);
            lsxxr = lsxxr * __hpi(&taper[DIV*NZ-FDOH-DIV*gidz-1]);
            lszzr = lszzr * __hpi(&taper[DIV*NZ-FDOH-DIV*gidz-1]);
            lsxzr = lsxzr * __hpi(&taper[DIV*NZ-FDOH-DIV*gidz-1]);
        }
        if (gidy-FDOH<NAB){
            lsxyr = lsxyr * taper[gidy-FDOH];
            lsyzr = lsyzr * taper[gidy-FDOH];
            lsxzr = lsxzr * taper[gidy-FDOH];
            lsxxr = lsxxr * taper[gidy-FDOH];
            lszzr = lszzr * taper[gidy-FDOH];
            lsxzr = lsxzr * taper[gidy-FDOH];
        }

        if (gidy>NY-NAB-FDOH-1){
            lsxyr = lsxyr * taper[NY-FDOH-gidy-1];
            lsyzr = lsyzr * taper[NY-FDOH-gidy-1];
            lsxzr = lsxzr * taper[NY-FDOH-gidy-1];
            lsxxr = lsxxr * taper[NY-FDOH-gidy-1];
            lszzr = lszzr * taper[NY-FDOH-gidy-1];
            lsxzr = lsxzr * taper[NY-FDOH-gidy-1];
        }
        #if DEVID==0 & MYLOCALID==0
        if (gidx-FDOH<NAB){
            lsxyr = lsxyr * taper[gidx-FDOH];
            lsyzr = lsyzr * taper[gidx-FDOH];
            lsxzr = lsxzr * taper[gidx-FDOH];
            lsxxr = lsxxr * taper[gidx-FDOH];
            lszzr = lszzr * taper[gidx-FDOH];
            lsxzr = lsxzr * taper[gidx-FDOH];
        }
        #endif

        #if DEVID==NUM_DEVICES-1 & MYLOCALID==NLOCALP-1
        if (gidx>NX-NAB-FDOH-1){
            lsxyr = lsxyr * taper[NX-FDOH-gidx-1];
            lsyzr = lsyzr * taper[NX-FDOH-gidx-1];
            lsxzr = lsxzr * taper[NX-FDOH-gidx-1];
            lsxxr = lsxxr * taper[NX-FDOH-gidx-1];
            lszzr = lszzr * taper[NX-FDOH-gidx-1];
            lsxzr = lsxzr * taper[NX-FDOH-gidx-1];
        }
        #endif
        }
    #endif
    
    //Write updated values to global memory
    sxxr[indv] = __f22h2(lsxxr);
    sxyr[indv] = __f22h2(lsxyr);
    sxzr[indv] = __f22h2(lsxzr);
    syyr[indv] = __f22h2(lsyyr);
    syzr[indv] = __f22h2(lsyzr);
    szzr[indv] = __f22h2(lszzr);
        
    }
    
    //Shear wave modulus and P-wave modulus gradient calculation on the fly
    #if BACK_PROP_TYPE==1
    lsxyr=lmuipjp*(vxr_y1+vyr_x1);
    lsyzr=lmujpkp*(vyr_z1+vzr_y1);
    lsxzr=lmuipkp*(vxr_z1+vzr_x1);
    lsxxr=lM*(vxr_x2+vyr_y2+vzr_z2) - 2.0 * lmu*(vyr_y2+vzr_z2);
    lsyyr=lM*(vxr_x2+vyr_y2+vzr_z2) - 2.0 * lmu*(vxr_x2+vzr_z2);
    lszzr=lM*(vxr_x2+vyr_y2+vzr_z2) - 2.0 * lmu*(vxr_x2+vyr_y2);

    #if RESTYPE==0
    __gprec c1=1.0/(3.0*lM-4.0*lmu)/(3.0*lM-4.0*lmu);
    __gprec c3=1.0/lmu/lmu;
    __gprec c5=1.0/6.0*c3;
    
    __gprec dM=c1*__h22f2c(( lsxx+lsyy+lszz )*( lsxxr+lsyyr+lszzr ));
    gradM[indp] = gradM[indp] - scalefun(dM, 2*par_scale-src_scale - res_scale);
    
    gradmu[indp]=gradmu[indp] \
                 + scalefun(-c3*(lsxz*lsxzr +lsxy*lsxyr +lsyz*lsyzr)
                            + 4.0/3*dM
                            -c5*(lsxxr*(2.0*lsxx-lsyy-lszz)
                                +lsyyr*(2.0*lsyy-lsxx-lszz)
                                +lszzr*(2.0*lszz-lsxx-lsyy)),
                            2*par_scale-src_scale - res_scale);

    #if HOUT==1
    dM=c1*__h22f2c(( lsxx+lsyy+lszz )*( lsxx+lsyy+lszz ));
    HM[indp] = HM[indp] + scalefun(dM, 2*par_scale-src_scale - res_scale);

    Hmu[indp]=Hmu[indp] + scalefun(-c3*(lsxz*lsxz +lsxy*lsxy +lsyz*lsyz)
                                   + 4.0/3*dM
                                   -c5*(lsxx*(2.0*lsxx-lsyy-lszz)
                                        +lsyy*(2.0*lsyy-lsxx-lszz)
                                        +lszz*(2.0*lszz-lsxx-lsyy)),
                                   2*par_scale-src_scale - res_scale);
    #endif
    #endif

    #if RESTYPE==1
    __gprec dM=__h22f2c(( lsxx+lsyy+lszz )*( lsxxr+lsyyr+lszzr ));

    gradM[indp] = gradM[indp] - scalefun(dM, 2*par_scale-src_scale - res_scale);

    #if HOUT==1
    dM= __h22f2c(( lsxx+lsyy+lszz )*( lsxx+lsyy+lszz ));
    HM[indp] = HM[indp] - scalefun(dM, 2*par_scale-src_scale - res_scale);
    #endif
    #endif

    #endif
    
    #if GRADSRCOUT==1
    //TODO
    //    float pressure;
    //    if (nsrc>0){
    //
    //        for (int srci=0; srci<nsrc; srci++){
    //
    //            int SOURCE_TYPE= (int)srcpos_loc(4,srci);
    //
    //            if (SOURCE_TYPE==1){
    //                int i=(int)(srcpos_loc(0,srci)-0.5)+FDOH;
    //                int j=(int)(srcpos_loc(1,srci)-0.5)+FDOH;
    //                int k=(int)(srcpos_loc(2,srci)-0.5)+FDOH;
    //
    //
    //                if (i==gidx && j==gidy && k==gidz){
    //
    //                    pressure=(lsxxr+lsyyr+lszzr )/(2.0*DH*DH*DH);
    //                    if ( (nt>0) && (nt< NT ) ){
    //                        gradsrc(srci,nt+1)+=pressure;
    //                        gradsrc(srci,nt-1)-=pressure;
    //                    }
    //                    else if (nt==0)
    //                        gradsrc(srci,nt+1)+=pressure;
    //                    else if (nt==NT)
    //                        gradsrc(srci,nt-1)-=pressure;
    //
    //                }
    //            }
    //        }
    //    }
    
    #endif
    
}

