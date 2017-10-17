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




#include "F.h"


int kernel_varout(device * dev,
                  clprogram * prog){
 
    int state=0;
    int i,j,k;
    int scaler=0;
    
    char temp[MAX_KERN_STR]={0};;
    char temp2[100]={0};
    char * p=(char*)temp;
    variable * vars = dev->vars;
    variable * tvars = dev->trans_vars;
    
    strcat(temp, "extern \"C\" __global__ void varsout"
                 "(int nt, int nrec, float * rec_pos, float src_scale,");
    for (i=0;i<dev->nvars;i++){
        if (vars[i].to_output){
            if (dev->FP16==1){
                strcat(temp, "half * ");
            }
            else{
                strcat(temp, "float * ");
            }
            strcat(temp, vars[i].name);
            strcat(temp, ", ");
            strcat(temp, "float * ");
            strcat(temp, vars[i].name);
            strcat(temp, "out, ");
        }
    }
    for (i=0;i<dev->ntvars;i++){
        if (tvars[i].to_output){
            for (j=0;j<tvars[i].n2ave;j++){
                if (dev->FP16==1){
                    strcat(temp, "half * ");
                }
                else{
                    strcat(temp, "float * ");
                }
                strcat(temp, tvars[i].var2ave[j]);
                strcat(temp, ", ");
            }

            strcat(temp, "float * ");
            strcat(temp, tvars[i].name);
            strcat(temp, "out, ");
        }
    }
    while (*p)
        p++;
    p[-2]='\0';
    strcat(temp, "){\n\n");

    //This only supports 3 dimensions (need for more ?)
    strcat(temp,"    int gid = blockIdx.x*blockDim.x + threadIdx.x;\n");
    sprintf(temp2,"    if (gid > nrec-1){\n" );
    strcat(temp, temp2);
    strcat(temp,
           "        return;\n"
           "    };\n\n");
    
    strcat(temp,       "    int i=(int)(rec_pos[0+8*gid]/DH)+FDOH;\n"
           "    int j=(int)(rec_pos[1+8*gid]/DH)+FDOH;\n"
           "    int k=(int)(rec_pos[2+8*gid]/DH)+FDOH;\n"
           "\n");
    
    sprintf(temp2,"    if (i-OFFSET<FDOH || i-OFFSET>N%s-FDOH-1){\n",
            dev->N_names[dev->NDIM-1] );
    strcat(temp, temp2);
    strcat(temp,
           "        return;\n"
           "    };\n\n");

    
    char posstr[100]={0};
    
    if (dev->NDIM==2){
        if (dev->FP16==1){
            sprintf(posstr,"[(i-OFFSET)*N%s*2+k]",dev->N_names[0]);
        }
        else{
            sprintf(posstr,"[(i-OFFSET)*N%s+k]",dev->N_names[0]);
        }
    }
    else if (dev->NDIM==3){
        if (dev->FP16==1){
            sprintf(posstr,"[(i-OFFSET)*N%s*N%s*2+j*(N%s*2)+k]",
                    dev->N_names[1], dev->N_names[0], dev->N_names[0]);
        }
        else{
            sprintf(posstr,"[(i-OFFSET)*N%s*N%s*2+j*(N%s*2)+k]",
                    dev->N_names[1], dev->N_names[0], dev->N_names[0]);
        }
        
    }
    
    for (i=0;i<dev->nvars;i++){
        if (vars[i].to_output){
            strcat(temp, "    ");
            strcat(temp, vars[i].name);
            strcat(temp, "out[NT*gid+nt]=1.0/src_scale*");
            if (abs(vars[i].scaler)>0){
                strcat(temp, "scalbnf(");
            }
            if (dev->FP16==1){
                strcat(temp, "__half2float(");
            }
            strcat(temp, vars[i].name);
            strcat(temp, posstr);
            if (dev->FP16==1){
                strcat(temp, ")");
            }
            if (abs(vars[i].scaler)>0){
                sprintf(temp2,", %d)", -vars[i].scaler );
                strcat(temp, temp2);
            }
            strcat(temp, ";\n");
        }
    }
    for (i=0;i<dev->ntvars;i++){
        if (tvars[i].to_output){
            strcat(temp, "    ");
            strcat(temp, tvars[i].name);
            strcat(temp, "out[NT*gid+nt]=1.0/src_scale*(");
            for (j=0;j<tvars->n2ave;j++){
                for (k=0;k<dev->nvars;k++){
                    if (strcmp(tvars[i].var2ave[j],dev->vars[k].name)==0){
                        scaler=dev->vars[k].scaler;
                        break;
                    }
                }
                if (abs(scaler)>0){
                    strcat(temp, "scalbnf(");
                }
                if (dev->FP16==1){
                    strcat(temp, "__half2float(");
                }
                strcat(temp, tvars[i].var2ave[j]);
                strcat(temp, posstr);
                if (dev->FP16==1){
                    strcat(temp, ")");
                }
                if (abs(scaler)>0){
                    sprintf(temp2,", %d)", -scaler );
                    strcat(temp, temp2);
                }
                strcat(temp, "+");
            }
            while (*p)
                p++;
            p[-1]='\0';
            strcat(temp, ")/");
            sprintf(temp2,"%f",(float)tvars->n2ave);
            strcat(temp, temp2);
            strcat(temp, ";\n");
        }
    }

    strcat(temp, "\n}");
    
    
    printf("%s\n\n%lu\n",temp, strlen(temp));
    
    
    (*prog).src=temp;
    
    __GUARD prog_source(prog, "varsout", (*prog).src);
    
    prog->wdim=1;
    
    return state;

}


int kernel_varoutinit(device * dev,
                      clprogram * prog){
    
    int state=0;
    int i;
    variable * vars = dev->vars;
    variable * tvars = dev->trans_vars;
    char temp[MAX_KERN_STR]={0};;
    
    char * p=(char*)temp;
    
    strcat(temp, "extern \"C\" __global__ void varsoutinit(int nrec,");
    for (i=0;i<dev->nvars;i++){
        if (vars[i].to_output){
            strcat(temp, "float * ");
            strcat(temp, vars[i].name);
            strcat(temp, "out, ");
        }
    }
    for (i=0;i<dev->ntvars;i++){
        if (tvars[i].to_output){
            strcat(temp, "float * ");
            strcat(temp, tvars[i].name);
            strcat(temp, "out, ");
        }
    }
    while (*p)
        p++;
    p[-2]='\0';
    strcat(temp, "){\n\n");
    
    
    strcat(temp,"    int gid = blockIdx.x*blockDim.x + threadIdx.x;\n\n");
    
    strcat(temp,"    if (gid > nrec*NT-1){\n"
                "        return;\n"
                "    };\n\n");
    
    for (i=0;i<dev->nvars;i++){
        if (vars[i].to_output){
            strcat(temp, "    ");
            strcat(temp, vars[i].name);
            strcat(temp, "out[gid]=0;\n");
        }
    }
    for (i=0;i<dev->ntvars;i++){
        if (tvars[i].to_output){
            strcat(temp, "    ");
            strcat(temp, tvars[i].name);
            strcat(temp, "out[gid]=0;\n");
        }
    }

    strcat(temp, "\n}");
    
    
//    printf("%s\n\n%lu\n",temp, strlen(temp));
    
    
    (*prog).src=temp;
    
    __GUARD prog_source(prog, "varsoutinit", (*prog).src);
    
    prog->wdim=1;
    
    return state;
    
}

int kernel_varinit(device * dev,
                   model * m,
                   variable * vars,
                   clprogram * prog){
    
    int state=0;
    int i;
    
    char temp[MAX_KERN_STR]={0};;
    
    char * p=(char*)temp;
    char ptemp[50];
    
    int maxsize=0;
    for (i=0;i<dev->nvars;i++){
        if (vars[i].num_ele>maxsize){
            maxsize=vars[i].num_ele;
        }
    }
    
    
    strcat(temp, "extern \"C\" __global__ void vars_init(");
    for (i=0;i<dev->nvars;i++){
        if (dev->FP16==1){
            strcat(temp, "half2 * ");
        }
        else{
            strcat(temp, "float * ");
        }
            strcat(temp, vars[i].name);
            strcat(temp, ", ");
    }
    while (*p)
        p++;
    p[-2]='\0';
    strcat(temp, "){\n\n");
    
    if (dev->NDIM==2){
       
        strcat(temp,"int gidz = blockIdx.x*blockDim.x + threadIdx.x;\n"
                    "int gidx = blockIdx.y*blockDim.y + threadIdx.y;\n"
                    "int gid = gidx*blockDim.x*gridDim.x+gidz;\n");
    }
    else if (dev->NDIM==3){
        strcat(temp,"int gidz = blockIdx.x*blockDim.x + threadIdx.x;\n"
                    "int gidy = blockIdx.y*blockDim.y + threadIdx.y;\n"
                    "int gidx = blockIdx.y*blockDim.z + threadIdx.z;\n"
                    "int gid = gidx*blockDim.x*gridDim.x*blockDim.y*gridDim.y"
                    "+gidy*blockDim.x*gridDim.x +gidz;\n");
    }
    
    for (i=0;i<dev->nvars;i++){
        if (dev->FP16==1){
            sprintf(ptemp,"    if (gid<%d)\n", vars[i].num_ele/2);
        }
        else{
            sprintf(ptemp,"    if (gid<%d)\n", vars[i].num_ele);
        }
        strcat(temp,ptemp);
        strcat(temp, "    ");
        strcat(temp, "    ");
        strcat(temp, vars[i].name);
        if (dev->FP16==1){
            strcat(temp, "[gid]= __float2half2_rn(0);\n");
        }
        else{
            strcat(temp, "[gid]=0;\n");
        }
        
    }
    
    strcat(temp, "\n}");
    
    
//    printf("%s\n\n%lu\n",temp, strlen(temp));
    
    
    (*prog).src=temp;
    
    __GUARD prog_source(prog, "vars_init", (*prog).src);
    
    if (dev->FP16==1){
        prog->gsize[0]=dev->N[0]/2+m->FDOH;
    }
    else{
        prog->gsize[0]=dev->N[0]+m->FDORDER;
    }
    prog->gsize[1]=dev->N[1]+m->FDORDER;
    if (dev->NDIM==3){
        prog->gsize[2]=dev->N[2]+m->FDORDER;
    }
    
    prog->wdim=dev->NDIM;
    
    return state;
    
}

int kernel_sources(device * dev,
                   clprogram * prog){
    
    int state=0;
    int i,j;
    variable * vars = dev->vars;
    variable * tvars = dev->trans_vars;
    char temp[MAX_KERN_STR]={0};
    char temp2[100]={0};
//    
    char * p=(char*)temp;
    
    
    
    int * tosources=NULL;
    int * tosources2=NULL;
    int ntypes=0;
    int ind;
    GMALLOC(tosources,dev->nvars*sizeof(int));
    GMALLOC(tosources2,dev->ntvars*sizeof(int));
    for (i=0;i<dev->src_recs.allns;i++){
        ind =dev->src_recs.src_pos[0][4+i*5];
        if (ind<dev->nvars && ind>-1)
            tosources[ind]=1;
        if (ind-100<dev->ntvars && ind-100>-1)
            tosources2[ind-100]=1;
    }

    for (i=0;i<dev->nvars;i++){
        if (tosources[i]==1){
            ntypes++;
        }
    }
    for (i=0;i<dev->ntvars;i++){
        if (tosources2[i]==1){
            ntypes++;
        }
    }
    
    if (ntypes==0){
        state=1;
        fprintf(stderr,"Error: No sources for variable list found\n");
    }
    
    strcat(temp, "extern \"C\" __global__ void sources(int nt, int nsrc,"
                 "float src_scale, float * src_pos, float * src, int pdir, ");
    for (i=0;i<dev->nvars;i++){
        if (tosources[i]){
            if (dev->FP16==1){
                strcat(temp, "half * ");
            }
            else{
                strcat(temp, "float * ");
            }
            strcat(temp, vars[i].name);
            strcat(temp, ", ");
        }
    }
    for (i=0;i<dev->ntvars;i++){
        if (tosources2[i]){
            for (j=0;j<tvars[i].n2ave;j++){
                if (dev->FP16==1){
                    strcat(temp, "half * ");
                }
                else{
                    strcat(temp, "float * ");
                }
                strcat(temp, tvars[i].var2ave[j]);
                strcat(temp, ", ");
            }
        }
    }

    while (*p)
        p++;
    p[-2]='\0';
    strcat(temp, "){\n\n");
    
    //This only supports 3 dimensions (need for more ?)
    strcat(temp,"    int gid = blockIdx.x*blockDim.x + threadIdx.x;\n");
    sprintf(temp2,"    if (gid > nsrc -1){\n");
    strcat(temp, temp2);
    strcat(temp,
           "        return;\n"
           "    }\n\n");
    strcat(temp,"    int i=(int)(src_pos[0+5*gid]/DH)+FDOH;\n"
           "    int j=(int)(src_pos[1+5*gid]/DH)+FDOH;\n"
           "    int k=(int)(src_pos[2+5*gid]/DH)+FDOH;\n"
           "\n");

    sprintf(temp2,"    if (i-OFFSET<FDOH || i-OFFSET>N%s-FDOH-1){\n",
            dev->N_names[dev->NDIM-1] );
    strcat(temp, temp2);
    strcat(temp,
           "        return;\n"
           "    }\n\n"
           "    int source_type= src_pos[4+5*gid];\n"
           "    float amp=(float)pdir*(DT*src[gid*NT+nt])/(DH*DH*DH)*src_scale;\n\n");
    
    char posstr[100]={0};

    
    if (dev->NDIM==2){
        if (dev->FP16==1){
            sprintf(posstr,"[(i-OFFSET)*N%s*2+k]",dev->N_names[0]);
        }
        else{
            sprintf(posstr,"[(i-OFFSET)*N%s+k]",dev->N_names[0]);
        }
    }
    else if (dev->NDIM==3){
        if (dev->FP16==1){
            sprintf(posstr,"[(i-OFFSET)*N%s*N%s*2+j*(N%s*2)+k]",
                    dev->N_names[1], dev->N_names[0], dev->N_names[0]);
        }
        else{
            sprintf(posstr,"[(i-OFFSET)*N%s*N%s*2+j*(N%s*2)+k]",
                    dev->N_names[1], dev->N_names[0], dev->N_names[0]);
        }
        
    }
    else{
        state=1;
        fprintf(stderr,"Error: Sources for a number of dimensions higher "
                       "than 3 are not supported yet\n");
    }
    
    
    for (i=0;i<dev->nvars;i++){
        
        if (tosources[i]){
            if (ntypes>1){
                sprintf(temp2,"    if (source_type==%d)\n", i);
                strcat(temp, temp2);
                strcat(temp, "    ");
            }
            strcat(temp, "    ");
            strcat(temp, vars[i].name);
            strcat(temp, posstr);
            if (dev->FP16==1){
                strcat(temp, "=__float2half(__half2float(");
                strcat(temp, vars[i].name);
                strcat(temp, posstr);
                strcat(temp, ")+amp);\n");
            }
            else{
                strcat(temp, "+=amp;\n");
            }

            
        }
    }
    for (i=0;i<dev->ntvars;i++){
        
        if (tosources2[i]){
            if (ntypes>1){
                sprintf(temp2,"    if (source_type==%d){\n", i+100);
                strcat(temp, temp2);
                strcat(temp, "    ");
            }
            for (j=0;j<tvars[i].n2ave;j++){
                strcat(temp, "    ");
                strcat(temp, tvars[i].var2ave[j]);
                strcat(temp, posstr);
                if (dev->FP16==1){
                    strcat(temp, "=__float2half(__half2float(");
                    strcat(temp, tvars[i].var2ave[j]);
                    strcat(temp, posstr);
                    sprintf(temp2,")+amp/%f);\n", (float)tvars[i].n2ave);
                    strcat(temp, temp2);
                }
                else{
                    sprintf(temp2,"+=amp/%f;\n", (float)tvars[i].n2ave);
                    strcat(temp, temp2);
                }
            }
            if (ntypes>1){
                strcat(temp, "    }");
            }
        }
    }
    
    strcat(temp, "\n}");
    
    (*prog).src=temp;
    
    __GUARD prog_source(prog, "sources", (*prog).src);
    
    printf("%s\n\n%lu\n",temp, strlen(temp));
    
    free(tosources);
    free(tosources2);
    
    prog->wdim=1;
    
    return state;
    
}

int kernel_residuals(device * dev,
                     clprogram * prog,
                     int BACK_PROP_TYPE){
    
    int state=0;
    int i,j;
    
    char temp[MAX_KERN_STR]={0};;
    char temp2[100]={0};
    char * p=(char*)temp;
    variable * vars = dev->vars;
    variable * tvars = dev->trans_vars;
    
   
    
    strcat(temp, "extern \"C\" __global__ void residuals(int nt, int nrec,"
                 "float * rec_pos,");
    for (i=0;i<dev->nvars;i++){
        if (vars[i].to_output){
            strcat(temp, "float * ");
            strcat(temp, vars[i].name);
            if (BACK_PROP_TYPE==1)
                strcat(temp, "_r");
            strcat(temp, ", ");
            strcat(temp, "float * ");
            strcat(temp, vars[i].name);
            strcat(temp, "out, ");
        }
    }
    for (i=0;i<dev->ntvars;i++){
        if (tvars[i].to_output){
            for (j=0;j<tvars[i].n2ave;j++){
                strcat(temp, "float * ");
                strcat(temp, tvars[i].var2ave[j]);
                if (BACK_PROP_TYPE==1)
                    strcat(temp, "_r");
                strcat(temp, ", ");
            }
            
            strcat(temp, "float * ");
            strcat(temp, tvars[i].name);
            strcat(temp, "out, ");
        }
    }
    while (*p)
        p++;
    p[-2]='\0';
    strcat(temp, "){\n\n");
    
    
    strcat(temp,"    int gid = blockIdx.x*blockDim.x + threadIdx.x;\n");
    
    sprintf(temp2,"    if (gid > nrec-1){\n");
    strcat(temp, temp2);
    strcat(temp,
           "        return;\n"
           "    };\n\n");
    
     strcat(temp,"    int i=(int)(rec_pos[0+8*gid]/DH)+FDOH;\n"
           "    int j=(int)(rec_pos[1+8*gid]/DH)+FDOH;\n"
           "    int k=(int)(rec_pos[2+8*gid]/DH)+FDOH;\n\n");
    
    sprintf(temp2,"    if ( i-OFFSET<FDOH || i-OFFSET>N%s-FDOH-1){\n",
            dev->N_names[dev->NDIM-1] );
    strcat(temp, temp2);
    strcat(temp,
           "        return;\n"
           "    };\n\n");
    
    char posstr[100]={0};

    if (dev->NDIM==2){
        if (dev->FP16==1){
            sprintf(posstr,"[(i-OFFSET)*N%s*2+k]",dev->N_names[0]);
        }
        else{
            sprintf(posstr,"[(i-OFFSET)*N%s+k]",dev->N_names[0]);
        }
    }
    else if (dev->NDIM==3){
        if (dev->FP16==1){
            sprintf(posstr,"[(i-OFFSET)*N%s*N%s*2+j*(N%s*2)+k]",
                    dev->N_names[1], dev->N_names[0], dev->N_names[0]);
        }
        else{
            sprintf(posstr,"[(i-OFFSET)*N%s*N%s*2+j*(N%s*2)+k]",
                    dev->N_names[1], dev->N_names[0], dev->N_names[0]);
        }
        
    }

    
    
    for (i=0;i<dev->nvars;i++){
        if (vars[i].to_output){
            strcat(temp, "        ");
            strcat(temp, vars[i].name);
            if (BACK_PROP_TYPE==1)
                strcat(temp, "_r");
            strcat(temp, posstr);
            strcat(temp, "+=");
            strcat(temp, vars[i].name);
            strcat(temp, "out[NT*gid+nt];\n");

        }
    }
    for (i=0;i<dev->ntvars;i++){
        if (tvars[i].to_output){
            for (j=0;j<tvars[i].n2ave;j++){
                strcat(temp, "        ");
                strcat(temp, tvars[i].var2ave[j]);
                if (BACK_PROP_TYPE==1)
                    strcat(temp, "_r");
                strcat(temp, posstr);
                strcat(temp, "+=");
                strcat(temp, tvars[i].name);
                strcat(temp, "out[NT*gid+nt]/");
                sprintf(temp2,"%d",tvars[i].n2ave);
                strcat(temp, temp2);
                strcat(temp, ";\n");
            }
        }
    }
    
    strcat(temp, "\n}");
    
    (*prog).src=temp;
    
    __GUARD prog_source(prog, "residuals", (*prog).src);

//    printf("%s\n\n%lu\n",temp, strlen(temp));
    
    prog->wdim=1;
    
    return state;
    
}



int kernel_gradinit(device * dev,
                    parameter * pars,
                    clprogram * prog){
    
    int state=0;
    int i;
    
    char temp[MAX_KERN_STR]={0};;
    char temp2[100]={0};
    char * p=(char*)temp;
    

    strcat(temp, "extern \"C\" __global__ void gradinit(");
    for (i=0;i<dev->npars;i++){
        if (pars[i].to_grad){
            strcat(temp, "float * grad");
            strcat(temp, pars[i].name);
            strcat(temp, ", ");
        }
    }
    while (*p)
        p++;
    p[-2]='\0';
    strcat(temp, "){\n\n");
    
    
    if (dev->NDIM==2){
        
        strcat(temp,"int gidz = blockIdx.x*blockDim.x + threadIdx.x;\n"
               "int gidx = blockIdx.y*blockDim.y + threadIdx.y;\n"
               "int gid = gidx*blockDim.x*gridDim.x+gidz;\n");
    }
    else if (dev->NDIM==3){
        strcat(temp,"int gidz = blockIdx.x*blockDim.x + threadIdx.x;\n"
               "int gidy = blockIdx.y*blockDim.y + threadIdx.y;\n"
               "int gidx = blockIdx.y*blockDim.z + threadIdx.z;\n"
               "int gid = gidx*blockDim.x*gridDim.x*blockDim.y*gridDim.y"
               "+gidy*blockDim.x*gridDim.x +gidz;\n");
    }
    
    if (dev->npars>0){
        if (dev->FP16==1){
            sprintf(temp2,"    if (gid>%d-1){\n", pars[0].num_ele/2);
        }
        else{
            sprintf(temp2,"    if (gid>%d-1){\n", pars[0].num_ele);
        }
        strcat(temp,temp2);
        strcat(temp,  "        return;\n"
                      "    };\n\n");
    }
    
    for (i=0;i<dev->npars;i++){
        if (pars[i].to_grad){
            strcat(temp, "    grad");
            strcat(temp, pars[i].name);
            strcat(temp, "[gid]=0;\n");
        }
    }

    strcat(temp, "\n}");
    
   (*prog).src=temp;
    
    __GUARD prog_source(prog, "gradinit", (*prog).src);
    
//    printf("%s\n\n%lu\n",temp, strlen(temp));
    
    if (dev->FP16==1){
        prog->gsize[0]=dev->N[0]/2;
    }
    else{
        prog->gsize[0]=dev->N[0];
    }
    prog->gsize[1]=dev->N[1];
    if (dev->NDIM==3){
        prog->gsize[2]=dev->N[2];
    }
    
    prog->wdim=dev->NDIM;
    
    return state;
    
}

int kernel_initsavefreqs(device * dev,
                         variable * vars,
                         clprogram * prog){
    
    int state=0;
    int i;
    
    char temp[MAX_KERN_STR]={0};;
    
    char * p=(char*)temp;
    char ptemp[50];
    
    int maxsize=0;
    for (i=0;i<dev->nvars;i++){
        if (vars[i].num_ele>maxsize && vars[i].for_grad){
            maxsize=vars[i].num_ele;
        }
    }
    
    strcat(temp, "extern \"C\" __global__ void initsavefreqs(");
    for (i=0;i<dev->nvars;i++){
        if (vars[i].for_grad){
            strcat(temp, "float2 * f");
            strcat(temp, vars[i].name);
            strcat(temp, ", ");
        }
    }
    while (*p)
        p++;
    p[-2]='\0';
    strcat(temp, "){\n\n");
    
    
    if (dev->NDIM==2){
        
        strcat(temp,"int gidz = blockIdx.x*blockDim.x + threadIdx.x;\n"
               "int gidx = blockIdx.y*blockDim.y + threadIdx.y;\n"
               "int gid = gidx*blockDim.x*gridDim.x+gidz;\n");
    }
    else if (dev->NDIM==3){
        strcat(temp,"int gidz = blockIdx.x*blockDim.x + threadIdx.x;\n"
               "int gidy = blockIdx.y*blockDim.y + threadIdx.y;\n"
               "int gidx = blockIdx.y*blockDim.z + threadIdx.z;\n"
               "int gid = gidx*blockDim.x*gridDim.x*blockDim.y*gridDim.y"
               "+gidy*blockDim.x*gridDim.x +gidz;\n");
    }
    
    
    for (i=0;i<dev->nvars;i++){
        if (vars[i].for_grad){
            if (dev->FP16==1){
                 sprintf(ptemp,"    if (gid<%d)\n", vars[i].num_ele/2);
            }
            else{
                 sprintf(ptemp,"    if (gid<%d)\n", vars[i].num_ele);
            }
            strcat(temp,ptemp);
            strcat(temp, "    ");
            strcat(temp, "    f");
            strcat(temp, vars[i].name);
            strcat(temp, "[gid]=0;\n");
        }
    }
    
    strcat(temp, "\n}");
    
    (*prog).src=temp;
    
    __GUARD prog_source(prog, "initsavefreqs", (*prog).src);
    
//    printf("%s\n\n%lu\n",temp, strlen(temp));
    
    if (dev->FP16==1){
        prog->gsize[0]=dev->N[0]/2;
    }
    else{
        prog->gsize[0]=dev->N[0];
    }
    prog->gsize[1]=dev->N[1];
    if (dev->NDIM==3){
        prog->gsize[2]=dev->N[2];
    }
    
    prog->wdim=dev->NDIM;
    
    return state;
    
}

int kernel_savefreqs(device * dev,
                     variable * vars,
                     clprogram * prog){
    
    int state=0;
    int i;
    
    char temp[MAX_KERN_STR]={0};;
    
    char * p=(char*)temp;
    char ptemp[50];
    
    int maxsize=0;
    for (i=0;i<dev->nvars;i++){
        if (vars[i].num_ele>maxsize && vars[i].for_grad){
            maxsize=vars[i].num_ele;
        }
    }
    
    strcat(temp,"extern \"C\" __global__ void savefreqs(__constant float *gradfreqsn, int nt, ");
    for (i=0;i<dev->nvars;i++){
        if (vars[i].for_grad){
            strcat(temp, "float * ");
            strcat(temp, vars[i].name);
            strcat(temp, ", ");
            strcat(temp, "float2 * f");
            strcat(temp, vars[i].name);
            strcat(temp, ", ");
        }
    }
    while (*p)
        p++;
    p[-2]='\0';
    strcat(temp, "){\n\n");
    
    strcat(temp,"    int freq,l;\n"
                "    float2 fact[NFREQS]={0};\n");
    
    for (i=0;i<dev->nvars;i++){
        if (vars[i].for_grad){
            strcat(temp, "    float  l");
            strcat(temp, vars[i].name);
            strcat(temp, ";\n");
        }
    }
    while (*p)
        p++;
    p[-2]='\0';
    strcat(temp, ";\n\n");

    strcat(temp,"    int gid = blockIdx.x*blockDim.x + threadIdx.x;\n"
                "    int gsize=blockDim.x * gridDim.x;\n\n" );
    if (dev->NDIM==2){
        
        strcat(temp,"int gidz = blockIdx.x*blockDim.x + threadIdx.x;\n"
               "    int gidx = blockIdx.y*blockDim.y + threadIdx.y;\n"
               "    int gid = gidx*blockDim.x*gridDim.x+gidz;\n"
               "    int gsize=blockDim.x * gridDim.x*blockDim.y * gridDim.y;\n\n");
    }
    else if (dev->NDIM==3){
        strcat(temp,"    int gidz = blockIdx.x*blockDim.x + threadIdx.x;\n"
               "    int gidy = blockIdx.y*blockDim.y + threadIdx.y;\n"
               "    int gidx = blockIdx.y*blockDim.z + threadIdx.z;\n"
               "    int gid = gidx*blockDim.x*gridDim.x*blockDim.y*gridDim.y"
               "+gidy*blockDim.x*gridDim.x +gidz;\n"
               "    int gsize=blockDim.x * gridDim.x*blockDim.y * gridDim.y"
               "*blockDim.z * gridDim.z;\n\n");
    }
    
    for (i=0;i<dev->nvars;i++){
        if (vars[i].for_grad){
            if (dev->FP16==1){
                sprintf(ptemp,"    if (gid<%d)\n", vars[i].num_ele/2);
            }
            else{
                sprintf(ptemp,"    if (gid<%d)\n", vars[i].num_ele);
            }
            strcat(temp,ptemp);
            strcat(temp, "    ");
            strcat(temp, "    l");
            strcat(temp, vars[i].name);
            strcat(temp, "=");
            strcat(temp, vars[i].name);
            strcat(temp, "[gid];\n");
        }
    }

    
    strcat(temp,"\n"
        "    for (freq=0;freq<NFREQS;freq++){\n"
        "        fact[freq].x = DTNYQ*DT*cospi(2.0*gradfreqsn[freq]*nt/NTNYQ);\n"
        "        fact[freq].y = -DTNYQ*DT*sinpi(2.0*gradfreqsn[freq]*nt/NTNYQ);\n"
        "    }\n\n"
           );
        
        
    
    for (i=0;i<dev->nvars;i++){
        if (vars[i].for_grad){
            if (dev->FP16==1){
                sprintf(ptemp,"    if (gid<%d){\n", vars[i].num_ele/2);
            }
            else{
                sprintf(ptemp,"    if (gid<%d){\n", vars[i].num_ele);
            }
            
            strcat(temp,ptemp);
            strcat(temp, "    #pragma unroll\n");
            strcat(temp, "    for (freq=0;freq<NFREQS;freq++){\n");
            strcat(temp, "        f");
            strcat(temp, vars[i].name);
            strcat(temp, "[gid+freq*");
            sprintf(ptemp, "%d]+=fact[freq]*l", vars[i].num_ele);
            strcat(temp,ptemp);
            strcat(temp, vars[i].name);
            strcat(temp, ";\n");
            strcat(temp, "    }\n");
            strcat(temp,"    }\n");
        }
    }
    
    strcat(temp, "\n}");
    
    (*prog).src=temp;
    __GUARD prog_source(prog, "savefreqs", (*prog).src);
    
//    printf("%s\n\n%lu\n",temp, strlen(temp));
    
    if (dev->FP16==1){
        prog->gsize[0]=dev->N[0]/2;
    }
    else{
        prog->gsize[0]=dev->N[0];
    }
    prog->gsize[1]=dev->N[1];
    if (dev->NDIM==3){
        prog->gsize[2]=dev->N[2];
    }
    
    prog->wdim=dev->NDIM;
    
    return state;
    
}

int kernel_init_gradsrc(clprogram * prog){
    
    int state=0;
    
    char temp[MAX_KERN_STR]={0};;
    
    
    strcat(temp,
           "extern \"C\" __global__ void init_gradsrc(float *gradsrc, int nsrc)\n"
           "{\n\n"
           "    int gid = blockIdx.x*blockDim.x + threadIdx.x;\n"
           "    if (gid>nsrc*NT-1){\n"
           "        return;\n"
           "    }\n"
           "    gradsrc[gid]=0.0;\n\n"
           "}");
    
    (*prog).src=temp;
    
//    printf("%s\n\n%lu\n",temp, strlen(temp));
    
    __GUARD prog_source(prog, "init_gradsrc", (*prog).src);
    
    prog->wdim=1;
    
    return state;
    
}

int kernel_fcom_out(device * dev,
                    variable * vars,
                    clprogram * prog,
                    int upid, int buff12){
    
    int state=0;
    int i,j;
    
    char temp[MAX_KERN_STR]={0};;
    
    char * p=(char*)temp;
    char ptemp[200];
    
    int maxsize=0;
    for (i=0;i<dev->nvars;i++){
        if (vars[i].num_ele>maxsize){
            maxsize=vars[i].num_ele;
        }
    }
    
    strcat(temp, "extern \"C\" __global__ void fill_transfer_buff_out(");
    for (i=0;i<dev->nvars;i++){
        if (vars[i].to_comm==upid){
            strcat(temp, "float * ");
            strcat(temp, vars[i].name);
            strcat(temp, ", ");
            strcat(temp, "float * ");
            strcat(temp, vars[i].name);
            if (buff12==1)
                strcat(temp, "_buf1, ");
            else if (buff12==2)
                strcat(temp, "_buf2, ");
        }
    }
    while (*p)
        p++;
    p[-2]='\0';
    strcat(temp, "){\n\n");
    
    
    //Indice if local memory is used
    char * names[] = {"x","y","z"};
    if (dev->LOCAL_OFF==0){
        for (i=0;i<dev->NDIM;i++){
            sprintf(ptemp,"    int gid%s=blockIdx.%s*blockDim.%s + threadIdx.%s+FDOH;\n",
                            dev->N_names[i],names[i],names[i] ,names[i]  );
            strcat(temp, ptemp);
        }
    }
    //if we use directly global memory, with 1 working dim
    else{
        strcat(temp,"    int gid = blockIdx.x*blockDim.x + threadIdx.x;\n");

        sprintf(ptemp,"    int gid%s=gid%%(N%s-2*FDOH);\n",
                           dev->N_names[0],dev->N_names[0]);
        strcat(temp, ptemp);
        
        for (i=1;i<dev->NDIM;i++){
            sprintf(ptemp,"    int gid%s=(gid",dev->N_names[i]);
            strcat(temp, ptemp);
            for (j=0;j<i;j++){
                sprintf(ptemp,"/(N%s-2*FDOH)",dev->N_names[j]);
                strcat(temp, ptemp);
            }
            if (i<dev->NDIM-1){
                sprintf(ptemp,")%%(N%s-2*fdoh);\n",dev->N_names[i]);
                strcat(temp, ptemp);
            }
            else
                strcat(temp,");\n");
        }
        
        
    }
    strcat(temp,"\n");
    
    strcat(temp,"    int idbuf=");
    for (i=0;i<dev->NDIM;i++){
        sprintf(ptemp,"gid%s",dev->N_names[i]);
        strcat(temp, ptemp);
        for (j=0;j<i;j++){
            sprintf(ptemp,"*(N%s-2*FDOH)",dev->N_names[j]);
            strcat(temp, ptemp);
        }
        if (i!=dev->NDIM-1){
            strcat(temp, "+");
        }
    }
    char temp2[100]={0};
    if (buff12==1){
        sprintf(temp2,"    if (idbuf>%d-1){\n",
                (int)(vars[0].cl_buf1.size/sizeof(float)));
    }
    else{
        sprintf(temp2,"    if (idbuf>%d-1){\n",
                (int)(vars[0].cl_buf1.size/sizeof(float)));
    }
    strcat(temp,temp2);
    strcat(temp,  "        return;\n"
           "    };\n\n");
    
    strcat(temp,";\n");
    strcat(temp,"    int idvar=");
    for (i=0;i<dev->NDIM;i++){
        if (i!=dev->NDIM-1)
            sprintf(ptemp,"(gid%s+FDOH)",dev->N_names[i]);
        else{
            if (buff12==1)
                sprintf(ptemp,"(gid%s+FDOH)",dev->N_names[i]);
            else
                sprintf(ptemp,"(gid%s+N%s-2*FDOH)",
                              dev->N_names[i],dev->N_names[i]);
        }
        strcat(temp, ptemp);
        for (j=0;j<i;j++){
            sprintf(ptemp,"*N%s",dev->N_names[j]);
            strcat(temp, ptemp);
        }
        if (i!=dev->NDIM-1){
            strcat(temp, "+");
        }
    }
    strcat(temp,";\n\n");
    
    for (i=0;i<dev->nvars;i++){
        if (vars[i].to_comm==upid){
            sprintf(ptemp,"    %s_buf%d[idbuf]=%s[idvar];\n",
                          vars[i].name, buff12, vars[i].name);
            strcat(temp, ptemp);
        }
    }
    
    strcat(temp, "\n}");
    
    
//    printf("%s\n\n%lu\n",temp, strlen(temp));
    
    
    (*prog).src=temp;
    
    __GUARD prog_source(prog, "fill_transfer_buff_out", (*prog).src);
    
    return state;
    
}

int kernel_fcom_in(device * dev,
                   variable * vars,
                   clprogram * prog,
                   int upid, int buff12){
    
    int state=0;
    int i,j;
    
    char temp[MAX_KERN_STR]={0};;
    
    char * p=(char*)temp;
    char ptemp[200];
    
    int maxsize=0;
    for (i=0;i<dev->nvars;i++){
        if (vars[i].num_ele>maxsize){
            maxsize=vars[i].num_ele;
        }
    }
    
    strcat(temp, "extern \"C\" __global__ void fill_transfer_buff_in(");
    for (i=0;i<dev->nvars;i++){
        if (vars[i].to_comm==upid){
            strcat(temp, "float * ");
            strcat(temp, vars[i].name);
            strcat(temp, ", ");
            strcat(temp, "float * ");
            strcat(temp, vars[i].name);
            if (buff12==1)
                strcat(temp, "_buf1, ");
            else if (buff12==2)
                strcat(temp, "_buf2, ");
        }
    }
    while (*p)
        p++;
    p[-2]='\0';
    strcat(temp, "){\n\n");
    
    
    char * names[] = {"x","y","z"};
    if (dev->LOCAL_OFF==0){
        for (i=0;i<dev->NDIM;i++){
            sprintf(ptemp,"    int gid%s=blockIdx.%s*blockDim.%s + threadIdx.%s+FDOH;\n",
                    dev->N_names[i],names[i],names[i] ,names[i]  );
            strcat(temp, ptemp);
        }
    }
    //if we use directly global memory, with 1 working dim
    else{
        strcat(temp,"    int gid = blockIdx.x*blockDim.x + threadIdx.x;\n");
        
        sprintf(ptemp,"    int gid%s=gid%%(N%s-2*FDOH);\n",
                          dev->N_names[0],dev->N_names[0]);
        strcat(temp, ptemp);
        
        for (i=1;i<dev->NDIM;i++){
            sprintf(ptemp,"    int gid%s=(gid",dev->N_names[i]);
            strcat(temp, ptemp);
            for (j=0;j<i;j++){
                sprintf(ptemp,"/(N%s-2*FDOH)",dev->N_names[j]);
                strcat(temp, ptemp);
            }
            if (i<dev->NDIM-1){
                sprintf(ptemp,")%%(N%s-2*fdoh);\n",dev->N_names[i]);
                strcat(temp, ptemp);
            }
            else
                strcat(temp,");\n");
        }
        
        
    }
    strcat(temp,"\n");
    
    strcat(temp,"    int idbuf=");
    for (i=0;i<dev->NDIM;i++){
        sprintf(ptemp,"gid%s",dev->N_names[i]);
        strcat(temp, ptemp);
        for (j=0;j<i;j++){
            sprintf(ptemp,"*(N%s-2*FDOH)",dev->N_names[j]);
            strcat(temp, ptemp);
        }
        if (i!=dev->NDIM-1){
            strcat(temp, "+");
        }
    }
    strcat(temp,";\n");
    strcat(temp,"    int idvar=");
    for (i=0;i<dev->NDIM;i++){
        if (i!=dev->NDIM-1)
            sprintf(ptemp,"(gid%s+FDOH)",dev->N_names[i]);
        else{
            if (buff12==1)
                sprintf(ptemp,"(gid%s)",dev->N_names[i]);
            else
                sprintf(ptemp,"(gid%s+N%s-FDOH)",
                             dev->N_names[i],dev->N_names[i]);
        }
        strcat(temp, ptemp);
        for (j=0;j<i;j++){
            sprintf(ptemp,"*N%s",dev->N_names[j]);
            strcat(temp, ptemp);
        }
        if (i!=dev->NDIM-1){
            strcat(temp, "+");
        }
    }
    strcat(temp,";\n\n");
    
    char temp2[100]={0};
    if (buff12==1){
        sprintf(temp2,"    if (idbuf>%d-1){\n",
                (int)(vars[0].cl_buf1.size/sizeof(float)));
    }
    else{
        sprintf(temp2,"    if (idbuf>%d-1){\n",
                (int)(vars[0].cl_buf1.size/sizeof(float)));
    }
    strcat(temp,temp2);
    strcat(temp,  "        return;\n"
           "    };\n\n");
    
    
    for (i=0;i<dev->nvars;i++){
        if (vars[i].to_comm==upid){
            sprintf(ptemp,"    %s[idvar]=%s_buf%d[idbuf];\n",
                               vars[i].name, vars[i].name, buff12);
            strcat(temp, ptemp);
        }
    }
    
    strcat(temp, "\n}");
    
    
//        printf("%s\n\n%lu\n",temp, strlen(temp));
    
    
    (*prog).src=temp;
    
    __GUARD prog_source(prog, "fill_transfer_buff_in", (*prog).src);
    
    
    return state;
    
}



