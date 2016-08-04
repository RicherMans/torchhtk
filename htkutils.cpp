#include <fstream>
#include <TH.h>
#include <luaT.h>
#include <string>
#include <stdexcept>
#include <sstream>
#include <iostream>
#include <vector>
#include <stdint.h>
#include <endian.h>
#include <ctime>
#include <algorithm>

template <class T>
inline void endswap_32b(T &objp)
{
  unsigned char *memp = reinterpret_cast<unsigned char*>(&objp);
  std::swap(memp[0],memp[3]);
  std::swap(memp[1],memp[2]);
}

extern "C" {
    typedef struct {
        int nsamples, sample_period;
        short samplesize, parmkind;
    } htkheader_t;


    int readhtkheaderopen(std::ifstream &input,htkheader_t *header){
        input.read((char*)header,sizeof(htkheader_t));
        header->nsamples = be32toh(header->nsamples);
        header->sample_period = be32toh(header->sample_period);
        header->samplesize = be16toh(header->samplesize);
        header->parmkind  = be16toh(header->parmkind);
        return 0;
    }

    int readhtkheader(const char *fname,htkheader_t *header){
        std::ifstream inp;
        try{
          inp.open(fname,std::ios::binary);
        }catch(std::ios_base::failure& e){
          std::string exept= "File cannot be opened !\n";
          inp.close();
          throw std::runtime_error(exept.c_str());
        }
        readhtkheaderopen(inp,header);
        inp.close();
        return 0;
    }

    // Only reads a single sample. Sample needs to be in range 1...N not in 0...N-1
    int readhtksample(const char* fname,int sample,THFloatTensor* output){
        htkheader_t header;
        std::ifstream inp;
        try{
          inp.open(fname,std::ios::binary);
        }catch(std::ios_base::failure& e){
          std::string exept= "File cannot be opened !\n";
          inp.close();
          throw std::runtime_error(exept.c_str());
        }
        readhtkheaderopen(inp,&header);
        // Feature dimension in char size ( featdim *4)
        int sample_bytes = header.samplesize/sizeof(char);
        // Actual feature dimension
        int featdim = header.samplesize/sizeof(float);
        // the overall length of the output array
        int tlen = featdim;
        // sample is out of range
        if (sample > header.nsamples){
            return 1;
        }
        bool reusebuffer = output->nDimension > 0;
        float *storage;
        // Passed output is an empty tensor
        if (! reusebuffer) {
            storage = (float*) malloc(tlen*sizeof(float));
        }
        // Passed output is an already allocated tensor- > reuse it
        else{
            storage = THFloatTensor_data(output);
        }
        // We already read the input header, so need to skip the first 12 bytes. Also we assume that sample is on range 1..N
        inp.seekg(12 + (sample_bytes * (sample - 1))  );

        // Read the file directly into the storage Insert into the storage. 
        inp.read(reinterpret_cast<char*>(storage),sample_bytes);
        inp.close();
        // Swapping the elements from big endian to little endian
        std::for_each(storage,storage+featdim,endswap_32b<float>);


        // Passed tensor is empty, thus we allocate a new one and return it
        if (! reusebuffer){
            // Allocate the outputstorage vector
            THFloatStorage* outputstorage  = THFloatStorage_newWithData(storage,tlen);
            if (outputstorage){
                // Set the strides
                long sizedata[1]   = { featdim };
                long stridedata[1] = { 1 };
                // Put the strides into the lua torch tensors
                THLongStorage* size    = THLongStorage_newWithData(sizedata, 1);
                THLongStorage* stride  = THLongStorage_newWithData(stridedata, 1);

                THFloatTensor_setStorage(output,outputstorage,0, size, stride);
                THFloatStorage_free(outputstorage);
                return 0;
            }
        }
        // Reuse tensor
        else{
            // Result is already stored in storage
            return 0;
        }

        return 1;
    }

    int readhtkfile(const char* fname,THFloatTensor* output){
        // Input is the given filename and the output Float Tensor.
        htkheader_t header;
        std::ifstream inp;
        try{
          inp.open(fname,std::ios::binary);
        }catch(std::ios_base::failure& e){
          std::string exept= "File cannot be opened !\n";
          inp.close();
          throw std::runtime_error(exept.c_str());
        }
        readhtkheaderopen(inp,&header);
        // We already read the input header, so need to skip the first 12 bytes.
        inp.seekg(12);

        int sample_bytes = header.samplesize/sizeof(char) * header.nsamples;
        int featdim = header.samplesize/sizeof(float);
        // the overall length of the output array
        int tlen = featdim*header.nsamples;
        float *storage = (float*) malloc(tlen*sizeof(float));
        inp.read(reinterpret_cast<char*>(storage),sample_bytes);
        inp.close();
        std::for_each(storage,storage+tlen,endswap_32b<float>);


        // Allocate the outputstorage vector
        THFloatStorage* outputstorage  = THFloatStorage_newWithData(storage,tlen);
        if (outputstorage){
            // Set the strides
            long sizedata[2]   = { header.nsamples,featdim };
            long stridedata[2] = { featdim, 1};
            // Put the strides into the lua torch tensors
            THLongStorage* size    = THLongStorage_newWithData(sizedata, 2);
            THLongStorage* stride  = THLongStorage_newWithData(stridedata, 2);

            THFloatTensor_setStorage(output,outputstorage,0, size, stride);
            THFloatStorage_free(outputstorage);
            return 0;
        }
        return 1;
    }

    int writehtkfile(const char* fname,htkheader_t *header,THFloatTensor* data){

        std::ofstream outputfile;
        outputfile.open(fname,std::ios::binary);
        if (outputfile){
            // Little to big endian swaps
            int sample_period = htobe32(header->sample_period);
            int nsamples = htobe32(header->nsamples);
            short samplesize = htobe16(header->samplesize);
            short parmkind = htobe16(header->parmkind);

            // std::cout << header->sample_period << " " << header->nsamples << " " << header->samplesize <<std::endl;

            float* tensordata = THFloatTensor_data(data);
            auto nelement = THFloatTensor_nElement(data);

        //     // Write out the header
            outputfile.write(reinterpret_cast<char*>(&nsamples),sizeof(int));
            outputfile.write(reinterpret_cast<char*>(&sample_period),sizeof(int));
            outputfile.write(reinterpret_cast<char*>(&samplesize),sizeof(short));
            outputfile.write(reinterpret_cast<char*>(&parmkind),sizeof(short));

            std::for_each(tensordata,tensordata+nelement,endswap_32b<float>);
            outputfile.write(reinterpret_cast<char*>(tensordata),sizeof(float)*nelement);
            outputfile.close();
        }

    }
}


