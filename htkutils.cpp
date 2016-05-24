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

extern "C" {
    typedef struct {
        int nsamples, sample_period;
        short samplesize, parmkind;
    } htkheader_t;

    float swapfloatendian( const float inFloat )
    {
       float retVal;
       char *floatToConvert = ( char* ) & inFloat;
       char *returnFloat = ( char* ) & retVal;

       // swap the bytes into a temporary buffer
       returnFloat[0] = floatToConvert[3];
       returnFloat[1] = floatToConvert[2];
       returnFloat[2] = floatToConvert[1];
       returnFloat[3] = floatToConvert[0];

       return retVal;
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
        inp.read((char*)header,sizeof(htkheader_t));
        inp.close();
        // Convert big endian to small endian
        header->nsamples = be32toh(header->nsamples);
        header->sample_period = be32toh(header->sample_period);
        header->samplesize = be16toh(header->samplesize);
        header->parmkind  = be16toh(header->parmkind);
        return 0;
    }

    int readhtkfile(const char* fname,THFloatTensor* output){
        // Input is the given filename and the output Float Tensor.
        htkheader_t header;
        readhtkheader(fname,&header);
        std::ifstream inp;
        try{
          inp.open(fname,std::ios::binary);
        }catch(std::ios_base::failure& e){
          std::string exept= "File cannot be opened !\n";
          inp.close();
          throw std::runtime_error(exept.c_str());
        }
        // We already read the input header, so need to skip the first 12 bytes.
        inp.seekg(12);

        int sample_bytes = header.samplesize/sizeof(char);
        int featdim = header.samplesize/sizeof(float);
        // the overall length of the output array
        int tlen = featdim*header.nsamples;
        float *storage = (float*) malloc(tlen*sizeof(float));
        std::vector<char> samplebuf(sample_bytes);
        auto row = 0;
        for ( auto i=0 ; i < header.nsamples; i++) {
            // Reading in the input data
            inp.read(reinterpret_cast<char*>(samplebuf.data()),sample_bytes);
            row = i * featdim;
            for(auto j = 0 ; j < sample_bytes ;j+=4){
                // Swapping the elements from big endian to little endian
                std::swap(samplebuf[j+3],samplebuf[j]);
                std::swap(samplebuf[j+2],samplebuf[j+1]);
                // Now copy the char bit array to a float
                float result=0;
                // Copying the little to big endian
                memcpy(&result, &samplebuf[j], sizeof(result));
                // Insert into the storage. i*featdim is the current row, j/4 is the current col
                storage[row+(j/4)] = result;
            }
        }
        inp.close();

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

            int sample_period = htobe32(header->sample_period);
            int nsamples = htobe32(header->nsamples);
            short samplesize = htobe16(header->samplesize);
            short parmkind = htobe16(header->parmkind);

            // std::cout << header->sample_period << " " << header->nsamples << " " << header->samplesize <<std::endl;

            float* tensordata = THFloatTensor_data(data);

        //     // Write out the header
            outputfile.write(reinterpret_cast<char*>(&nsamples),sizeof(int));
            outputfile.write(reinterpret_cast<char*>(&sample_period),sizeof(int));
            outputfile.write(reinterpret_cast<char*>(&samplesize),sizeof(short));
            outputfile.write(reinterpret_cast<char*>(&parmkind),sizeof(short));

            auto nElement = 1;
            for(auto d = 0; d < data->nDimension; d++)
                nElement *= data->size[d];

            for(auto i=0u; i < nElement; i++){
                float endianswapped = swapfloatendian(tensordata[i]);
                outputfile.write(reinterpret_cast<char*>(&endianswapped),sizeof(float));
            }
            outputfile.close();
        }

    }
}


