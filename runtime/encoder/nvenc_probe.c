/* Open one encoder session on the only visible RTX 4070. No other GPU is used. */
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>
#include "nvEncodeAPI.h"
#define REQUIRE(call) do { int r=(call); if(r) { printf("%s: %d\n", #call, r); return r; } } while(0)
int main(void) {
    void *cuda=dlopen("libcuda.so.1",RTLD_NOW), *nv=dlopen("libnvidia-encode.so.1",RTLD_NOW);
    if(!cuda || !nv) { puts(dlerror()); return 90; }
    int (*init)(unsigned)=dlsym(cuda,"cuInit");
    int (*count)(int*)=dlsym(cuda,"cuDeviceGetCount");
    int (*name)(char*,int,int)=dlsym(cuda,"cuDeviceGetName");
    int (*uuid)(void*,int)=dlsym(cuda,"cuDeviceGetUuid");
    int (*ctxcreate)(void**,unsigned,int)=dlsym(cuda,"cuCtxCreate_v2");
    int (*ctxdestroy)(void*)=dlsym(cuda,"cuCtxDestroy_v2");
    NVENCSTATUS (*create)(NV_ENCODE_API_FUNCTION_LIST*)=dlsym(nv,"NvEncodeAPICreateInstance");
    if(!init || !count || !name || !uuid || !ctxcreate || !ctxdestroy || !create) return 91;
    int n=0; char gpu[128]={0}; unsigned char id[16];
    const unsigned char expected[16]={0xcd,0xe2,0x39,0x16,0xc5,0xfa,0xf1,0x8e,0xa3,0x43,0x12,0x7f,0x73,0x2c,0xc2,0xf6};
    REQUIRE(init(0)); REQUIRE(count(&n));
    if(n!=1) { printf("Refusing %d visible GPUs\n",n); return 92; }
    REQUIRE(uuid(id,0)); if(memcmp(id,expected,16)) return 93;
    REQUIRE(name(gpu,sizeof(gpu),0)); printf("GPU: %s\n",gpu);
    void *ctx=NULL,*encoder=NULL;
    REQUIRE(ctxcreate(&ctx,0,0));
    NV_ENCODE_API_FUNCTION_LIST api={0}; api.version=NV_ENCODE_API_FUNCTION_LIST_VER;
    REQUIRE(create(&api));
    NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS p={0};
    p.version=NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER; p.device=ctx;
    p.deviceType=NV_ENC_DEVICE_TYPE_CUDA; p.apiVersion=NVENCAPI_VERSION;
    NVENCSTATUS status=api.nvEncOpenEncodeSessionEx(&p,&encoder);
    printf("NvEncOpenEncodeSessionEx: %d\n",status);
    if(status==NV_ENC_SUCCESS) {
        unsigned codecs=0; REQUIRE(api.nvEncGetEncodeGUIDCount(encoder,&codecs));
        printf("Available codecs: %u\n",codecs);
        REQUIRE(api.nvEncDestroyEncoder(encoder));
    }
    REQUIRE(ctxdestroy(ctx));
    return status;
}
