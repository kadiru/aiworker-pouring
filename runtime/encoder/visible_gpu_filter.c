/* AIWorker-local workaround for NVIDIA's multi-GPU NVENC enumeration bug.
 * Only filters a successful user-space enumeration reply. It never changes the
 * kernel, device permissions, or driver files. Load ONLY in CloudXR's process.
 * NVIDIA API definitions: open-gpu-kernel-modules 580.95.05, ctrl0000gpu.h/nvos.h.
 * Diagnosis: NVIDIA/nvidia-container-toolkit#1249. No guessed PCI-ID mapping.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <unistd.h>

typedef struct {
    uint32_t client, object, command, flags;
    void *params __attribute__((aligned(8)));
    uint32_t size, status;
} Control;
typedef struct {
    uint32_t id, flags, device, subdevice, sli, board, instance;
    int32_t numa;
} IdInfoV2;
_Static_assert(sizeof(Control)==32, "Unexpected RM control ABI");
_Static_assert(sizeof(IdInfoV2)==32, "Unexpected RM GPU info ABI");
static int (*real_ioctl)(int,unsigned long,...);
static pthread_once_t once=PTHREAD_ONCE_INIT;
static int enabled;
static void initialize(void) {
    real_ioctl=dlsym(RTLD_NEXT,"ioctl");
    if(!real_ioctl) _exit(94);
    const char *value=getenv("AIWORKER_NVENC_GPU");
    enabled=value && !strcmp(value,"GPU-cde23916-c5fa-f18e-a343-127f732cc2f6");
    if(getenv("AIWORKER_NVENC_DEBUG")) fprintf(stderr,"AIWorker encoder: filter enabled=%d visible=%s\n",enabled,value?value:"unset");
}
int ioctl(int fd,unsigned long request,...) {
    pthread_once(&once,initialize);
    va_list args; va_start(args,request); void *argument=va_arg(args,void*); va_end(args);
    int result=real_ioctl(fd,request,argument), saved_errno=errno;
    if(!enabled || result || request!=_IOWR('F',0x2a,Control) || !argument) return result;
    Control *control=argument;
    if(control->command!=0x201 || control->status || control->size!=128 || !control->params) return result;
    struct stat st;
    if(fstat(fd,&st) || !S_ISCHR(st.st_mode) || major(st.st_rdev)!=195 || minor(st.st_rdev)!=255) {
        errno=saved_errno; return result;
    }
    uint32_t *ids=control->params, kept[32]; unsigned total=0, count=0;
    for(unsigned i=0;i<32;i++) kept[i]=UINT32_MAX;
    for(;total<32 && ids[total]!=UINT32_MAX;total++) {
        IdInfoV2 info={.id=ids[total]};
        Control query={.client=control->client,.object=control->client,.command=0x205,
                       .params=&info,.size=sizeof(info)};
        if(real_ioctl(fd,request,&query) || query.status) {
            if(getenv("AIWORKER_NVENC_DEBUG")) fprintf(stderr,"AIWorker encoder: GPU ID 0x%x mapping failed status=0x%x\n",info.id,query.status);
            /* Unknown mapping leaves the reply untouched; encoding may fail. */
            errno=saved_errno; return result;
        }
        char path[64]; snprintf(path,sizeof(path),"/dev/nvidia%u",info.device);
        if(getenv("AIWORKER_NVENC_DEBUG")) fprintf(stderr,"AIWorker encoder: GPU ID 0x%x maps to %s\n",info.id,path);
        if(!stat(path,&st) && S_ISCHR(st.st_mode) && major(st.st_rdev)==195 && minor(st.st_rdev)==info.device)
            kept[count++]=ids[total];
    }
    /* This project has exactly one assigned GPU. Never widen its device list. */
    if(count==1 && total>1) {
        memcpy(ids,kept,sizeof(kept));
        if(getenv("AIWORKER_NVENC_DEBUG"))
            fprintf(stderr,"AIWorker encoder: filtered %u attached GPU IDs to assigned ID 0x%x\n",total,kept[0]);
    }
    errno=saved_errno;
    return result;
}
