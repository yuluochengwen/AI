#include <stdio.h>

int main(){
    int var[] = {1, 3, 5, 7, 9};
    int *p, i;
    p = var;
    for (i = 0; i < 5; i++){
        printf("存储地址：var[%d] = %p\n",i ,p);
        printf("存储的值：var[%d] = %d\n", i, *p);
        p++;
    }
    return 0;
}